import pandas as pd
import urllib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import roc_auc_score
import numpy as np
import gensim
import sklearn
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import  LogisticRegressionCV
import os
import pickle
import gc
import scipy

def transformConfusionMatrix(matrix):
    TN = matrix[0][0]
    TP = matrix[1][1]
    matrix[0][0] = TP
    matrix[1][1] = TN
    return matrix

def compute_bnsp_auc(df, subgroup):
    """Computes the AUC of the within-subgroup positive examples and the background negative examples."""
    subgroup_positive_examples = df[(df[subgroup]>0.5) & (df['toxic']==True)] #df[df[subgroup] & df[label]]
    non_subgroup_negative_examples = df[(df[subgroup]<=0.5) & ~(df['toxic']==True)] #df[~df[subgroup] & ~df[label]]
    examples = pd.concat([subgroup_positive_examples,non_subgroup_negative_examples],axis=0)
    return roc_auc_score(examples['toxic'], examples['pred']) # compute_auc(examples[label], examples[model_name])

def compute_bpsn_auc(df, subgroup):
    """Computes the AUC of the within-subgroup negative examples and the background positive examples."""
    subgroup_negative_examples = df[(df[subgroup]>0.5) & ~(df['toxic'])]
    non_subgroup_positive_examples = df[(df[subgroup]<=0.5) & (df['toxic'])]
    examples = pd.concat([subgroup_negative_examples,non_subgroup_positive_examples],axis=0)
    return roc_auc_score(examples['toxic'], examples['pred']) # compute_auc(examples[label], examples[model_name])


def calculate_overall_auc(df):
    true_labels = df['toxic']
    predicted_labels = df['pred']
    return sklearn.metrics.roc_auc_score(true_labels, predicted_labels)


def power_mean(series, p):
    total = sum(np.power(series, p))
    return np.power(total / len(series), 1 / p)


def get_final_metric(bias_df, overall_auc, POWER=-5, OVERALL_MODEL_WEIGHT=0.25):
    bias_score = np.average([
        power_mean(bias_df['subgroup_auc'], POWER),
        power_mean(bias_df['bpsn_auc'], POWER),
        power_mean(bias_df['bnsp_auc'], POWER)
    ])
    return (OVERALL_MODEL_WEIGHT * overall_auc) + ((1 - OVERALL_MODEL_WEIGHT) * bias_score)

file_path ="train.csv"

groups = ['black','christian','female',
          'homosexual_gay_or_lesbian','jewish','male','muslim',
          'psychiatric_or_mental_illness','white']



# Now starts the evaluation of the model regarding bias

X_test = pd.read_csv('balanced_test.csv', sep = ',')
Y_test = pd.read_csv('balanced_test_Y.csv', sep = ',',usecols=['toxic'])
pred = pickle.load(open('baseline_predictions.save', 'rb'))
X_test = pd.concat([X_test,pd.DataFrame({'pred':pred})],axis=1)
X_test = pd.concat([X_test,Y_test],axis=1)


auc = roc_auc_score(Y_test, pred)
print('Overall Test ROC AUC: %.3f' %auc)
print(sklearn.metrics.accuracy_score(Y_test, pred>0.5))
confusionMatrix = sklearn.metrics.confusion_matrix(Y_test, pred>0.5)
confusionMatrix = transformConfusionMatrix(confusionMatrix)
print(confusionMatrix)
print("Acceptance rate: %.3f" %(100*((confusionMatrix[0][0]+confusionMatrix[0][1])/len(Y_test))))
print("TPR: %.3f" % (100 * (confusionMatrix[0][0] / (confusionMatrix[0][0] + confusionMatrix[1][0]))))
print("TNR: %.3f" % (100 * (confusionMatrix[1][1] / (confusionMatrix[1][1] + confusionMatrix[0][1]))))

print('\n Analysis per group')

records = []

for g in groups:
    record = {}
    record['subgroup'] = g
    test = X_test[X_test[g] > 0.5]
    test.reset_index(inplace=True)
    record['subgroup_size'] = len(test.index)
    pred = test['pred']
    y_test = test['toxic']
    # features = tokens.fit_transform(test['comment_text'])'
    # y_test = X_test.loc[X_test[g] > 0.5,['toxic']]
    # pred = loaded_model.predict_proba(features)[:, 1]
    auc = roc_auc_score(test['toxic'], pred)
    record['subgroup_auc'] = auc
    print('\nTest ROC AUC for group %s: %.3f' %(g,auc))
    print(sklearn.metrics.accuracy_score(y_test, pred>0.5))
    confusionMatrix = sklearn.metrics.confusion_matrix(y_test, pred > 0.5)
    confusionMatrix= transformConfusionMatrix(confusionMatrix)
    print(confusionMatrix)
    print("Acceptance rate: %.3f" % (100 * ((confusionMatrix[0][0] + confusionMatrix[0][1]) / len(y_test))))
    print("TPR: %.3f" % (100 * (confusionMatrix[0][0] / ( confusionMatrix[0][0] + confusionMatrix[1][0]) )))
    print("TNR: %.3f" % (100 * (confusionMatrix[1][1] / ( confusionMatrix[1][1] + confusionMatrix[0][1]) )))
    FNR = (100 * (confusionMatrix[1][0] / (confusionMatrix[1][0] + confusionMatrix[0][0])))
    BNSP = compute_bnsp_auc(X_test,g)
    print("BNSP: %.3f - FNR: %.3f" % (BNSP,FNR))
    record['bnsp_auc' ] = BNSP
    FPR = (100 * (confusionMatrix[0][1] / ( confusionMatrix[0][1] + confusionMatrix[1][1]) ))
    BPSN = compute_bpsn_auc(X_test,g)
    print("BPSN: %.3f - FPR: %.3f" % (BPSN,FPR))
    record['bpsn_auc'] = BPSN
    print('List of false positives')
    #print(([ v['comment_text'] if (pred[i]>0.5 and v['toxic']==False) else '' for i,v in test.iterrows() ]))
    records.append(record)

bias_metrics_df = pd.DataFrame(records).sort_values('subgroup_auc', ascending=True)
#print(bias_metrics_df)
print("\n Model accuracy: %.3f" % get_final_metric(bias_metrics_df, calculate_overall_auc(X_test)))


loaded_model = pickle.load(open('logistic_model.save', 'rb'))


print('\nUsing the jigsaw debias dataset \n')

# This dataset is one tool in evaluating our de-biasing efforts. For a given template, a large difference in
# model scores when single words are substituted may point to a bias problem. For example, if "I am a gay man" g
# ets a much higher score than "I am a tall man", this may indicate bias in the model.
#
# The madlibs dataset contains 89k examples generated from templates and word lists. The dataset is
# eval_datasets/bias_madlibs_89k.csv, a CSV consisting of 2 columns. The generated text is in Text, and
# the label is Label, either BAD or NOT_BAD.

debias_path = "/home/christian/GWU/Bias in AI/unintended-ml-bias-analysis-master/unintended_ml_bias/eval_datasets/bias_madlibs_89k.csv"
debias_test = pd.read_csv(debias_path)
debias_test['Label'] = debias_test['Label'].apply(lambda x: 1 if x == 'BAD' else 0)
vocabulary = pickle.load(open('vocabulary.save', 'rb'))
tokens = CountVectorizer(max_features=10000, lowercase=True, binary=True, vocabulary = vocabulary)
X_test = tokens.fit_transform(debias_test['Text'])
Y_test = debias_test['Label']
pred = loaded_model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(Y_test, pred)
print('Global Test ROC AUC: %.3f' %auc)
print('Accuracy ' + str(loaded_model.score(X_test, Y_test)))
print(transformConfusionMatrix(sklearn.metrics.confusion_matrix(Y_test, pred>0.5)))

print('\nUsing the fuzzy dataset \n')

# This technique involves modifying a test set by "fuzzing" over a set of identity terms in order to evaluate a model
# for bias. Given a test set and a set of terms, we replace all instances of each term in the test data with a
# random other term from the set. The idea is that the specific term used for each example should not be the key
# feature in determining the label for the example. For example, the sentence "I had a friend growing up" should
# be considered non-toxic, and "All people must be wiped off the earth" should be considered toxic for all values
# of x in the terms set.
#
# The code in src/Bias_fuzzed_test_set.ipynb reads the Wikipedia Toxicity dataset and builds an identity-term-focused
# test set. It writes unmodified and fuzzed versions of that test set. One can then evaluate a model on both test sets
# . Doing significantly worse on the fuzzed version may indicate a bias in the model. The datasets are
# eval_datasets/toxicity_fuzzed_testset.csv and eval_datasets/toxicity_nonfuzzed_testset.csv. Each CSV consists of
# 3 columns: ID unedr rev_id, the comment text under comment, and the True/False label under toxic.
#
# This is similar to the bias madlibs technique, but has the advantage of using more realistic data.
# One can also use the model's performance on the original vs. fuzzed test set as a bias metric.

print('First the fuzzed \n')
debias_path = "/home/christian/GWU/Bias in AI/unintended-ml-bias-analysis-master/unintended_ml_bias/eval_datasets/toxicity_fuzzed_testset.csv"
debias_test = pd.read_csv(debias_path)
#debias_test['toxic'] = debias_test['toxic'].apply(lambda x: 1 if x == 'True' else 0)
vocabulary = pickle.load(open('vocabulary.save', 'rb'))
tokens = CountVectorizer(max_features=10000, lowercase=True, binary=True, vocabulary = vocabulary)
X_test = tokens.fit_transform(debias_test['comment'])
Y_test = debias_test['toxic']
pred = loaded_model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(Y_test, pred)
print('Test ROC AUC: %.3f' %auc)
print(loaded_model.score(X_test, Y_test))
print(transformConfusionMatrix(sklearn.metrics.confusion_matrix(Y_test, pred>0.5)))

print('\n Now the non-fuzzed test set \n')
debias_path = "/home/christian/GWU/Bias in AI/unintended-ml-bias-analysis-master/unintended_ml_bias/eval_datasets/toxicity_nonfuzzed_testset.csv"
debias_test = pd.read_csv(debias_path)
#debias_test['toxic'] = debias_test['toxic'].apply(lambda x: 1 if x == 'True' else 0)
vocabulary = pickle.load(open('vocabulary.save', 'rb'))
tokens = CountVectorizer(max_features=10000, lowercase=True, binary=True, vocabulary = vocabulary)
X_test = tokens.fit_transform(debias_test['comment'])
Y_test = debias_test['toxic']
pred = loaded_model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(Y_test, pred)
print('Test ROC AUC: %.3f' %auc)
print(loaded_model.score(X_test, Y_test))
print(transformConfusionMatrix(sklearn.metrics.confusion_matrix(Y_test, pred>0.5)))

print('\n Wiki Toxicity test set \n')
debias_path = "/home/christian/GWU/Bias in AI/unintended-ml-bias-analysis-master/data/wiki_toxicity_test.csv"
debias_test = pd.read_csv(debias_path)
#debias_test['toxic'] = debias_test['toxic'].apply(lambda x: 1 if x == 'True' else 0)
vocabulary = pickle.load(open('vocabulary.save', 'rb'))
tokens = CountVectorizer(max_features=10000, lowercase=True, binary=True, vocabulary = vocabulary)
X_test = tokens.fit_transform(debias_test['comment'])
Y_test = debias_test['is_toxic']
pred = loaded_model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(Y_test, pred)
print('Test ROC AUC: %.3f' %auc)
print(loaded_model.score(X_test, Y_test))
print(transformConfusionMatrix(sklearn.metrics.confusion_matrix(Y_test, pred>0.5)))


