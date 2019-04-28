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

file_path ="train.csv"

groups = ['black','christian','female',
          'homosexual_gay_or_lesbian','jewish','male','muslim',
          'psychiatric_or_mental_illness','white']



# Now starts the evaluation of the model regarding bias

X_test = pd.read_csv('balanced_test.csv', sep = ',')
Y_test = pd.read_csv('balanced_test_Y.csv', sep = ',',usecols=['toxic'])

loaded_model = pickle.load(open('logistic_model.save', 'rb'))
vocabulary = pickle.load(open('vocabulary.save', 'rb'))
tokens = CountVectorizer(max_features=10000, lowercase=True, binary=True, vocabulary = vocabulary)

test = tokens.fit_transform(X_test['comment_text'])
pred = loaded_model.predict_proba(test)[:, 1]
auc = roc_auc_score(Y_test, pred)
print('Overall Test ROC AUC: %.3f' %auc)
print(loaded_model.score(test, Y_test['toxic']))
confusionMatrix = sklearn.metrics.confusion_matrix(Y_test, pred>0.5)
print(confusionMatrix)
print("Acceptance rate: %.3f" %(100*((confusionMatrix[0][0]+confusionMatrix[0][1])/len(Y_test))))

print('\n Analysis per group')

for g in groups:
    test = X_test[X_test[g] > 0.5]
    test.reset_index(inplace=True)
    features = tokens.fit_transform(test['comment_text'])
    y_test = X_test.loc[X_test[g] > 0.5,['toxic']]
    pred = loaded_model.predict_proba(features)[:, 1]
    auc = roc_auc_score(y_test, pred)
    print('\nTest ROC AUC for group %s: %.3f' %(g,auc))
    print(loaded_model.score(features, y_test['toxic']))
    print(sklearn.metrics.confusion_matrix(y_test, pred>0.5))
    confusionMatrix = sklearn.metrics.confusion_matrix(y_test, pred > 0.5)
    print("Acceptance rate: %.3f" % (100 * ((confusionMatrix[0][0] + confusionMatrix[0][1]) / len(y_test))))
    print('List of false positives')
    print(([ v['comment_text'] if (pred[i]>0.5 and v['toxic']==False) else '' for i,v in test.iterrows() ]))



print('\nUsing the jigsaw debias dataset \n')

# This dataset is one tool in evaluating our de-biasing efforts. For a given template, a large difference in
# model scores when single words are substituted may point to a bias problem. For example, if "I am a gay man" g
# ets a much higher score than "I am a tall man", this may indicate bias in the model.
#
# The madlibs dataset contains 89k examples generated from templates and word lists. The dataset is
# eval_datasets/bias_madlibs_89k.csv, a CSV consisting of 2 columns. The generated text is in Text, and
# the label is Label, either BAD or NOT_BAD.

debias_path = "/home/christian/GWU/Bias in AI/unintended-ml-bias-analysis-master/unintended_ml_bias/eval_datasets/bias_madlibs_77k.csv"
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
print(sklearn.metrics.confusion_matrix(Y_test, pred>0.5))

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
print(sklearn.metrics.confusion_matrix(Y_test, pred>0.5))

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
print(sklearn.metrics.confusion_matrix(Y_test, pred>0.5))

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
print(sklearn.metrics.confusion_matrix(Y_test, pred>0.5))


