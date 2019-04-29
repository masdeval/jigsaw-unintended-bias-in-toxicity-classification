import numpy as np
import gensim
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from nltk.tokenize import TweetTokenizer
from sklearn.model_selection import cross_validate
import pandas as pd
import os
import preprocess_twitter as stanfordPreprocessing
from sklearn.metrics import roc_auc_score
import sklearn
from sklearn.linear_model import  LogisticRegressionCV
import gc
import pickle
import gensim.downloader as api



groups = ['black','christian','female',
          'homosexual_gay_or_lesbian','jewish','male','muslim',
          'psychiatric_or_mental_illness','white']

def compute_bnsp_auc(df, subgroup):
    """Computes the AUC of the within-subgroup positive examples and the background negative examples."""
    # error here means False Negative for subgroup and False Positive for others
    subgroup_positive_examples = df[(df[subgroup]>0.5) & (df['toxic']==True)] #df[df[subgroup] & df[label]]
    non_subgroup_negative_examples = df[(df[subgroup]<=0.5) & ~(df['toxic']==True)] #df[~df[subgroup] & ~df[label]]
    examples = pd.concat([subgroup_positive_examples,non_subgroup_negative_examples],axis=0)
    return roc_auc_score(examples['toxic'], examples['pred']) # compute_auc(examples[label], examples[model_name])

def compute_bpsn_auc(df, subgroup):
    """Computes the AUC of the within-subgroup negative examples and the background positive examples."""
    # error here means False Positive for subgroup and False Negative for others
    subgroup_negative_examples = df[(df[subgroup]>0.5) & ~(df['toxic'])]
    non_subgroup_positive_examples = df[(df[subgroup]<=0.5) & (df['toxic'])]
    examples = pd.concat([subgroup_negative_examples,non_subgroup_positive_examples],axis=0)
    return roc_auc_score(examples['toxic'], examples['pred']) # compute_auc(examples[label], examples[model_name])

def identityDetails(comment,target_group,word2vec):

    identity_detail['name'] = target_group
    identity_detail['comment_length'] = len(comment)

    similarities = list()
    similarity = 1.0
    normalized_identity = gensim.matutils.unitvec(word2vec[target_group]).astype(np.float32)
    for x in gensim.utils.simple_preprocess(comment):
        try:
          similarity = (normalized_identity@gensim.matutils.unitvec(word2vec[x]).astype(np.float32))
        except:
          None
        if similarity >= 1.0:
            continue
        else:
            similarities.append(similarity)

    identity_detail['number_embeddings'] = len(similarities)
    identity_detail['mean_similarity'] = np.array(similarities).mean()
    return  identity_detail

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


# embeddings = api.load(EMBEDDINGS)
# toxic_frame = wiki_data[(wiki_data['toxic']==False) & (wiki_data['black'] > 0.5)]
# for index, sample in toxic_frame.iterrows():
#   #for g in groups:
#       #if sample.loc['black'] > 0.5:
#           print('\n'+str(identityDetails(sample.loc['comment_text'],'black',embeddings)))
# del embeddings
# gc.collect()


# Now starts the evaluation of the model regarding bias

X_test = pd.read_csv('balanced_test.csv', sep = ',')
Y_test = pd.read_csv('balanced_test_Y.csv', sep = ',',usecols=['toxic'])
pred = pickle.load(open('word2vec_prediction.save', 'rb'))
X_test = pd.concat([X_test,pd.DataFrame({'pred':pred})],axis=1)

auc = roc_auc_score(Y_test, pred)
print('Test ROC AUC: %.3f' %auc)
print(sklearn.metrics.accuracy_score(Y_test, pred>0.5))
confusionMatrix = sklearn.metrics.confusion_matrix(Y_test, pred>0.5)
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
    # features = createFeatures(test['comment_text'],glove)
    # y_test = X_test.loc[X_test[g] > 0.5,['toxic']]
    # pred = loaded_model.predict_proba(features)[:, 1]
    auc = roc_auc_score(test['toxic'], pred)
    record['subgroup_auc'] = auc
    print('\nTest ROC AUC for group %s: %.3f' %(g,auc))
    print(sklearn.metrics.accuracy_score(y_test, pred>0.5))
    print(sklearn.metrics.confusion_matrix(y_test, pred>0.5))
    confusionMatrix = sklearn.metrics.confusion_matrix(y_test, pred > 0.5)
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


#EMBEDDINGS = 'conceptnet-numberbatch-17-06-300'
EMBEDDINGS = 'glove-wiki-gigaword-300'

import gensim.downloader as api
glove = api.load(EMBEDDINGS)

identity_detail = {

    'name': str,
    'comment_length': int,
    'number_embeddings': int,
    'mean_similarity': np.float32
}


loaded_model = pickle.load(open('logistic_model.save', 'rb'))

groups = ['black','christian','female',
          'homosexual', 'gay', 'lesbian','jewish','male','muslim',
          'white']

for w in groups:
  print('\n' + w + ' : ' + str(loaded_model.predict_proba(glove[gensim.utils.simple_preprocess(w)[0]].reshape(1,-1))[:,1]))
















# result = cross_validate(LogisticRegression(penalty='l2'),X=features,y=sentiment,cv=5,scoring=['accuracy','f1'], return_train_score=False)
#
# from prettytable import PrettyTable
# print("\n Logistic in train")
# x = PrettyTable()
# x.field_names = [" ","Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"]
# x.add_row(["Accuracy: "] + [str(v) for v in result['test_accuracy']])
# x.add_row(["F1: "] + [str(v) for v in result['test_f1']])
# print(x)
# print("Overall accuracy: %f" % np.mean(result['test_accuracy']))
# print("Overall F1-score: %f" % np.mean(result['test_f1']))
#
# result = cross_validate(svm.SVC(C=1.0,kernel='linear'),X=features,y=sentiment,cv=5,scoring=['accuracy','f1'], return_train_score=False)
#
# from prettytable import PrettyTable
# print("\n SVM in train")
# x = PrettyTable()
# x.field_names = [" ","Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"]
# x.add_row(["Accuracy: "] + [str(v) for v in result['test_accuracy']])
# x.add_row(["F1: "] + [str(v) for v in result['test_f1']])
# print(x)
# print("Overall accuracy: %f" % np.mean(result['test_accuracy']))
# print("Overall F1-score: %f" % np.mean(result['test_f1']))
#
#
#
