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

groups = ['black','christian','female',
          'homosexual_gay_or_lesbian','jewish','male','muslim',
          'psychiatric_or_mental_illness','white']


def createFeatures(X_train,glove):

    #import gensim.downloader as api
    # model = api.load("glove-twitter-200")
    #glove = api.load(EMBEDDINGS)

    ######  TF-IDF #####
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidfVectorizer = TfidfVectorizer(encoding='latin-1', vocabulary=glove.wv.vocab.keys()
                                      ,lowercase=True
                                      ,tokenizer=gensim.utils.simple_preprocess)
    tfidf = tfidfVectorizer.fit_transform(X_train)
    #####################

    features = []
    # Creating a representation for the whole tweet using Glove wordvec
    for i,comment in enumerate(X_train):

      words = gensim.utils.simple_preprocess(comment)
      #words = stanfordPreprocessing.tokenize(word).split()
      #Without TF_IDF
      #features.append(buildVector(words,glove,size=300))

      features.append(buildVectorTFIDF(words, glove, tfidfVectorizer, tfidf.getrow(i).toarray(), size=300))

    #del glove
    return features

def buildVector(tokens, word2vec, size=150):

    vec = np.zeros(size)
    count = 0.
    for word in tokens:
        try:
            vec += word2vec[word]
            count += 1.
        except KeyError: # handling the case where the token is not present
            print("\nWord not found : " + word)
            continue
    if count != 0:
        vec /= count

    assert(len(vec) == size)
    return vec

def buildVectorTFIDF(tokens, word2vec, tfidfVectorizer, tfidf, size=150):
    vec = np.zeros(size)
    count = 0.
    for word in tokens:
        try:
            vec += word2vec[word] * tfidf[0,tfidfVectorizer.vocabulary_[word]]
            count += 1.
        except KeyError: # handling the case where the token is not present
            continue
    if count != 0:
        vec /= count

    assert(len(vec) == size)
    return vec



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

loaded_model = pickle.load(open('logistic_model_word2vec.save', 'rb'))
features = createFeatures(X_test['comment_text'],glove)
pred = loaded_model.predict_proba(features)[:, 1]
auc = roc_auc_score(Y_test, pred)
print('Test ROC AUC: %.3f' %auc)
print(loaded_model.score(features, Y_test))
confusionMatrix = sklearn.metrics.confusion_matrix(Y_test, pred>0.5)
print(confusionMatrix)
print("Acceptance rate: %.3f" %(100*((confusionMatrix[0][0]+confusionMatrix[0][1])/len(Y_test))))


print('\n Analysis per group')

for g in groups:
    test = X_test[X_test[g] > 0.5]
    test.reset_index(inplace=True)
    features = createFeatures(test['comment_text'],glove)
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
