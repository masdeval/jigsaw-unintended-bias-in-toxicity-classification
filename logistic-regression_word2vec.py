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
from sklearn.linear_model import  SGDClassifier
import gc
import pickle
import gensim.downloader as api
import random
from sklearn.externals import joblib

#EMBEDDINGS = 'conceptnet-numberbatch-17-06-300'
EMBEDDINGS = 'glove-wiki-gigaword-200'
#EMBEDDINGS = 'glove-wiki-gigaword-300'


groups = ['black','christian','female',
          'homosexual_gay_or_lesbian','jewish','male','muslim',
          'psychiatric_or_mental_illness','white']



def readWordvec(file, kv = True):
    if kv == True:
        return KeyedVectors.load(file, mmap='r')
    else:
        return Word2Vec.load(file)

def buildVector(tokens, word2vec, size=150):

    vec = np.zeros(size)
    count = 0.
    for word in tokens:
        try:
            vec += word2vec[word]
            count += 1.
        except KeyError: # handling the case where the token is not present
            #print("\nWord not found : " + word)
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


TF_IDF = False


def createFeatures(X_train, X_test,embedding,size):

  if TF_IDF:
    ######  TF-IDF #####
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidfVectorizer = TfidfVectorizer(encoding='latin-1', vocabulary=glove.wv.vocab.keys()
                                     ,lowercase=True
                                     ,tokenizer=gensim.utils.simple_preprocess)
    tfidf = tfidfVectorizer.fit_transform(X_train.loc[:,'comment_text'])
    # #####################
  features_train = []
  # Creating a representation for the whole tweet
  for i,comment in enumerate(X_train['comment_text']):

     words = gensim.utils.simple_preprocess(comment)
     # words = stanfordPreprocessing.tokenize(word).split()

     if TF_IDF:
      #With tfidf
      features_train.append(buildVectorTFIDF(words, embedding, tfidfVectorizer, tfidf.getrow(i).toarray(), size))
     else:
      #Without TF_IDF
      features_train.append(buildVector(words,embedding,size))

  joblib.dump(features_train, open('word2vec_features_train.save', 'wb'))
  del features_train
  gc.collect()

  features_test = []
  for i, comment in enumerate(X_test['comment_text']):
      words = gensim.utils.simple_preprocess(comment)
      # words = stanfordPreprocessing.tokenize(word).split()

      if TF_IDF:
         # With tfidf
         features_test.append(buildVectorTFIDF(words, embedding, tfidfVectorizer, tfidf.getrow(i).toarray(), size))
      else:
         # Without TF_IDF
         features_test.append(buildVector(words, embedding, size))

  joblib.dump(features_test, open('word2vec_features_test.save', 'wb'))




def create_balanced_train_test(wiki_data):

    # Creating a balanced Test/Train for different identities
    X_train = X_test = Y_train = Y_test = None
    train = test = set()
    for x in groups:
        X_train_aux, X_test_aux, Y_train_aux, Y_test_aux = train_test_split(
             wiki_data.loc[wiki_data[x] > 0.5, ['id','target','comment_text']+groups], wiki_data.loc[wiki_data[x] > 0.5, ['toxic']],
             test_size=0.3, random_state=666)
        train = train.union(set(X_train_aux.index))
        test = test.union(set(X_test_aux.index))
        # X_train = pd.concat([X_train, X_train_aux])
        # X_test = pd.concat([X_test, X_test_aux])
        # Y_train = pd.concat([Y_train, Y_train_aux])
        # Y_test = pd.concat([Y_test, Y_test_aux])

    # This way do not work. Missing data?
    # index = set()
    # for x in groups:
    #     index = wiki_data[wiki_data[x] <= 0.5].index
    #     rows = rows.union(set(index))
    index = wiki_data[~wiki_data.index.isin(list(train.union(test)))].index

    X_train_aux, X_test_aux, Y_train_aux, Y_test_aux = train_test_split(
            wiki_data.loc[index , ['id','target','comment_text']+groups],
            wiki_data.loc[index, ['toxic']],
            test_size=0.3, random_state=666)
    train = train.union(set(X_train_aux.index))
    test = test.union(set(X_test_aux.index))
    # X_train = pd.concat([X_train, X_train_aux])
    # X_test = pd.concat([X_test, X_test_aux])
    # Y_train = pd.concat([Y_train, Y_train_aux])
    # Y_test = pd.concat([Y_test, Y_test_aux])

    #X_train = sklearn.utils.shuffle(pd.concat([X_train, Y_train], axis=1))
    #train = random.sample(train, int(len(train)*.6)) # get only 60% of the train
    X_train = wiki_data.loc[train,['id','target','comment_text']+groups]
    #X_test = sklearn.utils.shuffle(pd.concat([X_test, Y_test], axis=1))
    X_test = wiki_data.loc[test,['id','target','comment_text']+groups]
    #Y_train = X_train.loc[:, ['id', 'toxic']]
    Y_train = wiki_data.loc[train,'toxic']
    #Y_test = X_test.loc[:, ['id', 'toxic']]
    Y_test = wiki_data.loc[test,'toxic']
    #X_train.drop(['id', 'toxic'], inplace=True, axis=1)
    #X_test.drop(['id', 'toxic'], inplace=True, axis=1)

    X_train.to_csv('balanced_train.csv')
    Y_train.to_csv('balanced_train_Y.csv', header=['toxic'])
    X_test.to_csv('balanced_test.csv')
    Y_test.to_csv('balanced_test_Y.csv', header=['toxic'])

    return X_train, X_test, Y_train, Y_test

def trainModel(X_train , Y_train):
    model = None
    try:
      model = pickle.load(open('logistic_model_word2vec.save', 'rb'))
    except:
      None

    if model != None:
      model.partial_fit(X_train, Y_train)
      # save the model to disk
      pickle.dump(model, open('logistic_model_word2vec.save', 'wb'), protocol=2)
    else:
      # loss hinge = SVM
      # loss log = LogisticRegression
      model = SGDClassifier(loss='log',penalty='l2',n_jobs=2)
      model.fit(X_train,Y_train)
      # save the model to disk
      pickle.dump(model, open('logistic_model_word2vec.save', 'wb'), protocol=2)

    return model

def firstExecution(epochs):

    features_train = None
    features_test = None
    try:
        features_train = joblib.load(open('word2vec_features_train.save', 'rb'))
        features_test = joblib.load(open('word2vec_features_test.save', 'rb'))
        Y_train = pd.read_csv('balanced_train_Y.csv', sep=',', usecols=['toxic'])
        Y_test = pd.read_csv('balanced_test_Y.csv', sep=',', usecols=['toxic'])
    except:
        None

    if features_train == None:
      if os.path.isfile('balanced_train.csv'):
            X_train = pd.read_csv('balanced_train.csv', sep=',', usecols=['comment_text'])
            X_test = pd.read_csv('balanced_test.csv', sep=',', usecols=['comment_text'])
            Y_train = pd.read_csv('balanced_train_Y.csv', sep=',', usecols=['toxic'])
            Y_test = pd.read_csv('balanced_test_Y.csv', sep=',', usecols=['toxic'])
      else:
            file_path = "train.csv"
            wiki_data = pd.read_csv(file_path, sep=',')
            wiki_data['toxic'] = wiki_data['target'] > 0.5
            X_train, X_test, Y_train, Y_test = create_balanced_train_test(wiki_data)
            del wiki_data
            gc.collect()

      embedding = api.load(EMBEDDINGS)
      createFeatures(X_train,X_test,embedding,200)
      del embedding
      del X_train
      del X_test
      gc.collect()

    features_train = joblib.load(open('word2vec_features_train.save', 'rb'))
    features_test = joblib.load(open('word2vec_features_test.save', 'rb'))

    for i in range(epochs):
     model = trainModel(features_train, Y_train)

    pred = model.predict_proba(features_test)[:, 1]
    auc = roc_auc_score(Y_test, pred)
    print('Test ROC AUC: %.3f' %auc) #Test ROC AUC: 0.828
    pickle.dump(pred, open('word2vec_prediction.save', 'wb'), protocol=2)

firstExecution(1)

# Simple evaluation
Y_test = pd.read_csv('balanced_test_Y.csv', sep = ',', usecols=['toxic'])
pred = pickle.load(open('word2vec_prediction.save', 'rb'))
auc = roc_auc_score(Y_test, pred)
print('Overall Test ROC AUC: %.3f' %auc)
print(sklearn.metrics.accuracy_score(Y_test, pred>0.5))
confusionMatrix = sklearn.metrics.confusion_matrix(Y_test, pred>0.5)
print(confusionMatrix)
print("Acceptance rate: %.3f" %(100*((confusionMatrix[0][0]+confusionMatrix[0][1])/len(Y_test))))
print("TPR: %.3f" % (100 * (confusionMatrix[0][0] / (confusionMatrix[0][0] + confusionMatrix[1][0]))))
print("TNR: %.3f" % (100 * (confusionMatrix[1][1] / (confusionMatrix[1][1] + confusionMatrix[0][1]))))









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
