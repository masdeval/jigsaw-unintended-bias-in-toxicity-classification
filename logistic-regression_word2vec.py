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


def createFeatures(X_train):

    import gensim.downloader as api
    # model = api.load("glove-twitter-200")
    glove = api.load(EMBEDDINGS)

    ######  TF-IDF #####
    # from sklearn.feature_extraction.text import TfidfVectorizer
    # tfidfVectorizer = TfidfVectorizer(encoding='latin-1', vocabulary=glove.wv.vocab.keys()
    #                                   ,lowercase=True
    #                                   ,tokenizer=gensim.utils.simple_preprocess)
    # tfidf = tfidfVectorizer.fit_transform(X_train.loc[:,'comment_text'])
    # #####################

    features = []
    # Creating a representation for the whole tweet using Glove wordvec
    for i,comment in enumerate(X_train['comment_text']):

      words = gensim.utils.simple_preprocess(comment)
      #words = stanfordPreprocessing.tokenize(word).split()
      #Without TF_IDF
      features.append(buildVector(words,glove,size=300))
      #With tfidf
      #features.append(buildVectorTFIDF(words, glove, tfidfVectorizer, tfidf.getrow(i).toarray(), size=300))

    del glove
    return features


def create_balanced_train_test(wiki_data):
    # Creating a balanced Test/Train for different identities
    X_train = X_test = Y_train = Y_test = None
    for x in groups:
        X_train_aux, X_test_aux, Y_train_aux, Y_test_aux = train_test_split(
            wiki_data.loc[wiki_data[x] > 0.5, ['comment_text']], wiki_data.loc[wiki_data[x] > 0.5, ['id', 'toxic']],
            test_size=0.3, random_state=666)
        X_train = pd.concat([X_train, X_train_aux])
        X_test = pd.concat([X_test, X_test_aux])
        Y_train = pd.concat([Y_train, Y_train_aux])
        Y_test = pd.concat([Y_test, Y_test_aux])

    X_train = sklearn.utils.shuffle(pd.concat([X_train, Y_train], axis=1))
    X_test = sklearn.utils.shuffle(pd.concat([X_test, Y_test], axis=1))
    Y_train = X_train.loc[:, ['id', 'toxic']]
    Y_test = X_test.loc[:, ['id', 'toxic']]
    X_train.drop(['id', 'toxic'], inplace=True, axis=1)
    X_test.drop(['id', 'toxic'], inplace=True, axis=1)
    X_train.to_csv('balanced_train.csv')
    Y_train.to_csv('balanced_train_Y.csv')
    X_test.to_csv('balanced_test.csv')
    Y_test.to_csv('balanced_test_Y.csv', header=['id', 'toxic'])

    return X_train, X_test, Y_train['toxic'], Y_test['toxic']

def trainModel(X_train , Y_train):
    model = LogisticRegressionCV(penalty='l2',max_iter=200,solver='lbfgs',n_jobs=2)
    model.fit(X_train,Y_train)
    # save the model to disk
    pickle.dump(model, open('logistic_model_word2vec.save', 'wb'))
    return model

def firstExecution():
    file_path ="train.csv"
    wiki_data = pd.read_csv(file_path, sep=',')
    wiki_data['toxic'] = wiki_data['target'] > 0.5

    if os.path.isfile('balanced_train.csv'):
        X_train = pd.read_csv('balanced_train.csv', sep=',', usecols=['comment_text'])
        X_test = pd.read_csv('balanced_test.csv', sep=',', usecols=['comment_text'])
        Y_train = pd.read_csv('balanced_train_Y.csv', sep=',', usecols=['toxic'])
        Y_test = pd.read_csv('balanced_test_Y.csv', sep=',', usecols=['toxic'])
    else:
        X_train, X_test, Y_train, Y_test = create_balanced_train_test(wiki_data)

    del wiki_data
    gc.collect()


    model = trainModel(createFeatures(X_train), Y_train)
    pred = model.predict_proba(createFeatures(X_test))[:, 1]
    auc = roc_auc_score(Y_test, pred)
    print('Test ROC AUC: %.3f' %auc) #Test ROC AUC: 0.828
    pickle.dump(pred, open('word2vec_prediction.save', 'wb'))

firstExecution()

# Simple evaluation
X_test = pd.read_csv('balanced_test.csv', sep = ',', usecols=['comment_text','target'])
Y_test = pd.read_csv('balanced_test_Y.csv', sep = ',', usecols=['toxic'])
pred = pickle.load(open('word2vec_prediction.save', 'rb'))
X_test = pd.concat([X_test,pd.DataFrame({'pred':pred})],axis=1)
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
