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
from collections import defaultdict
import dill
import copy
import json

#EMBEDDINGS = 'conceptnet-numberbatch-17-06-300'
EMBEDDINGS = 'glove-wiki-gigaword-200'
#EMBEDDINGS = 'glove-wiki-gigaword-300'


groups = ['black','christian','female','homosexual_gay_or_lesbian','jewish','male','muslim',
          'psychiatric_or_mental_illness','white']

words_pleasent = ['freedom', 'health', 'peace', 'cheer', 'gentle', 'gift', 'honor', 'miracle',
                   'sunrise','christian']
# #
words_unpleasent = ['honest', 'filth', 'poison', 'stink','ugly','evil', 'kill', 'rotten', 'vomit', 'negative',
                'bad']

words_nontoxic = ['christian','jewish']
#
words_toxic = ['black','female','male','gay','homosexual','lesbian','muslim','white']

def createVector(model, words):
    #words = ['positive']
    result = list()
    for w in words:
        result.append(model.get_vector(w))

    #return matutils.unitvec(np.array(result).mean(axis=0)).astype(np.float32)
    return (np.array(result).mean(axis=0)).astype(np.float32)

#from gensim.models.keyedvectors import KeyedVectors
#embedding = KeyedVectors.load_word2vec_format("/home/christian/gensim-data/glove-wiki-gigaword-200/glove-wiki-gigaword-200", binary=False)

embedding = api.load(EMBEDDINGS)

nontoxicVector = createVector(embedding, words_nontoxic)
nontoxicVector_ = gensim.matutils.unitvec(nontoxicVector).astype(np.float32)
toxicVector = createVector(embedding, words_toxic)
toxicVector_ = gensim.matutils.unitvec(toxicVector).astype(np.float32)
pleasentVector = createVector(embedding, words_pleasent)
pleasentVector_ = gensim.matutils.unitvec(pleasentVector).astype(np.float32)
unpleasentVector = createVector(embedding, words_unpleasent)
unpleasentVector_ = gensim.matutils.unitvec(unpleasentVector).astype(np.float32)

## Debiasing
# weight_PLEASENT = 0.2
# weight_UNPLEASENT = 0.5
BLACK = embedding['black']
BLACK_ = gensim.matutils.unitvec(BLACK).astype(np.float32)
FEMALE = embedding['female']
FEMALE_ = gensim.matutils.unitvec(FEMALE).astype(np.float32)
HOMOSEXUAL = embedding['homosexual']
HOMOSEXUAL_ = gensim.matutils.unitvec(HOMOSEXUAL).astype(np.float32)
GAY = embedding['gay']
GAY_ = gensim.matutils.unitvec(GAY).astype(np.float32)
LESBIAN = embedding['lesbian']
LESBIAN_ = gensim.matutils.unitvec(LESBIAN).astype(np.float32)
MALE = embedding['male']
MALE_ = gensim.matutils.unitvec(MALE).astype(np.float32)
MUSLIM = embedding['muslim']
MUSLIM_ = gensim.matutils.unitvec(MUSLIM).astype(np.float32)
WHITE = embedding['white']
WHITE_ = gensim.matutils.unitvec(MUSLIM).astype(np.float32)
CHRISTIAN = embedding['christian']
CHRISTIAN_ = gensim.matutils.unitvec(CHRISTIAN).astype(np.float32)
JEWISH = embedding['jewish']
JEWISH_ = gensim.matutils.unitvec(JEWISH).astype(np.float32)


def transformConfusionMatrix(matrix):
    TN = matrix[0][0]
    TP = matrix[1][1]
    matrix[0][0] = TP
    matrix[1][1] = TN
    return matrix

def readWordvec(file, kv = True):
    if kv == True:
        return KeyedVectors.load(file, mmap='r')
    else:
        return Word2Vec.load(file)


def buildVector(tokens, word2vec, size=150, replacementIdentity=False,isInSomeGroup = False, isToxic = False, weights = None):
    vec = np.zeros(size)
    count = 0.

    for word in tokens:

        try:
            # if replaceIdentity == True and isToxic and word not in words_pleasent + words_unpleasent:
            #     if word in weights:
            #         aux = word2vec[word]
            #         try:
            #             aux = aux + weights[word]['add_pleasent'] * pleasentVector - weights[word]['sub_unpleasent'] * unpleasentVector
            #         except: None
            #
            #         try:
            #             aux = aux - weights[word]['sub_pleasent'] * pleasentVector + weights[word]['add_unpleasent'] * unpleasentVector
            #         except: None
            #
            #         vec += aux
            #     elif (model.predict_proba(word2vec[word].reshape(1, -1))[:, 1] > 0.8):  # very toxic word
            #         aux = gensim.matutils.unitvec(word2vec[word]).astype(np.float32)
            #         word_ = None
            #         for i in range(500):
            #             aux_ = word2vec[word]
            #             add = np.random.rand()
            #             sub = np.random.rand()
            #             aux_ = aux_ - sub * unpleasentVector + add * pleasentVector
            #             if (model.predict_proba(aux_.reshape(1, -1))[:, 1] > 0.5) and (
            #                     model.predict_proba(aux_.reshape(1, -1))[:, 1] <= 0.8):
            #                 if gensim.matutils.unitvec(aux_).astype(np.float32) @ aux >= .8:
            #                     if word_ is None:
            #                         word_ = aux_
            #                     elif model.predict_proba(aux_.reshape(1, -1))[:, 1] > model.predict_proba(
            #                             word_.reshape(1, -1))[:, 1]:
            #                         word_ = aux_
            #                         weights[word]['add_pleasent'] = add
            #                         weights[word]['sub_unpleasent'] = sub
            #
            #         if word_ is None:
            #             vec += word2vec[word]
            #         else:
            #             vec += word_
            #
            #     elif (model.predict_proba(word2vec[word].reshape(1, -1))[:, 1] < 0.1):  # very non toxic word
            #         aux = gensim.matutils.unitvec(word2vec[word]).astype(np.float32)
            #         word_ = word2vec[word]
            #         for i in range(500):
            #             aux_ = word2vec[word]
            #             add = np.random.rand()
            #             sub = np.random.rand()
            #             aux_ = aux_ + add * unpleasentVector - sub * pleasentVector
            #             if (model.predict_proba(aux_.reshape(1, -1))[:, 1] >= 0.0) and (
            #                     model.predict_proba(aux_.reshape(1, -1))[:, 1] <= 0.3):
            #                 if gensim.matutils.unitvec(aux_).astype(np.float32) @ aux >= .8:
            #                     if model.predict_proba(aux_.reshape(1, -1))[:, 1] > model.predict_proba(
            #                             word_.reshape(1, -1))[:, 1]:
            #                         word_ = aux_
            #                         weights[word]['sub_pleasent'] = sub
            #                         weights[word]['add_unpleasent'] = add
            #
            #
            #         vec += word_
            #     else:
            #         vec += word2vec[word]

            if replacementIdentity: # this is to create feature for train
                if isToxic and isInSomeGroup:
                    if word in weights:
                        aux = word2vec[word]
                        try:
                            aux = aux + weights[word]['add_pleasent'] * pleasentVector - weights[word]['sub_unpleasent'] * unpleasentVector
                        except: None

                        try:
                            aux = aux - weights[word]['sub_pleasent'] * pleasentVector + weights[word]['add_unpleasent'] * unpleasentVector
                        except: None

                        vec += aux
                    else:
                        vec += word2vec[word]

                else:
                    vec += word2vec[word]
            else: # this is to create feature for test
                vec += word2vec[word]

            count += 1.
        except KeyError: # handling the case where the token is not present
            #print("\nWord not found : " + word)
            continue

    if count != 0:
        vec /= count

    assert(len(vec) == size)
    return vec


def isSimilarSomeIdentity(word):
    similarity = np.float32(0.80)
    aux = gensim.matutils.unitvec(word).astype(np.float32)
    if aux@BLACK_ > similarity or aux@WHITE_ > similarity or aux@MALE_ > similarity or aux@FEMALE_ > similarity or aux@CHRISTIAN_ > similarity or aux@MUSLIM_ > similarity or aux@JEWISH_ > similarity or aux@GAY_ > similarity or aux@HOMOSEXUAL_ > similarity or aux@LESBIAN_ > similarity:
        return True
    else:
        return False

def isInSomeGroup(sample):
    for w in groups:
        if sample[w] > 0.5:
            return True

    return False

def buildVectorUltimateBySimilarity(tokens, word2vec, size=150, isToxic=False, model=None, global_weights=None):

    vec = np.zeros(size)
    count = 0.
    best_vec = None
    best_score = 0.0
    best_weight = global_weights

    if isToxic == True:

        for tries in range(10):
            vec = np.zeros(size)
            weights = copy.deepcopy(global_weights)
            count = 0
            for word in tokens:
                try:
                    if word not in words_pleasent + words_unpleasent:
                        if word in weights:
                            aux = word2vec[word]
                            try:
                                aux = aux + weights[word]['add_pleasent'] * pleasentVector - weights[word][
                                    'sub_unpleasent'] * unpleasentVector
                            except:
                                None

                            try:
                                aux = aux - weights[word]['sub_pleasent'] * pleasentVector + weights[word][
                                    'add_unpleasent'] * unpleasentVector
                            except:
                                None

                            vec += aux
                        # debias for identity words
                        elif isSimilarSomeIdentity(word2vec[word]):
                            if (model.predict_proba(word2vec[word].reshape(1, -1))[:, 1] > 0.9):  # very toxic word
                                aux = gensim.matutils.unitvec(word2vec[word]).astype(np.float32)
                                word_ = None
                                for i in range(20):
                                    aux_ = word2vec[word]
                                    add = np.random.rand()
                                    sub = np.random.rand()
                                    aux_ = aux_ - sub * unpleasentVector + add * pleasentVector
                                    if (model.predict_proba(aux_.reshape(1, -1))[:, 1] > 0.5) and (
                                            model.predict_proba(aux_.reshape(1, -1))[:, 1] <= 0.9):
                                        if gensim.matutils.unitvec(aux_).astype(np.float32) @ aux >= .8:
                                            if word_ is None:
                                                word_ = aux_
                                            #get the highest toxic replacement possible
                                        elif model.predict_proba(aux_.reshape(1, -1))[:, 1] > model.predict_proba(
                                                    word_.reshape(1, -1))[:, 1]:
                                                word_ = aux_
                                                weights[word]['add_pleasent'] = add
                                                weights[word]['sub_unpleasent'] = sub
                                if word_ is None:
                                 vec += word2vec[word]
                                else:
                                  vec += word_
                            elif (model.predict_proba(word2vec[word].reshape(1, -1))[:, 1] < 0.1):  # very non toxic word
                                aux = gensim.matutils.unitvec(word2vec[word]).astype(np.float32)
                                word_ = word2vec[word]
                                for i in range(20):
                                    aux_ = word2vec[word]
                                    add = np.random.rand()
                                    sub = np.random.rand()
                                    aux_ = aux_ + add * unpleasentVector - sub * pleasentVector
                                    if (model.predict_proba(aux_.reshape(1, -1))[:, 1] >= 0.0) and (
                                            model.predict_proba(aux_.reshape(1, -1))[:, 1] <= 0.3):
                                        if gensim.matutils.unitvec(aux_).astype(np.float32) @ aux >= .8:
                                            if model.predict_proba(aux_.reshape(1, -1))[:, 1] > model.predict_proba(
                                                    word_.reshape(1, -1))[:, 1]:
                                                word_ = aux_
                                                weights[word]['sub_pleasent'] = sub
                                                weights[word]['add_unpleasent'] = add

                                vec += word_
                            else:
                                vec += word2vec[word]
                        # debias for highly non-toxic words trying to decrease the number of FN
                        elif (model.predict_proba(word2vec[word].reshape(1, -1))[:, 1] < 0.1):  # very non toxic word
                            aux = gensim.matutils.unitvec(word2vec[word]).astype(np.float32)
                            word_ = word2vec[word]
                            for i in range(20):
                                aux_ = word2vec[word]
                                add = np.random.rand()
                                sub = np.random.rand()
                                aux_ = aux_ + add * unpleasentVector - sub * pleasentVector
                                if (model.predict_proba(aux_.reshape(1, -1))[:, 1] >= 0.0) and (
                                        model.predict_proba(aux_.reshape(1, -1))[:, 1] <= 0.3):
                                    if gensim.matutils.unitvec(aux_).astype(np.float32) @ aux >= .8:
                                        if model.predict_proba(aux_.reshape(1, -1))[:, 1] > model.predict_proba(
                                                word_.reshape(1, -1))[:, 1]:
                                            word_ = aux_
                                            weights[word]['sub_pleasent'] = sub
                                            weights[word]['add_unpleasent'] = add

                            vec += word_
                        else:
                            vec += word2vec[word]

                    else:
                        vec += word2vec[word]

                    count += 1
                except KeyError:  # handling the case where the token is not present
                    # print("\nWord not found : " + word)
                    continue

            if count != 0:
                vec /= count

            pred = model.predict_proba(vec.reshape(1, -1))[:, 1]  # select the most toxic prediction
            if pred > best_score:
                best_score = pred
                best_weight = copy.deepcopy(weights)
                best_vec = vec
    else:
        for word in tokens:
            try:
                vec += word2vec[word]
                count += 1
            except KeyError:  # handling the case where the token is not present
                continue

    # if isToxic:
    #   # update global_weight
    #   for key, value in best_weight.items():
    #       if key not in global_weights:
    #         for subkey, subvalue in value.items():
    #           global_weights[key][subkey] = subvalue

    if isToxic == False:
        if count != 0:
          vec /= count
        best_vec = vec

    assert (len(best_vec) == size)
    return best_vec, best_weight

def buildVectorUltimateByToxicity(tokens, word2vec, size=150, isToxic=False, model=None, global_weights = None):

    vec = np.zeros(size)
    count = 0.
    best_vec = None
    best_score = 0.0
    best_weight = global_weights

    if isToxic == True:

        for tries in range(10):
            vec = np.zeros(size)
            weights = copy.deepcopy(global_weights)
            count = 0
            for word in tokens:
                try:
                    if word not in words_pleasent + words_unpleasent:
                        if word in weights:
                            aux = word2vec[word]
                            try:
                                aux = aux + weights[word]['add_pleasent'] * pleasentVector - weights[word][
                                    'sub_unpleasent'] * unpleasentVector
                            except:
                                None

                            try:
                                aux = aux - weights[word]['sub_pleasent'] * pleasentVector + weights[word][
                                    'add_unpleasent'] * unpleasentVector
                            except:
                                None

                            vec += aux
                        # debias for identity words
                        #elif isSimilarSomeIdentity(word2vec[word]):
                        elif (model.predict_proba(word2vec[word].reshape(1, -1))[:, 1] > 0.9):  # very toxic word
                                aux = gensim.matutils.unitvec(word2vec[word]).astype(np.float32)
                                word_ = None
                                for i in range(20):
                                    aux_ = word2vec[word]
                                    add = np.random.rand()
                                    sub = np.random.rand()
                                    aux_ = aux_ - sub * unpleasentVector + add * pleasentVector
                                    if (model.predict_proba(aux_.reshape(1, -1))[:, 1] > 0.5) and (
                                            model.predict_proba(aux_.reshape(1, -1))[:, 1] <= 0.9):
                                        if gensim.matutils.unitvec(aux_).astype(np.float32) @ aux >= .8:
                                            if word_ is None:
                                                word_ = aux_
                                            #get the highest toxic replacement possible
                                            elif model.predict_proba(aux_.reshape(1, -1))[:, 1] > model.predict_proba(
                                                    word_.reshape(1, -1))[:, 1]:
                                                word_ = aux_
                                                weights[word]['add_pleasent'] = add
                                                weights[word]['sub_unpleasent'] = sub
                                if word_ is None:
                                 vec += word2vec[word]
                                else:
                                  vec += word_
                        elif (model.predict_proba(word2vec[word].reshape(1, -1))[:, 1] < 0.1):  # very non toxic word
                                aux = gensim.matutils.unitvec(word2vec[word]).astype(np.float32)
                                word_ = word2vec[word]
                                for i in range(20):
                                    aux_ = word2vec[word]
                                    add = np.random.rand()
                                    sub = np.random.rand()
                                    aux_ = aux_ + add * unpleasentVector - sub * pleasentVector
                                    if (model.predict_proba(aux_.reshape(1, -1))[:, 1] >= 0.0) and (
                                            model.predict_proba(aux_.reshape(1, -1))[:, 1] <= 0.3):
                                        if gensim.matutils.unitvec(aux_).astype(np.float32) @ aux >= .8:
                                            if model.predict_proba(aux_.reshape(1, -1))[:, 1] > model.predict_proba(
                                                    word_.reshape(1, -1))[:, 1]:
                                                word_ = aux_
                                                weights[word]['sub_pleasent'] = sub
                                                weights[word]['add_unpleasent'] = add

                                vec += word_
                        else:
                                vec += word2vec[word]

                    else:
                        vec += word2vec[word]

                    count += 1
                except KeyError:  # handling the case where the token is not present
                    # print("\nWord not found : " + word)
                    continue

            if count != 0:
                vec /= count

            pred = model.predict_proba(vec.reshape(1, -1))[:, 1]  # select the most toxic prediction
            if pred > best_score:
                best_score = pred
                best_weight = copy.deepcopy(weights)
                best_vec = vec
    else:
        for word in tokens:
            try:
                vec += word2vec[word]
                count += 1
            except KeyError:  # handling the case where the token is not present
                continue

    # if isToxic:
    #   # update global_weight
    #   for key, value in best_weight.items():
    #       if key not in global_weights:
    #         for subkey, subvalue in value.items():
    #           global_weights[key][subkey] = subvalue

    if isToxic == False:
        if count != 0:
          vec /= count
        best_vec = vec

    assert (len(best_vec) == size)
    return best_vec, best_weight


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


def createFeatures(data,embedding,size,tfidf=None, replaceIdentity = False, model = None):
  features = []
  # Creating a representation for the whole tweet
  for index,sample in (data.iterrows()):

     words = gensim.utils.simple_preprocess(sample['comment_text'])
     # words = stanfordPreprocessing.tokenize(word).split()

     if TF_IDF:
      #With tfidf
      features.append(buildVectorTFIDF(words, embedding, tfidfVectorizer, tfidf.getrow(i).toarray(), size))
     else:
       if replaceIdentity:
           # train
           try:
              weights_json = dill.load(open('weights_json', 'rb'))
           except Exception:
              print('Problem loading the weights!')
              raise

           # apply the weights only in examples that belong to groups
           #features.append(buildVector(words,embedding,size = size, replaceIdentity = replaceIdentity, isInSomeGroup = isInSomeGroup(sample), isToxic=sample['target']>0.5, weights = weights_json))

           # apply the weights to any toxic example
           #features.append(buildVector(words, embedding, size=size, replaceIdentity=replaceIdentity,
           #                            isInSomeGroup=True, isToxic=sample['target'] > 0.5,
           #                            weights=weights_json))

           # do not use weights
           #features.append(buildVector(words,embedding,size = size, replaceIdentity = replaceIdentity, isInSomeGroup = isInSomeGroup(sample), isToxic=sample['target']>0.5))

       else:
           # test
           features.append(buildVector(words, embedding, size=size))

  #joblib.dump(features, open('word2vec_features_train.save', 'wb'))
  #del features_train
  #gc.collect()
  return features


def create_balanced_train_test(wiki_data):
    # Creating a balanced Test/Train for different identities
    X_train = X_test = Y_train = Y_test = None
    train = test = set()
    for x in groups:
        X_train_aux, X_test_aux, Y_train_aux, Y_test_aux = train_test_split(
            wiki_data.loc[wiki_data[x] > 0.5, ['id', 'target', 'comment_text'] + groups],
            wiki_data.loc[wiki_data[x] > 0.5, ['toxic']],
            test_size=0.3, random_state=666, stratify=wiki_data.loc[wiki_data[x] > 0.5, ['toxic']])
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

    # get the rest of examples that do not match a specific group
    index = wiki_data[~wiki_data.index.isin(train.union(test))].index
    # too many general non toxic examples. keep fewer of them
    data_aux = wiki_data.iloc[index]
    data_aux_non_toxic = data_aux.loc[data_aux['toxic'] == False].index
    data_aux_non_toxic = np.random.choice(data_aux_non_toxic, int(0.6 * len(data_aux_non_toxic)), replace=False)
    index = list(data_aux_non_toxic) + list(data_aux.loc[data_aux['toxic'] == True].index)

    X_train_aux, X_test_aux, Y_train_aux, Y_test_aux = train_test_split(
        wiki_data.loc[index, ['id', 'target', 'comment_text'] + groups],
        wiki_data.loc[index, ['toxic']],
        test_size=0.3, random_state=666, stratify=wiki_data.loc[index, ['toxic']])
    train = train.union(set(X_train_aux.index))
    test = test.union(set(X_test_aux.index))
    # X_train = pd.concat([X_train, X_train_aux])
    # X_test = pd.concat([X_test, X_test_aux])
    # Y_train = pd.concat([Y_train, Y_train_aux])
    # Y_test = pd.concat([Y_test, Y_test_aux])

    # X_train = sklearn.utils.shuffle(pd.concat([X_train, Y_train], axis=1))
    # train = random.sample(train, int(len(train)*.6)) # get only 60% of the train
    X_train = wiki_data.loc[train, ['id', 'target', 'comment_text'] + groups]
    # X_test = sklearn.utils.shuffle(pd.concat([X_test, Y_test], axis=1))
    X_test = wiki_data.loc[test, ['id', 'target', 'comment_text'] + groups]
    # Y_train = X_train.loc[:, ['id', 'toxic']]
    Y_train = wiki_data.loc[train, 'toxic']
    # Y_test = X_test.loc[:, ['id', 'toxic']]
    Y_test = wiki_data.loc[test, 'toxic']
    # X_train.drop(['id', 'toxic'], inplace=True, axis=1)
    # X_test.drop(['id', 'toxic'], inplace=True, axis=1)

    X_train.to_csv('balanced_train.csv')
    Y_train.to_csv('balanced_train_Y.csv', header=['toxic'])
    X_test.to_csv('balanced_test.csv')
    Y_test.to_csv('balanced_test_Y.csv', header=['toxic'])

    return X_train, X_test, Y_train, Y_test

# Train the model in an iterative way
def trainModel(X_train , Y_train, embedding, replaceIdentity = False):
    model = None
    model_aux = None


    # try:
    #   model = pickle.load(open('logistic_model_word2vec.save', 'rb'))
    # except:
    #   None

    if model == None:
      model = SGDClassifier(loss='log',penalty='l2',n_jobs=2)


    # if replaceIdentity:
    #   model_aux = pickle.load(open('model_word2vec_no_debias.save', 'rb'))
    # global_weights = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0.0)))
    # def filter_frame_v2(frame, keyword=None, length=None):
    #     if keyword:
    #         frame = frame[frame[keyword] > 0.5]
    #     if length:
    #         frame = frame[frame['length'] <= length]
    #     return frame
    #
    # Y_train = Y_train.applymap(lambda x: 1 if x == True else 0)
    # data = pd.concat([X_train, Y_train], axis=1)
    # data_aux = None
    # for term in groups:
    #     frame = filter_frame_v2(data, term)
    #     data_aux = pd.concat([data_aux,frame],axis=0)
    # data_aux.reset_index(inplace=True)
    # data_aux_y = data_aux['toxic']
    # for index, sample in data_aux.iterrows():
    #   words = gensim.utils.simple_preprocess(sample['comment_text'])
    #   try:
    #     features, global_weights = buildVectorUltimate(words, embedding, size = 200, isToxic=sample['target'] > 0.5, model = model_aux, global_weights=global_weights)
    #     y = data_aux_y.iloc[index]
    #     y = np.array([int(y)])
    #     model.partial_fit(features.reshape(1,-1), y, classes=[1,0])
    #   except Exception as e:
    #       print(e)
    #       continue
    # dill.dump(global_weights, open('weights_local.save', 'wb'))

    if replaceIdentity:
        try:
           weights_json = json.load(open('weights_v90.save', 'r'))
        except Exception:
           print('Problem loading the weights!')
           raise
    data = X_train.sample(frac=1, random_state=1)
    for index, sample in data.iterrows():
     words = gensim.utils.simple_preprocess(sample['comment_text'])
     features = buildVector(words, embedding, size = 200, replacementIdentity=replaceIdentity,isInSomeGroup = isInSomeGroup(sample), isToxic=sample['target']>0.5, weights = weights_json)
     model.partial_fit(features.reshape(1,-1), Y_train.iloc[index], classes=[1,0])

    # data = pd.concat([X_train,Y_train],axis=1)
    # data_toxic = data[data['toxic'] == True]
    # data_nontoxic = data[data['toxic'] == False]
    # data_nontoxic = data_nontoxic.sample(frac = 0.1, random_state=1)
    # data = pd.concat([data_toxic,data_nontoxic],axis=0)
    # data_y = data['toxic']
    # for index, sample in data.iterrows():
    #   words = gensim.utils.simple_preprocess(sample['comment_text'])
    #   features = buildVector(words, embedding, size = 200, replaceIdentity = replaceIdentity, isToxic=sample['toxic'], model = model_aux)
    #   model.partial_fit(features.reshape(1,-1), data_y[index], classes=[1,0])


    pickle.dump(model, open('model_word2vec_debias_FULL_TRAIN_v90.save', 'wb'), protocol=2)

    return model


def firstExecution(epochs):

    features_train = None
    features_test = None
    # try:
    #     features_train = joblib.load(open('word2vec_features_train.save', 'rb'))
    #     features_test = joblib.load(open('word2vec_features_test.save', 'rb'))
    #     Y_train = pd.read_csv('balanced_train_Y.csv', sep=',', usecols=['toxic'])
    #     Y_test = pd.read_csv('balanced_test_Y.csv', sep=',', usecols=['toxic'])
    # except:
    #     None

    if features_train == None:
      if os.path.isfile('balanced_train.csv'):
            X_train = pd.read_csv('balanced_train.csv', sep=',')
            X_test = pd.read_csv('balanced_test.csv', sep=',')
            Y_train = pd.read_csv('balanced_train_Y.csv', sep=',', usecols=['toxic'])
            Y_test = pd.read_csv('balanced_test_Y.csv', sep=',', usecols=['toxic'])
      else:
            file_path = "train.csv"
            wiki_data = pd.read_csv(file_path, sep=',')
            wiki_data['toxic'] = wiki_data['target'] > 0.5
            X_train, X_test, Y_train, Y_test = create_balanced_train_test(wiki_data)
            del wiki_data
            gc.collect()

    for i in range(epochs):
        model = trainModel(X_train,Y_train,embedding, replaceIdentity = True)


firstExecution(1)

tfidf = None
if TF_IDF:
    ######  TF-IDF #####
    from sklearn.feature_extraction.text import TfidfVectorizer

    tfidfVectorizer = TfidfVectorizer(encoding='latin-1', vocabulary=glove.wv.vocab.keys()
                                      , lowercase=True, tokenizer=gensim.utils.simple_preprocess)
    tfidf = tfidfVectorizer.fit_transform(X_train.loc[:, 'comment_text'])
    # #####################

X_test = pd.read_csv('balanced_test.csv', sep=',')
Y_test = pd.read_csv('balanced_test_Y.csv', sep=',', usecols=['toxic'])

loaded_model = pickle.load(open('model_word2vec_debias_FULL_TRAIN_v90.save', 'rb'))

features_test = createFeatures(X_test, embedding, size=200, tfidf=tfidf)
pred = loaded_model.predict_proba(features_test)[:, 1]
auc = roc_auc_score(Y_test, pred)
print('Test ROC AUC: %.3f' % auc)  # Test ROC AUC: 0.828
pickle.dump(pred, open('word2vec_debias_FULL_TRAIN_prediction_v90.save', 'wb'), protocol=2)

#pred = pickle.load(open('word2vec_debias_FULL_TRAIN_prediction_v91.save', 'rb'))


# Simple evaluation
auc = roc_auc_score(Y_test, pred)
print('Overall Test ROC AUC: %.3f' %auc)
print(sklearn.metrics.accuracy_score(Y_test, pred>0.5))
confusionMatrix = sklearn.metrics.confusion_matrix(Y_test, pred>0.5)
confusionMatrix = transformConfusionMatrix(confusionMatrix)
print(confusionMatrix)
print("Acceptance rate: %.3f" %(100*((confusionMatrix[0][0]+confusionMatrix[0][1])/len(Y_test))))
print("TPR: %.3f" % (100 * (confusionMatrix[0][0] / (confusionMatrix[0][0] + confusionMatrix[1][0]))))
print("TNR: %.3f" % (100 * (confusionMatrix[1][1] / (confusionMatrix[1][1] + confusionMatrix[0][1]))))


groups = ['black','christian','female',
          'homosexual', 'gay', 'lesbian','jewish','male','muslim',
          'white']

for w in groups:
  print('\n' + w + ' : ' + str(loaded_model.predict_proba(embedding[w].reshape(1,-1))[:,1]))





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
