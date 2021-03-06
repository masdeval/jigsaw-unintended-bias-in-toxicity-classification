import pandas as pd
import urllib
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import roc_auc_score
import sklearn.metrics as metrics
import numpy as np
import gensim
import sklearn
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import  SGDClassifier
import os
import pickle
import gc
import scipy
import random

file_path ="train.csv"

groups = ['black','christian','female',
          'homosexual_gay_or_lesbian','jewish','male','muslim',
          'psychiatric_or_mental_illness','white']

def create_balanced_train_test(wiki_data):
        # Creating a balanced Test/Train for different identities
        X_train = X_test = Y_train = Y_test = None
        train = test = set()
        for x in groups:
            X_train_aux, X_test_aux, Y_train_aux, Y_test_aux = train_test_split(
                wiki_data.loc[wiki_data[x] > 0.5, ['id', 'target', 'comment_text'] + groups],
                wiki_data.loc[wiki_data[x] > 0.5, ['toxic']],
                test_size=0.3, random_state=666, stratify = wiki_data.loc[wiki_data[x] > 0.5, ['toxic']])
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
        #too many general non toxic examples. keep fewer of them
        data_aux = wiki_data.iloc[index]
        data_aux_non_toxic = data_aux.loc[data_aux['toxic'] == False].index
        data_aux_non_toxic = np.random.choice(data_aux_non_toxic, int(0.6*len(data_aux_non_toxic)), replace=False)
        index = list(data_aux_non_toxic) + list(data_aux.loc[data_aux['toxic'] == True].index)

        X_train_aux, X_test_aux, Y_train_aux, Y_test_aux = train_test_split(
            wiki_data.loc[index, ['id', 'target', 'comment_text'] + groups],
            wiki_data.loc[index, ['toxic']],
            test_size=0.3, random_state=666, stratify = wiki_data.loc[index, ['toxic']])
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


# save the test set to be able to test the model later

def trainModel(X_train , Y_train):
    model = None
    try:
      model = pickle.load(open('logistic_model.save', 'rb'))
    except:
      None

    if model != None:
      model.partial_fit(X_train, Y_train)
      # save the model to disk
      pickle.dump(model, open('logistic_model.save', 'wb'))
    else:
      # loss hinge = SVM
      # loss log = LogisticRegression
      model = SGDClassifier(loss='log',penalty='l2',n_jobs=2)
      model.fit(X_train,Y_train)
      # save the model to disk
      pickle.dump(model, open('logistic_model.save', 'wb'))

    return model

def augumentToxicityOfTrainData(toxicData, X_train, Y_train):
    toxicData = toxicData.query('is_toxic')
    toxicData.rename(columns={"comment":'comment_text',"is_toxic":'toxic'},inplace=True)
    toxicData_comment = toxicData.loc[:,'comment_text']
    toxicData_toxic = toxicData.loc[:,'toxic']
    X_train = pd.concat([X_train,toxicData_comment],axis=0,ignore_index=True)
    Y_train = pd.concat([Y_train,toxicData_toxic],axis=0,ignore_index=True)

    return X_train,Y_train

def firstExecution(epochs):

    if os.path.isfile('balanced_train.csv'):
        X_train = pd.read_csv('balanced_train.csv', sep=',', usecols=['comment_text'])
        X_test = pd.read_csv('balanced_test.csv', sep=',', usecols=['comment_text'])
        Y_train = pd.read_csv('balanced_train_Y.csv', sep=',', usecols=['toxic'])
        Y_test = pd.read_csv('balanced_test_Y.csv', sep=',', usecols=['toxic'])
    else:
        wiki_data = pd.read_csv(file_path, sep=',')
        wiki_data['toxic'] = wiki_data['target'] > 0.5
        X_train, X_test, Y_train, Y_test = create_balanced_train_test(wiki_data)
        del wiki_data
        gc.collect()

    #aux_train = pd.read_csv('../unintended-ml-bias-analysis-master/data/wiki_train.csv', sep=',', usecols=['comment', 'is_toxic'])
    #X_train, Y_train = augumentToxicityOfTrainData(aux_train,X_train,Y_train)

    # tokens = CountVectorizer(max_features = 10000, lowercase=True, binary=False, ngram_range=(1,2))
    tokens = CountVectorizer(max_features=10000, lowercase=True, binary=True)
    features_train = tokens.fit_transform(X_train['comment_text'])
    features_test = tokens.transform(X_test['comment_text'])
    # save the vocabulary
    pickle.dump(tokens.vocabulary_, open('vocabulary.save', 'wb'))

    for i in range(epochs):
        model = trainModel(features_train, Y_train)


    ## **** Train over false positive/false negative - seems do not work well
    # features_index = list(range(len(X_train.index)))
    # for i in range(epochs):
    #     model = trainModel(features_train[features_index,:] , Y_train.iloc[features_index,:])
    #     pred = model.predict(features_train[features_index,:])
    #     confusionMatrix = sklearn.metrics.confusion_matrix(Y_train.iloc[features_index,:], pred)
    #     if confusionMatrix[0][1] == 0 & confusionMatrix[1][0] == 0:
    #         break
    #     false_positives = list()
    #     false_negatives = list()
    #     i = 0
    #     for index, v in X_train.iloc[features_index,:].iterrows():
    #        if (pred[i] == True and v['target'] <= 0.5):
    #            false_positives.append(i)
    #        if (pred[i] == False and v['target'] > 0.5):
    #            false_negatives.append(index)
    #        i += 1
    #     features_index = (false_positives+false_negatives)
    pred = model.predict_proba(features_test)[:, 1]
    auc = roc_auc_score(Y_test, pred)
    print('Test ROC AUC: %.3f' %auc) #Test ROC AUC: 0.888
    pickle.dump(pred, open('baseline_predictions.save', 'wb'))

# Inside this function are all the steps to train a simple bow classifier to predict toxicity.
# Also, it creates balanced train/test sets taking into account the identities, specified here by the variable groups
# The test set is saved to use later on
# The logistic regression model is also saved after training
# firstExecution() encapsulates everything and is useful to quickly turn-off this part
firstExecution(1)


# Now starts the evaluation

Y_test = pd.read_csv('balanced_test_Y.csv', sep = ',', usecols=['toxic'])
pred = pickle.load(open('baseline_predictions.save', 'rb'))
auc = roc_auc_score(Y_test, pred)
print('Overall Test ROC AUC: %.3f' %auc)
print(sklearn.metrics.accuracy_score(Y_test, pred>0.5))

# The matrix returned by sklearn.metrics.confusion_matrix is in the form:
#    TN  FP
#    FN  TP
# This function changes it to the format:
#    TP  FP
#    FN  TN
def transformConfusionMatrix(matrix):
    TN = matrix[0][0]
    TP = matrix[1][1]
    matrix[0][0] = TP
    matrix[1][1] = TN
    return matrix

confusionMatrix = transformConfusionMatrix(sklearn.metrics.confusion_matrix(Y_test, pred>0.5))
print(confusionMatrix)
print("Acceptance rate: %.3f" %(100*((confusionMatrix[0][0]+confusionMatrix[0][1])/len(Y_test))))
print("TPR: %.3f" % (100 * (confusionMatrix[0][0] / (confusionMatrix[0][0] + confusionMatrix[1][0]))))
print("TNR: %.3f" % (100 * (confusionMatrix[1][1] / (confusionMatrix[1][1] + confusionMatrix[0][1]))))














# Training with cross-validation
# result = cross_validate(LogisticRegression(penalty='l2'),X=X,y=Y,cv=5,scoring=['accuracy','f1'], return_train_score=False)
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


