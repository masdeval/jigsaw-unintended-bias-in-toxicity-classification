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

def firstExecution():

 #wiki_data = pd.read_csv(file_path, sep = ',', usecols=['target','comment_text'])
 wiki_data = pd.read_csv(file_path, sep = ',')
 wiki_data['toxic'] = wiki_data['target'] > 0.5


 def create_balanced_train_test(wiki_data):
    # Creating a balanced Test/Train for different identities
    X_train = X_test = Y_train = Y_test = None
    for x in groups:
      X_train_aux,X_test_aux,Y_train_aux,Y_test_aux = train_test_split(wiki_data.loc[wiki_data[x]>0.5,['comment_text']],wiki_data.loc[wiki_data[x]>0.5,['id','toxic']],test_size=0.3,random_state=666)
      X_train = pd.concat([X_train,X_train_aux])
      X_test = pd.concat([X_test,X_test_aux])
      Y_train = pd.concat([Y_train,Y_train_aux])
      Y_test = pd.concat([Y_test,Y_test_aux])

    X_train = sklearn.utils.shuffle(pd.concat([X_train,Y_train],axis=1))
    X_test = sklearn.utils.shuffle(pd.concat([X_test, Y_test], axis=1))
    Y_train = X_train.loc[:,['id','toxic']]
    Y_test = X_test.loc[:,['id','toxic']]
    X_train.drop(['id','toxic'],inplace=True,axis=1)
    X_test.drop(['id','toxic'],inplace=True,axis=1)

    # tokens = CountVectorizer(max_features = 10000, lowercase=True, binary=False, ngram_range=(1,2))
    tokens = CountVectorizer(max_features=10000, lowercase=True, binary=True)
    X_train = tokens.fit_transform(X_train['comment_text'])
    X_test = tokens.transform(X_test['comment_text'])

    scipy.sparse.save_npz('baseline_balanced_test_X.npz', X_test)
    Y_test.to_csv('baseline_balanced_test_Y.csv', header=['id', 'toxic'])

    return X_train , X_test , Y_train['toxic'] , Y_test['toxic']


 X_train , X_test , Y_train , Y_test = create_balanced_train_test(wiki_data)

 del wiki_data
 gc.collect()


 def trainModel(X_train , Y_train):
    model = LogisticRegressionCV(penalty='l2',max_iter=200,solver='lbfgs',n_jobs=2)
    model.fit(X_train,Y_train)
    # save the model to disk
    pickle.dump(model, open('logistic_model.save', 'wb'))
    return model

 model = trainModel(X_train , Y_train)
 auc = roc_auc_score(Y_test, model.predict_proba(X_test)[:, 1])
 print('Test ROC AUC: %.3f' %auc) #Test ROC AUC: 0.888



# Inside this function are all the steps to train a simple bow classifier to predict toxicity.
# Also, it creates balanced train/test sets taking into account the identities, specified here by the variable groups
# The test set is saved to use later on
# The logistic regression model is also saved after training
# firstExecution() encapsulates everything and is useful to quickly turn-off this part

#firstExecution()


# Now begins the evaluation of the model regarding bias

X_test = scipy.sparse.load_npz('baseline_balanced_test_X.npz')
Y_test = pd.read_csv('baseline_balanced_test_Y.csv', sep = ',', usecols=['toxic'])
loaded_model = pickle.load(open('logistic_model.save', 'rb'))
pred = loaded_model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(Y_test, pred)
print('Test ROC AUC: %.3f' %auc)
print(loaded_model.score(X_test, Y_test))
print(sklearn.metrics.confusion_matrix(Y_test, pred>0.6))


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


