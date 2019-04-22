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
    # save the test set to be able to test the model later
    scipy.sparse.save_npz('baseline_balanced_test_X.npz', X_test)
    Y_test.to_csv('baseline_balanced_test_Y.csv', header=['id', 'toxic'])
    # save the vocabulary
    pickle.dump(tokens.vocabulary_, open('vocabulary.save', 'wb'))

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


# Now starts the evaluation of the model regarding bias

X_test = scipy.sparse.load_npz('baseline_balanced_test_X.npz')
Y_test = pd.read_csv('baseline_balanced_test_Y.csv', sep = ',', usecols=['toxic'])
loaded_model = pickle.load(open('logistic_model.save', 'rb'))
pred = loaded_model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(Y_test, pred)
print('Test ROC AUC: %.3f' %auc)
print(loaded_model.score(X_test, Y_test))
print(sklearn.metrics.confusion_matrix(Y_test, pred>0.5))

print('\nUsing the debias dataset \n')

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
print('Test ROC AUC: %.3f' %auc)
print(loaded_model.score(X_test, Y_test))
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

print('\n Now the original test set \n')
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


