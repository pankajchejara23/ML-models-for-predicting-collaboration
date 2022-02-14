import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from sklearn.metrics import cohen_kappa_score, precision_score,recall_score, accuracy_score
from sklearn.metrics import confusion_matrix

def classificationMetrics(estimator,X,y):
    results = {}
    y_pred = estimator.predict(X)

    tn, fp, fn, tp = confusion_matrix(y,y_pred).ravel()
    results['accuracy'] = accuracy_score(y,y_pred)
    results['recall'] = recall_score(y,y_pred)
    results['precision'] = precision_score(y,y_pred)

    results['kappa'] = cohen_kappa_score(y,y_pred)


    return results


def featureAugment(X,degree):
    poly = PolynomialFeatures(degree,interaction_only=True)
    return poly.fit_transform(X)


def printClassificationMetrics(metrics1,metrics2):
    print('\n***********  Across folds performance  ***********\n')
    for metric in metrics1.keys():
        print('%s  Train: %.2f (%.2f) Test: %.2f (%.2f)' %
              (metric.capitalize().ljust(15),np.nanmean(metrics1[metric]),np.nanstd(metrics1[metric]),
               np.nanmean(metrics2[metric]),np.nanstd(metrics2[metric])))
    print("")
def assessGen(estimator,X,y,search_params,level,verbose,**kwargs):
    # to store metric score
    train_scores = {}
    test_scores={}

    fold = 1

    # metric to compute
    metrics = ['kappa','accuracy','precision','recall']

    # initialize with empty list
    for metric in metrics:
        train_scores[metric] = list()
        test_scores[metric] = list()

    if level == 'instance':
        splits = StratifiedKFold(n_splits=5).split(X,y)
    else:
        splits = LeaveOneGroupOut().split(X,y,groups=kwargs['groups'])

    for train,test in splits:
        X_train,y_train = X.iloc[train,:],y[train]
        X_test,y_test = X.iloc[test,:],y[test]

        rcv = GridSearchCV(estimator,search_params,cv=5)
        rcv.fit(X_train,y_train)
        estimator = rcv.best_estimator_

        estimator.fit(X_train,y_train)

        scores_1 = classificationMetrics(estimator,X_train,y_train)
        scores_2 = classificationMetrics(estimator,X_test,y_test)

        for metric in metrics:
            train_scores[metric].append(scores_1[metric])
            test_scores[metric].append(scores_2[metric])

        if verbose:
            print('  ------Upper Split No.%d -------' % (fold))
            print('  Train data: %s Test data: %s' % (len(train),len(test)))
            print('  Train kappa:%.2f recall_neg:%.2f' % (scores_1['kappa'],scores_1['recall_neg']))
            print('  Test kappa:%.2f recall_neg:%.2f' % (scores_2['kappa'],scores_2['recall_neg']))
            print(' ')
        fold += 1
    printClassificationMetrics(train_scores,test_scores)
    return train_scores,test_scores

def assessModel(models,search_params,X,y):
    for model in models.keys():
        print("===============")
        print(' Model:',model)
        print("===============")
        pipe = Pipeline([('scaler',StandardScaler()),('pca',PCA()),(model,models[model])])
        s = assessGen(pipe,X,y,search_params[model],'instance',False)
