#Importamos Pandas para la lectura de datos

import pandas as pd
import numpy as np


#importamos librerías para gráficos

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


#Importamos de Sklearn para separar los datos para entrenamiento y prueba.

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold

#Import Models

from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn.linear_model import LogisticRegression #Regresión Logística
from sklearn.ensemble import RandomForestClassifier #Random Forest
from sklearn.svm import SVC #Suport Vector Machines Classifier
import xgboost as xgb #XGBoost
from sklearn.naive_bayes import GaussianNB #Naive bayes Classifier
from sklearn.ensemble import StackingClassifier


#Import metrics
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix


#Other Useful tools
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline

import optuna
import shap

from numpy import mean
from numpy import std

import time
import pickle
from scipy.stats import gmean


#===============================================================================================
#Funciones definidas.

#OPTUNA para Random Forest

def objective_rfc(trial):
  params = {'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 4, 50),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 150),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 60),
           }

      
  rfc = RandomForestClassifier(**params) 
  rfc.fit(X_train, y_train)

  y_test_pred = rfc.predict(X_test)
  recall = recall_score(y_test, y_test_pred)


  return recall

#===============================================================================================

#OPTUNA para Logistic Regression

def objective_lgro(trial):

  params = {'tol': trial.suggest_float('tol',1e-6, 1e-3),
            'C': trial.suggest_float('C',0.01,1),
            'solver': 'liblinear',
            'penalty' : 'l2',
           }
  
      
  lgro = LogisticRegression(**params) 
  lgro.fit(X_train, y_train)

  y_test_pred = lgro.predict(X_test)
  recall = recall_score(y_test, y_test_pred)


  return recall

#===============================================================================================

#OPTUNA FOR NAIVE BAYES (recall)

def objective_gnb(trial):
  params = {'var_smoothing': trial.suggest_loguniform('var_smoothing', 1e-8, 1.0)}
      
  gnbc = GaussianNB(**params) 
  gnbc.fit(X_train, y_train)

  y_test_pred = gnbc.predict(X_test)
  recall = recall_score(y_test, y_test_pred)


  return recall


#===============================================================================================

#OPTUNA FOR NAIVE BAYES (precison)

def objective_gnb2(trial):
  params = {'var_smoothing': trial.suggest_loguniform('var_smoothing', 1e-8, 1.0)}
      
  gnbc = GaussianNB(**params) 
  gnbc.fit(X_train, y_train)

  y_test_pred = gnbc.predict(X_test)
  precision = precision_score(y_test, y_test_pred)


  return precision


#===============================================================================================

#OPTUNA FOR XGBoost

def objective_xgbc(trial):
  params = {'max_depth': trial.suggest_int('max_depth', 1, 9),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 1.0),
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
            'subsample': trial.suggest_loguniform('subsample', 0.01, 1.0),
            'colsample_bytree': trial.suggest_loguniform('colsample_bytree', 0.01, 1.0),
            'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 1.0),
            'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 1.0),
            'eval_metric': 'mlogloss',
            'use_label_encoder': False,
           }
      
  xgbc = xgb.XGBClassifier(**params) 
  xgbc.fit(X_train, y_train)

  y_test_pred = xgbc.predict(X_test)
  recall = recall_score(y_test, y_test_pred)


  return recall


#===============================================================================================

#OPTUNA FOR KNN

def objective_knn(trial):
  params = {"n_neighbors" : trial.suggest_int("n_neighbors", 1, 30),
            "weights" : trial.suggest_categorical("weights", ['uniform', 'distance']),
            "metric" : trial.suggest_categorical("metric", ['euclidean', 'manhattan', 'minkowski']),
           }
      
  knno = KNeighborsClassifier (**params) 
  knno.fit(X_train, y_train)

  y_test_pred = knno.predict(X_test)
  recall = recall_score(y_test, y_test_pred)


  return recall



#===============================================================================================

#Funciones para modelos
    
# evaluate a given model using cross-validation
def evaluate_model(model, X, y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X, y, scoring='recall', cv=cv, n_jobs=-1, error_score='raise')
    return scores


#===============================================================================================

 
#Definimos Modelos con parámetros por default

# Get a list of models to evaluate

def get_models():
    models = dict()
    models['lrc'] = LogisticRegression()
    models['knn'] = KNeighborsClassifier()
    models['rfc'] = RandomForestClassifier()
    models['bayes'] = GaussianNB()
    models['xgbc'] = xgb.XGBClassifier()
    return models
   
#===============================================================================================

 
#Definimos Modelos con parámtros Optimizados con Optuna

# Get a list of models to evaluate

def get_models_opt():
    models = dict()
    models['lrc'] = LogisticRegression(tol = 0.0009454221395224076,
                                       C = 0.5767484263138968,
                                       penalty = "l2")
    models['knn'] = KNeighborsClassifier(n_neighbors = 1,
                                         weights = "distance",
                                         metric = "minkowski")
    models['rfc'] = RandomForestClassifier(n_estimators = 299,
                                           max_depth = 15,
                                           min_samples_split = 67,
                                           min_samples_leaf = 5)
    models['bayes'] = GaussianNB(var_smoothing = 0.016269942917168455)
    models['xgbc'] = xgb.XGBClassifier(max_depth = 7,
                                       learning_rate = 0.022970433313016812,
                                       n_estimators = 74,
                                       min_child_weight = 8,
                                       gamma = 0.7573940145001583,
                                       subsample = 0.9992839366266018,
                                       colsample_bytree = 0.9152533741972716,
                                       reg_alpha = 0.35230855350066587,
                                       reg_lambda = 0.04653282858249091)
    return models
   

#===============================================================================================

# Get a "list" of the stacked model to evaluate

def get_stacked_model():
    models = dict()
    models['stacking'] = get_stacking()
    return models

#===============================================================================================

# get a stacking ensemble of models (optimized models with Optuna)
def get_stacking():
    
    # define the base models
    
    level0 = list()
    
    level0.append(('lrc', LogisticRegression(tol = 0.0009454221395224076,
            C = 0.5767484263138968,
            penalty = "l2")))
    
    level0.append(('knn', KNeighborsClassifier(n_neighbors = 1,
            weights = "distance",
            metric = "minkowski")))
    
    level0.append(('gnbc', GaussianNB(var_smoothing = 0.016269942917168455)))
    
    level0.append(('xgbc', xgb.XGBClassifier(max_depth = 7,
            learning_rate = 0.022970433313016812,
            n_estimators = 74,
            min_child_weight = 8,
            gamma = 0.7573940145001583,
            subsample = 0.9992839366266018,
            colsample_bytree = 0.9152533741972716,
            reg_alpha = 0.35230855350066587,
            reg_lambda = 0.04653282858249091,)))
    
    # define meta learner model
    level1 = LogisticRegression()
    
    # define the stacking ensemble
    model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
    
    return model
