import sys
import time
import random
import pandas as pd
import numpy as np

import pickle
import copy

#Scikit-Learn Packages:
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from skeLCS import eLCS
from skXCS import XCS
from skExSTraCS import ExSTraCS
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn import metrics

import xgboost as xgb
import lightgbm as lgb
import optuna #hyperparameter optimization
from skrebate import ReliefF

def job(algorithm,train_file_path,test_file_path,full_path,n_trials,timeout,lcs_timeout,plot_hyperparam_sweep,instance_label,class_label,random_state,cvCount):
    job_start_time = time.time()
    random.seed(random_state)
    np.random.seed(random_state)

    #Get hyperparameter grid
    param_grid = hyperparameters()[algorithm]

    #Get X and Y
    train = pd.read_csv(train_file_path)
    if instance_label != 'None':
        train = train.drop(instance_label,axis=1)
    trainX = train.drop(class_label,axis=1).values
    trainY = train[class_label].values

    test = pd.read_csv(test_file_path)
    if instance_label != 'None':
        test = test.drop(instance_label,axis=1)
    testX = test.drop(class_label,axis=1).values
    testY = test[class_label].values

    #Run model
    abbrev = {'logistic_regression':'LR','decision_tree':'DT','random_forest':'RF','naive_bayes':'NB','XGB':'XGB','LGB':'LGB','ANN':'ANN','SVM':'SVM','ExSTraCS':'ExSTraCS','eLCS':'eLCS','XCS':'XCS','gradient_boosting':'GB','k_neighbors':'KN'}
    if algorithm == 'logistic_regression':
        ret = run_LR_full(trainX,trainY,testX,testY, random_state, cvCount,param_grid,n_trials,timeout,plot_hyperparam_sweep,full_path)
    elif algorithm == 'decision_tree':
        ret = run_DT_full(trainX, trainY, testX,testY, random_state,cvCount, param_grid,n_trials, timeout,plot_hyperparam_sweep,full_path)
    elif algorithm == 'random_forest':
        ret = run_RF_full(trainX, trainY, testX,testY, random_state,cvCount, param_grid,n_trials, timeout,plot_hyperparam_sweep, full_path)
    elif algorithm == 'naive_bayes':
        ret = run_NB_full(trainX, trainY, testX,testY, random_state,cvCount, param_grid,n_trials, timeout,plot_hyperparam_sweep,full_path)
    elif algorithm == 'XGB':
        ret = run_XGB_full(trainX, trainY, testX,testY, random_state,cvCount, param_grid,n_trials, timeout,plot_hyperparam_sweep,full_path)
    elif algorithm == 'LGB':
        ret = run_LGB_full(trainX, trainY, testX, testY, random_state, cvCount, param_grid, n_trials, timeout,plot_hyperparam_sweep, full_path)
    elif algorithm == 'ANN':
        ret = run_ANN_full(trainX, trainY, testX, testY, random_state, cvCount, param_grid, n_trials, timeout,plot_hyperparam_sweep, full_path)
    elif algorithm == 'SVM':
        ret = run_SVM_full(trainX, trainY, testX, testY, random_state, cvCount, param_grid, n_trials, timeout,plot_hyperparam_sweep, full_path)
    elif algorithm == 'eLCS':
        ret = run_eLCS_full(trainX, trainY, testX, testY, random_state, cvCount, param_grid, n_trials, lcs_timeout,plot_hyperparam_sweep, full_path)
    elif algorithm == 'XCS':
        ret = run_XCS_full(trainX, trainY, testX, testY, random_state, cvCount, param_grid, n_trials, lcs_timeout,plot_hyperparam_sweep, full_path)
    elif algorithm == 'ExSTraCS':
        ret = run_ExSTraCS_full(trainX, trainY, testX, testY, random_state, cvCount, param_grid, n_trials, lcs_timeout,plot_hyperparam_sweep, full_path)
    elif algorithm == 'gradient_boosting':
        ret = run_GB_full(trainX, trainY, testX, testY, random_state, cvCount, param_grid, n_trials, timeout,plot_hyperparam_sweep, full_path)
    elif algorithm == 'k_neighbors':
        ret = run_KN_full(trainX, trainY, testX, testY, random_state, cvCount, param_grid, n_trials, timeout,plot_hyperparam_sweep, full_path)
    pickle.dump(ret, open(full_path + '/training/' + abbrev[algorithm] + '_CV_' + str(cvCount) + "_metrics", 'wb'))

    # Save Runtime
    runtime_file = open(full_path + '/runtime/runtime_'+abbrev[algorithm]+'_CV'+str(cvCount)+'.txt', 'w')
    runtime_file.write(str(time.time() - job_start_time))
    runtime_file.close()

    # Print completion
    print(full_path.split('/')[-1] + " CV" + str(cvCount) + " phase 4 "+abbrev[algorithm]+" training complete")
    experiment_path = '/'.join(full_path.split('/')[:-1])
    job_file = open(experiment_path + '/jobsCompleted/job_model_' + full_path.split('/')[-1] + '_' + str(cvCount) +'_' +abbrev[algorithm]+'.txt', 'w')
    job_file.write('complete')
    job_file.close()

def hyper_eval(est, x_train, y_train, randSeed, hype_cv, params, scoring_metric):
    cv = StratifiedKFold(n_splits=hype_cv, shuffle=True, random_state=randSeed)
    model = clone(est).set_params(**params)
    #Flexibly handle whether random seed is given as 'random_seed' or 'seed' - scikit learn uses 'random_seed'
    for a in ['random_state','seed']:
        if hasattr(model,a):
            setattr(model,a,randSeed)
    performance = np.mean(cross_val_score(model,x_train,y_train,cv=cv,scoring=scoring_metric ))
    return performance

def objective_LR(trial, est, x_train, y_train, randSeed, hype_cv, param_grid, scoring_metric):
    params = {'penalty' : trial.suggest_categorical('penalty',param_grid['penalty']),
			  'dual' : trial.suggest_categorical('dual', param_grid['dual']),
			  'C' : trial.suggest_loguniform('C', param_grid['C'][0], param_grid['C'][1]),
			  'solver' : trial.suggest_categorical('solver',param_grid['solver']),
			  'class_weight' : trial.suggest_categorical('class_weight',param_grid['class_weight']),
			  'max_iter' : trial.suggest_loguniform('max_iter',param_grid['max_iter'][0], param_grid['max_iter'][1]),
			  'n_jobs' : trial.suggest_categorical('n_jobs',param_grid['n_jobs'])}
    return hyper_eval(est, x_train, y_train, randSeed, hype_cv, params, scoring_metric)

def run_LR_full(x_train, y_train, x_test, y_test,randSeed,i,param_grid,n_trials,timeout,do_plot,full_path):
    est = LogisticRegression()
    sampler = optuna.samplers.TPESampler(seed=randSeed)  # Make the sampler behave in a deterministic way.
    study = optuna.create_study(direction='maximize', sampler=sampler)
    optuna.logging.set_verbosity(optuna.logging.CRITICAL)
    study.optimize(lambda trial: objective_LR(trial, est, x_train, y_train, randSeed, 3, param_grid, 'balanced_accuracy'),n_trials=n_trials, timeout=timeout, catch=(ValueError,))

    if do_plot == 'True':
        fig = optuna.visualization.plot_parallel_coordinate(study)
        fig.write_image(full_path+'/training/LR_ParamOptimization_'+str(i)+'.png')
    best_trial = study.best_trial

    # Train model using 'best' hyperparameters
    est = LogisticRegression()
    clf = clone(est).set_params(**best_trial.params)
    setattr(clf, 'random_state', randSeed)

    model = clf.fit(x_train, y_train)

    # Save model
    pickle.dump(model, open(full_path+'/training/pickledModels/LR_'+str(i), 'wb'))

    # Prediction evaluation
    y_pred = clf.predict(x_test)

    metricList = classEval(y_test, y_pred)

    # Determine probabilities of class predictions for each test instance (this will be used much later in calculating an ROC curve)
    probas_ = model.predict_proba(x_test)

    # Compute ROC curve and AUC
    fpr, tpr, thresholds = metrics.roc_curve(y_test, probas_[:, 1])
    roc_auc = auc(fpr, tpr)

    # Compute Precision/Recall curve and AUC
    prec, recall, thresholds = metrics.precision_recall_curve(y_test, probas_[:, 1])
    prec, recall, thresholds = prec[::-1], recall[::-1], thresholds[::-1]
    prec_rec_auc = auc(recall, prec)
    ave_prec = metrics.average_precision_score(y_test, probas_[:, 1])

    # Feature Importance Estimates
    fi = computeImportances(clf, x_train, y_train, x_test, y_test, metricList[0])

    return [metricList, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, fi]

def objective_DT(trial, est, x_train, y_train, randSeed, hype_cv, param_grid, scoring_metric):
    params = {'criterion' : trial.suggest_categorical('criterion',param_grid['criterion']),
                'splitter' : trial.suggest_categorical('splitter', param_grid['splitter']),
                'max_depth' : trial.suggest_int('max_depth', param_grid['max_depth'][0], param_grid['max_depth'][1]),
                'min_samples_split' : trial.suggest_int('min_samples_split', param_grid['min_samples_split'][0], param_grid['min_samples_split'][1]),
                'min_samples_leaf' : trial.suggest_int('min_samples_leaf', param_grid['min_samples_leaf'][0], param_grid['min_samples_leaf'][1]),
                'max_features' : trial.suggest_categorical('max_features',param_grid['max_features']),
                'class_weight' : trial.suggest_categorical('class_weight',param_grid['class_weight'])}
    return hyper_eval(est, x_train, y_train, randSeed, hype_cv, params, scoring_metric)

def run_DT_full(x_train, y_train, x_test, y_test,randSeed,i,param_grid,n_trials,timeout,do_plot,full_path):
    # Run Hyperparameter sweep
    est = tree.DecisionTreeClassifier()
    sampler = optuna.samplers.TPESampler(seed=randSeed)  # Make the sampler behave in a deterministic way.
    study = optuna.create_study(direction='maximize', sampler=sampler)
    optuna.logging.set_verbosity(optuna.logging.CRITICAL)
    study.optimize(lambda trial: objective_DT(trial, est, x_train, y_train, randSeed, 3, param_grid, "balanced_accuracy"),n_trials=n_trials, timeout=timeout, catch=(ValueError,))

    if do_plot == 'True':
        fig = optuna.visualization.plot_parallel_coordinate(study)
        fig.write_image(full_path+'/training/DT_ParamOptimization_'+str(i)+'.png')

    best_trial = study.best_trial

    # Train model using 'best' hyperparameters
    est = tree.DecisionTreeClassifier()
    clf = clone(est).set_params(**best_trial.params)
    setattr(clf, 'random_state', randSeed)

    model = clf.fit(x_train, y_train)

    # Save model
    pickle.dump(model, open(full_path+'/training/pickledModels/DT_'+str(i), 'wb'))

    # Prediction evaluation
    y_pred = clf.predict(x_test)

    metricList = classEval(y_test, y_pred)

    # Determine probabilities of class predictions for each test instance (this will be used much later in calculating an ROC curve)
    probas_ = model.predict_proba(x_test)

    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = metrics.roc_curve(y_test, probas_[:, 1])
    roc_auc = auc(fpr, tpr)

    # Compute Precision/Recall curve and AUC
    prec, recall, thresholds = metrics.precision_recall_curve(y_test, probas_[:, 1])
    prec, recall, thresholds = prec[::-1], recall[::-1], thresholds[::-1]
    prec_rec_auc = auc(recall, prec)
    ave_prec = metrics.average_precision_score(y_test, probas_[:, 1])

    # Feature Importance Estimates
    fi = clf.feature_importances_

    return [metricList, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, fi]

def objective_RF(trial, est, x_train, y_train, randSeed, hype_cv, param_grid, scoring_metric):
    params = {'n_estimators' : trial.suggest_int('n_estimators',param_grid['n_estimators'][0], param_grid['n_estimators'][1]),
                'criterion' : trial.suggest_categorical('criterion',param_grid['criterion']),
                'max_depth' : trial.suggest_int('max_depth', param_grid['max_depth'][0], param_grid['max_depth'][1]),
                'min_samples_split' : trial.suggest_int('min_samples_split', param_grid['min_samples_split'][0], param_grid['min_samples_split'][1]),
                'min_samples_leaf' : trial.suggest_int('min_samples_leaf', param_grid['min_samples_leaf'][0], param_grid['min_samples_leaf'][1]),
                'max_features' : trial.suggest_categorical('max_features',param_grid['max_features']),
                'bootstrap' : trial.suggest_categorical('bootstrap',param_grid['bootstrap']),
                'oob_score' : trial.suggest_categorical('oob_score',param_grid['oob_score']),
                'n_jobs' : trial.suggest_categorical('n_jobs',param_grid['n_jobs']),
                'class_weight' : trial.suggest_categorical('class_weight',param_grid['class_weight'])}
    return hyper_eval(est, x_train, y_train, randSeed, hype_cv, params, scoring_metric)

def run_RF_full(x_train, y_train, x_test, y_test,randSeed,i,param_grid,n_trials,timeout,do_plot,full_path):
    # Run Hyperparameter sweep
    est = RandomForestClassifier()
    sampler = optuna.samplers.TPESampler(seed=randSeed)  # Make the sampler behave in a deterministic way.
    study = optuna.create_study(direction='maximize', sampler=sampler)
    optuna.logging.set_verbosity(optuna.logging.CRITICAL)
    study.optimize(lambda trial: objective_RF(trial, est, x_train, y_train, randSeed, 3, param_grid, 'balanced_accuracy'),n_trials=n_trials, timeout=timeout, catch=(ValueError,))

    if do_plot == 'True':
        fig = optuna.visualization.plot_parallel_coordinate(study)
        fig.write_image(full_path+'/training/RF_ParamOptimization_'+str(i)+'.png')

    best_trial = study.best_trial

    # Train model using 'best' hyperparameters
    est = RandomForestClassifier()
    clf = clone(est).set_params(**best_trial.params)
    setattr(clf, 'random_state', randSeed)

    model = clf.fit(x_train, y_train)

    # Save model
    pickle.dump(model, open(full_path+'/training/pickledModels/RF_'+str(i), 'wb'))

    # Prediction evaluation
    y_pred = clf.predict(x_test)

    metricList = classEval(y_test, y_pred)

    # Determine probabilities of class predictions for each test instance (this will be used much later in calculating an ROC curve)
    probas_ = model.predict_proba(x_test)

    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = metrics.roc_curve(y_test, probas_[:, 1])
    roc_auc = auc(fpr, tpr)

    # Compute Precision/Recall curve and AUC
    prec, recall, thresholds = metrics.precision_recall_curve(y_test, probas_[:, 1])
    prec, recall, thresholds = prec[::-1], recall[::-1], thresholds[::-1]
    prec_rec_auc = auc(recall, prec)
    ave_prec = metrics.average_precision_score(y_test, probas_[:, 1])

    # Feature Importance Estimates
    fi = clf.feature_importances_

    return [metricList, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, fi]

def run_NB_full(x_train, y_train, x_test, y_test,randSeed,i,param_grid,n_trials,timeout,do_plot,full_path):
    #No hyperparameters to optimize.

    #Train model using 'best' hyperparameters - Uses default 3-fold internal CV (training/validation splits)
    clf = GaussianNB()
    model = clf.fit(x_train, y_train)

    #Save model
    pickle.dump(model, open(full_path+'/training/pickledModels/NB_'+str(i), 'wb'))

    #Prediction evaluation
    y_pred = clf.predict(x_test)

    metricList = classEval(y_test, y_pred)

    #Determine probabilities of class predictions for each test instance (this will be used much later in calculating an ROC curve)
    probas_ = model.predict_proba(x_test)

    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = metrics.roc_curve(y_test, probas_[:, 1])
    roc_auc = auc(fpr, tpr)

    # Compute Precision/Recall curve and AUC
    prec, recall, thresholds = metrics.precision_recall_curve(y_test, probas_[:, 1])
    prec, recall, thresholds = prec[::-1], recall[::-1], thresholds[::-1]
    prec_rec_auc = auc(recall, prec)
    ave_prec = metrics.average_precision_score(y_test, probas_[:, 1])

    #Feature Importance Estimates
    fi = computeImportances(clf, x_train, y_train, x_test, y_test, metricList[0])

    return [metricList, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, fi]

def objective_XGB(trial, est, x_train, y_train, randSeed, hype_cv, param_grid, scoring_metric):
    posInst = sum(y_train)
    negInst = len(y_train) - posInst
    classWeight = negInst/float(posInst)
    params = {'booster' : trial.suggest_categorical('booster',param_grid['booster']),
                'objective' : trial.suggest_categorical('objective',param_grid['objective']),
                'verbosity' : trial.suggest_categorical('verbosity',param_grid['verbosity']),
                'reg_lambda' : trial.suggest_loguniform('reg_lambda', param_grid['reg_lambda'][0], param_grid['reg_lambda'][1]),
                'alpha' : trial.suggest_loguniform('alpha', param_grid['alpha'][0], param_grid['alpha'][1]),
                'eta' : trial.suggest_loguniform('eta', param_grid['eta'][0], param_grid['eta'][1]),
                'gamma' : trial.suggest_loguniform('gamma', param_grid['gamma'][0], param_grid['gamma'][1]),
                'max_depth' : trial.suggest_int('max_depth', param_grid['max_depth'][0], param_grid['max_depth'][1]),
                'grow_policy' : trial.suggest_categorical('grow_policy',param_grid['grow_policy']),
                'n_estimators' : trial.suggest_int('n_estimators',param_grid['n_estimators'][0], param_grid['n_estimators'][1]),
                'min_samples_split' : trial.suggest_int('min_samples_split', param_grid['min_samples_split'][0], param_grid['min_samples_split'][1]),
                'min_samples_leaf' : trial.suggest_int('min_samples_leaf', param_grid['min_samples_leaf'][0], param_grid['min_samples_leaf'][1]),
                'subsample' : trial.suggest_uniform('subsample', param_grid['subsample'][0], param_grid['subsample'][1]),
                'min_child_weight' : trial.suggest_loguniform('min_child_weight', param_grid['min_child_weight'][0], param_grid['min_child_weight'][1]),
                'colsample_bytree' : trial.suggest_uniform('colsample_bytree', param_grid['colsample_bytree'][0], param_grid['colsample_bytree'][1]),
                'scale_pos_weight' : trial.suggest_categorical('scale_pos_weight', [1.0, classWeight])}
    return hyper_eval(est, x_train, y_train, randSeed, hype_cv, params, scoring_metric)

def run_XGB_full(x_train, y_train, x_test, y_test,randSeed,i,param_grid,n_trials,timeout,do_plot,full_path):
    # Run Hyperparameter sweep
    est = xgb.XGBClassifier()
    sampler = optuna.samplers.TPESampler(seed=randSeed)  # Make the sampler behave in a deterministic way.
    study = optuna.create_study(direction='maximize', sampler=sampler)
    optuna.logging.set_verbosity(optuna.logging.CRITICAL)
    study.optimize(lambda trial: objective_XGB(trial, est, x_train, y_train, randSeed, 3, param_grid, 'balanced_accuracy'),n_trials=n_trials, timeout=timeout, catch=(ValueError,))

    if do_plot == 'True':
        fig = optuna.visualization.plot_parallel_coordinate(study)
        fig.write_image(full_path+'/training/XGB_ParamOptimization_'+str(i)+'.png')

    best_trial = study.best_trial

    # Train model using 'best' hyperparameters
    est = xgb.XGBClassifier()
    clf = clone(est).set_params(**best_trial.params)
    setattr(clf, 'random_state', randSeed)

    model = clf.fit(x_train, y_train)

    # Save model
    pickle.dump(model, open(full_path+'/training/pickledModels/XGB_'+str(i), 'wb'))

    # Prediction evaluation
    y_pred = clf.predict(x_test)

    metricList = classEval(y_test, y_pred)

    # Determine probabilities of class predictions for each test instance (this will be used much later in calculating an ROC curve)
    probas_ = model.predict_proba(x_test)

    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = metrics.roc_curve(y_test, probas_[:, 1])
    roc_auc = auc(fpr, tpr)

    # Compute Precision/Recall curve and AUC
    prec, recall, thresholds = metrics.precision_recall_curve(y_test, probas_[:, 1])
    prec, recall, thresholds = prec[::-1], recall[::-1], thresholds[::-1]
    prec_rec_auc = auc(recall, prec)
    ave_prec = metrics.average_precision_score(y_test, probas_[:, 1])

    # Feature Importance Estimates
    fi = computeImportances(clf, x_train, y_train, x_test, y_test, metricList[0])

    return [metricList, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, fi]

def objective_LGB(trial, est, x_train, y_train, randSeed, hype_cv, param_grid, scoring_metric):
    posInst = sum(y_train)
    negInst = len(y_train) - posInst
    classWeight = negInst / float(posInst)
    params = {'objective': trial.suggest_categorical('objective', param_grid['objective']),
              'metric': trial.suggest_categorical('metric', param_grid['metric']),
              'verbosity': trial.suggest_categorical('verbosity', param_grid['verbosity']),
              'boosting_type': trial.suggest_categorical('boosting_type', param_grid['boosting_type']),
              'num_leaves': trial.suggest_int('num_leaves', param_grid['num_leaves'][0], param_grid['num_leaves'][1]),
              'max_depth': trial.suggest_int('max_depth', param_grid['max_depth'][0], param_grid['max_depth'][1]),
              'lambda_l1': trial.suggest_loguniform('lambda_l1', param_grid['lambda_l1'][0],param_grid['lambda_l1'][1]),
              'lambda_l2': trial.suggest_loguniform('lambda_l2', param_grid['lambda_l2'][0],param_grid['lambda_l2'][1]),
              'feature_fraction': trial.suggest_uniform('feature_fraction', param_grid['feature_fraction'][0],param_grid['feature_fraction'][1]),
              'bagging_fraction': trial.suggest_uniform('bagging_fraction', param_grid['bagging_fraction'][0],param_grid['bagging_fraction'][1]),
              'bagging_freq': trial.suggest_int('bagging_freq', param_grid['bagging_freq'][0],param_grid['bagging_freq'][1]),
              'min_child_samples': trial.suggest_int('min_child_samples', param_grid['min_child_samples'][0],param_grid['min_child_samples'][1]),
              'n_estimators': trial.suggest_int('n_estimators', param_grid['n_estimators'][0],param_grid['n_estimators'][1]),
              'scale_pos_weight': trial.suggest_categorical('scale_pos_weight', [1.0, classWeight])}
    return hyper_eval(est, x_train, y_train, randSeed, hype_cv, params, scoring_metric)

def run_LGB_full(x_train, y_train, x_test, y_test,randSeed,i,param_grid,n_trials,timeout,do_plot,full_path):
    # Run Hyperparameter sweep
    est = lgb.LGBMClassifier()
    sampler = optuna.samplers.TPESampler(seed=randSeed)  # Make the sampler behave in a deterministic way.
    study = optuna.create_study(direction='maximize', sampler=sampler)
    optuna.logging.set_verbosity(optuna.logging.CRITICAL)
    study.optimize(lambda trial: objective_LGB(trial, est, x_train, y_train, randSeed, 3, param_grid, 'balanced_accuracy'),n_trials=n_trials, timeout=timeout, catch=(ValueError,))

    if do_plot == 'True':
        fig = optuna.visualization.plot_parallel_coordinate(study)
        fig.write_image(full_path+'/training/LGB_ParamOptimization_'+str(i)+'.png')

    best_trial = study.best_trial

    # Train model using 'best' hyperparameters
    est = lgb.LGBMClassifier()
    clf = clone(est).set_params(**best_trial.params)
    setattr(clf, 'random_state', randSeed)

    model = clf.fit(x_train, y_train)

    # Save model
    pickle.dump(model, open(full_path+'/training/pickledModels/LGB_'+str(i), 'wb'))

    # Prediction evaluation
    y_pred = clf.predict(x_test)

    metricList = classEval(y_test, y_pred)

    # Determine probabilities of class predictions for each test instance (this will be used much later in calculating an ROC curve)
    probas_ = model.predict_proba(x_test)

    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = metrics.roc_curve(y_test, probas_[:, 1])
    roc_auc = auc(fpr, tpr)

    # Compute Precision/Recall curve and AUC
    prec, recall, thresholds = metrics.precision_recall_curve(y_test, probas_[:, 1])
    prec, recall, thresholds = prec[::-1], recall[::-1], thresholds[::-1]
    prec_rec_auc = auc(recall, prec)
    ave_prec = metrics.average_precision_score(y_test, probas_[:, 1])

    # Feature Importance Estimates
    fi = computeImportances(clf, x_train, y_train, x_test, y_test, metricList[0])

    return [metricList, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, fi]

def objective_SVM(trial, est, x_train, y_train, randSeed, hype_cv, param_grid, scoring_metric):
    params = {'kernel': trial.suggest_categorical('kernel', param_grid['kernel']),
              'C': trial.suggest_loguniform('C', param_grid['C'][0], param_grid['C'][1]),
              'gamma': trial.suggest_categorical('gamma', param_grid['gamma']),
              'degree': trial.suggest_int('degree', param_grid['degree'][0], param_grid['degree'][1]),
              'probability': trial.suggest_categorical('probability', param_grid['probability']),
              'class_weight': trial.suggest_categorical('class_weight', param_grid['class_weight'])}
    return hyper_eval(est, x_train, y_train, randSeed, hype_cv, params, scoring_metric)

def run_SVM_full(x_train, y_train, x_test, y_test,randSeed,i,param_grid,n_trials,timeout,do_plot,full_path):
    # Run Hyperparameter sweep
    est = SVC()
    sampler = optuna.samplers.TPESampler(seed=randSeed)  # Make the sampler behave in a deterministic way.
    study = optuna.create_study(direction='maximize', sampler=sampler)
    optuna.logging.set_verbosity(optuna.logging.CRITICAL)
    study.optimize(lambda trial: objective_SVM(trial, est, x_train, y_train, randSeed, 3, param_grid, 'balanced_accuracy'),n_trials=n_trials, timeout=timeout, catch=(ValueError,))

    if do_plot == 'True':
        fig = optuna.visualization.plot_parallel_coordinate(study)
        fig.write_image(full_path+'/training/SVM_ParamOptimization_'+str(i)+'.png')

    best_trial = study.best_trial

    # Train model using 'best' hyperparameters
    est = SVC()
    clf = clone(est).set_params(**best_trial.params)
    setattr(clf, 'random_state', randSeed)

    model = clf.fit(x_train, y_train)

    # Save model
    pickle.dump(model, open(full_path+'/training/pickledModels/SVM_'+str(i), 'wb'))

    # Prediction evaluation
    y_pred = clf.predict(x_test)

    metricList = classEval(y_test, y_pred)

    # Determine probabilities of class predictions for each test instance (this will be used much later in calculating an ROC curve)
    probas_ = model.predict_proba(x_test)

    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = metrics.roc_curve(y_test, probas_[:, 1])
    roc_auc = auc(fpr, tpr)

    # Compute Precision/Recall curve and AUC
    prec, recall, thresholds = metrics.precision_recall_curve(y_test, probas_[:, 1])
    prec, recall, thresholds = prec[::-1], recall[::-1], thresholds[::-1]
    prec_rec_auc = auc(recall, prec)
    ave_prec = metrics.average_precision_score(y_test, probas_[:, 1])

    # Feature Importance Estimates
    fi = computeImportances(clf, x_train, y_train, x_test, y_test, metricList[0])

    return [metricList, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, fi]

def objective_GB(trial, est, x_train, y_train, randSeed, hype_cv, param_grid, scoring_metric):
    params = {'loss': trial.suggest_categorical('loss', param_grid['loss']),
              'learning_rate': trial.suggest_loguniform('learning_rate', param_grid['learning_rate'][0],param_grid['learning_rate'][1]),
              'min_samples_leaf': trial.suggest_int('min_samples_leaf', param_grid['min_samples_leaf'][0],param_grid['min_samples_leaf'][1]),
              'max_depth': trial.suggest_int('max_depth', param_grid['max_depth'][0], param_grid['max_depth'][1]),
              'max_leaf_nodes': param_grid['max_leaf_nodes'][0],
              'tol': param_grid['tol'][0],
              'n_iter_no_change': trial.suggest_int('n_iter_no_change', param_grid['n_iter_no_change'][0],param_grid['n_iter_no_change'][1]),
              'validation_fraction': trial.suggest_discrete_uniform('validation_fraction',param_grid['validation_fraction'][0],param_grid['validation_fraction'][1],param_grid['validation_fraction'][2])}
    return hyper_eval(est, x_train, y_train, randSeed, hype_cv, params, scoring_metric)

def run_GB_full(x_train, y_train, x_test, y_test,randSeed,i,param_grid,n_trials,timeout,do_plot,full_path):
    # Run Hyperparameter sweep
    est = GradientBoostingClassifier()
    sampler = optuna.samplers.TPESampler(seed=randSeed)  # Make the sampler behave in a deterministic way.
    study = optuna.create_study(direction='maximize', sampler=sampler)
    optuna.logging.set_verbosity(optuna.logging.CRITICAL)
    study.optimize(lambda trial: objective_GB(trial, est, x_train, y_train, randSeed, 3, param_grid, 'balanced_accuracy'),n_trials=n_trials, timeout=timeout, catch=(ValueError,))

    if do_plot == 'True':
        fig = optuna.visualization.plot_parallel_coordinate(study)
        fig.write_image(full_path+'/training/GB_ParamOptimization_'+str(i)+'.png')

    best_trial = study.best_trial

    # Train model using 'best' hyperparameters
    est = GradientBoostingClassifier()
    clf = clone(est).set_params(**best_trial.params)
    setattr(clf, 'random_state', randSeed)

    model = clf.fit(x_train, y_train)

    # Save model
    pickle.dump(model, open(full_path+'/training/pickledModels/GB_'+str(i), 'wb'))

    # Prediction evaluation
    y_pred = clf.predict(x_test)

    metricList = classEval(y_test, y_pred)

    # Determine probabilities of class predictions for each test instance (this will be used much later in calculating an ROC curve)
    probas_ = model.predict_proba(x_test)

    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = metrics.roc_curve(y_test, probas_[:, 1])
    roc_auc = auc(fpr, tpr)

    # Compute Precision/Recall curve and AUC
    prec, recall, thresholds = metrics.precision_recall_curve(y_test, probas_[:, 1])
    prec, recall, thresholds = prec[::-1], recall[::-1], thresholds[::-1]
    prec_rec_auc = auc(recall, prec)
    ave_prec = metrics.average_precision_score(y_test, probas_[:, 1])

    # Feature Importance Estimates
    fi = clf.feature_importances_

    return metricList, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, fi

def objective_KN(trial, est, x_train, y_train, randSeed, hype_cv, param_grid, scoring_metric):
    params = {
        'n_neighbors': trial.suggest_int('n_neighbors', param_grid['n_neighbors'][0], param_grid['n_neighbors'][1]),
        'weights': trial.suggest_categorical('weights', param_grid['weights']),
        'p': trial.suggest_int('p', param_grid['p'][0], param_grid['p'][1]),
        'metric': trial.suggest_categorical('metric', param_grid['metric'])}
    return hyper_eval(est, x_train, y_train, randSeed, hype_cv, params, scoring_metric)

def run_KN_full(x_train, y_train, x_test, y_test,randSeed,i,param_grid,n_trials,timeout,do_plot,full_path):
    # Run Hyperparameter sweep
    est = KNeighborsClassifier()
    sampler = optuna.samplers.TPESampler(seed=randSeed)  # Make the sampler behave in a deterministic way.
    study = optuna.create_study(direction='maximize', sampler=sampler)
    optuna.logging.set_verbosity(optuna.logging.CRITICAL)
    study.optimize(lambda trial: objective_KN(trial, est, x_train, y_train, randSeed, 3, param_grid, 'balanced_accuracy'),n_trials=n_trials, timeout=timeout, catch=(ValueError,))

    if do_plot == 'True':
        fig = optuna.visualization.plot_parallel_coordinate(study)
        fig.write_image(full_path+'/training/KN_ParamOptimization_'+str(i)+'.png')

    best_trial = study.best_trial

    # Train model using 'best' hyperparameters
    est = KNeighborsClassifier()
    clf = clone(est).set_params(**best_trial.params)

    model = clf.fit(x_train, y_train)

    # Save model
    pickle.dump(model, open(full_path+'/training/pickledModels/KN_'+str(i), 'wb'))

    # Prediction evaluation
    y_pred = clf.predict(x_test)

    metricList = classEval(y_test, y_pred)

    # Determine probabilities of class predictions for each test instance (this will be used much later in calculating an ROC curve)
    probas_ = model.predict_proba(x_test)

    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = metrics.roc_curve(y_test, probas_[:, 1])
    roc_auc = auc(fpr, tpr)

    # Compute Precision/Recall curve and AUC
    prec, recall, thresholds = metrics.precision_recall_curve(y_test, probas_[:, 1])
    prec, recall, thresholds = prec[::-1], recall[::-1], thresholds[::-1]
    prec_rec_auc = auc(recall, prec)
    ave_prec = metrics.average_precision_score(y_test, probas_[:, 1])

    # Feature Importance Estimates
    fi = computeImportances(clf, x_train, y_train, x_test, y_test, metricList[0])

    return metricList, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, fi


def objective_ANN(trial, est, x_train, y_train, randSeed, hype_cv, param_grid, scoring_metric):
    params = {'activation': trial.suggest_categorical('activation', param_grid['activation']),
              'learning_rate': trial.suggest_categorical('learning_rate', param_grid['learning_rate']),
              'momentum': trial.suggest_uniform('momentum', param_grid['momentum'][0], param_grid['momentum'][1]),
              'solver': trial.suggest_categorical('solver', param_grid['solver']),
              'batch_size': trial.suggest_categorical('batch_size', param_grid['batch_size']),
              'alpha': trial.suggest_loguniform('alpha', param_grid['alpha'][0], param_grid['alpha'][1]),
              'max_iter': trial.suggest_categorical('max_iter', param_grid['max_iter'])}
    n_layers = trial.suggest_int('n_layers', param_grid['n_layers'][0], param_grid['n_layers'][1])
    layers = []
    for i in range(n_layers):
        layers.append(
            trial.suggest_int('n_units_l{}'.format(i), param_grid['layer_size'][0], param_grid['layer_size'][1]))
        params['hidden_layer_sizes'] = tuple(layers)

    return hyper_eval(est, x_train, y_train, randSeed, hype_cv, params, scoring_metric)

def run_ANN_full(x_train, y_train, x_test, y_test,randSeed,i,param_grid,n_trials,timeout,do_plot,full_path):
    # Run Hyperparameter sweep
    est = MLPClassifier()
    sampler = optuna.samplers.TPESampler(seed=randSeed)  # Make the sampler behave in a deterministic way.
    study = optuna.create_study(direction='maximize', sampler=sampler)
    optuna.logging.set_verbosity(optuna.logging.CRITICAL)
    study.optimize(lambda trial: objective_ANN(trial, est, x_train, y_train, randSeed, 3, param_grid, 'balanced_accuracy'),n_trials=n_trials, timeout=timeout, catch=(ValueError,))

    if do_plot == 'True':
        fig = optuna.visualization.plot_parallel_coordinate(study)
        fig.write_image(full_path+'/training/ANN_ParamOptimization_'+str(i)+'.png')

    best_trial = study.best_trial

    # Handle special parameter requirement for ANN
    layers = []
    for j in range(best_trial.params['n_layers']):
        layer_name = 'n_units_l' + str(j)
        layers.append(best_trial.params[layer_name])
        del best_trial.params[layer_name]

    best_trial.params['hidden_layer_sizes'] = tuple(layers)
    del best_trial.params['n_layers']

    # Train model using 'best' hyperparameters
    est = MLPClassifier()
    clf = clone(est).set_params(**best_trial.params)
    setattr(clf, 'random_state', randSeed)

    model = clf.fit(x_train, y_train)

    # Save model
    pickle.dump(model, open(full_path+'/training/pickledModels/ANN_'+str(i), 'wb'))

    # Prediction evaluation
    y_pred = clf.predict(x_test)

    metricList = classEval(y_test, y_pred)

    # Determine probabilities of class predictions for each test instance (this will be used much later in calculating an ROC curve)
    probas_ = model.predict_proba(x_test)

    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = metrics.roc_curve(y_test, probas_[:, 1])
    roc_auc = auc(fpr, tpr)

    # Compute Precision/Recall curve and AUC
    prec, recall, thresholds = metrics.precision_recall_curve(y_test, probas_[:, 1])
    prec, recall, thresholds = prec[::-1], recall[::-1], thresholds[::-1]
    prec_rec_auc = auc(recall, prec)
    ave_prec = metrics.average_precision_score(y_test, probas_[:, 1])

    # Feature Importance Estimates
    fi = computeImportances(clf, x_train, y_train, x_test, y_test, metricList[0])

    return [metricList, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, fi]

def objective_eLCS(trial, est, x_train, y_train, randSeed, hype_cv, param_grid, scoring_metric):
    params = {'learning_iterations': trial.suggest_categorical('learning_iterations', param_grid['learning_iterations']),
        'N': trial.suggest_categorical('N', param_grid['N']),
        'nu': trial.suggest_categorical('nu', param_grid['nu'])}
    return hyper_eval(est, x_train, y_train, randSeed, hype_cv, params, scoring_metric)

def run_eLCS_full(x_train, y_train, x_test, y_test,randSeed,i,param_grid,n_trials,timeout,do_plot,full_path):
    isSingle = True
    for key, value in param_grid.items():
        if len(value) > 1:
            isSingle = False

    est = eLCS()
    if not isSingle:
        # Run Hyperparameter sweep
        sampler = optuna.samplers.TPESampler(seed=randSeed)  # Make the sampler behave in a deterministic way.
        study = optuna.create_study(direction='maximize', sampler=sampler)
        optuna.logging.set_verbosity(optuna.logging.CRITICAL)
        study.optimize(lambda trial: objective_eLCS(trial, est, x_train, y_train, randSeed, 3, param_grid, 'balanced_accuracy'),n_trials=n_trials, timeout=timeout, catch=(ValueError,))

        if do_plot == 'True':
            fig = optuna.visualization.plot_parallel_coordinate(study)
            fig.write_image(full_path+'/training/eLCS_ParamOptimization_'+str(i)+'.png')

        best_trial = study.best_trial

        # Train model using 'best' hyperparameters
        clf = clone(est).set_params(**best_trial.params)
    else:
        params = copy.deepcopy(param_grid)
        for key, value in param_grid.items():
            params[key] = value[0]
        clf = clone(est).set_params(**params)
    setattr(clf, 'random_state', randSeed)
    model = clf.fit(x_train, y_train)

    # Save model
    pickle.dump(model, open(full_path+'/training/pickledModels/eLCS_'+str(i), 'wb'))

    # Prediction evaluation
    y_pred = clf.predict(x_test)

    metricList = classEval(y_test, y_pred)

    # Determine probabilities of class predictions for each test instance (this will be used much later in calculating an ROC curve)
    probas_ = model.predict_proba(x_test)

    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = metrics.roc_curve(y_test, probas_[:, 1])
    roc_auc = auc(fpr, tpr)

    # Compute Precision/Recall curve and AUC
    prec, recall, thresholds = metrics.precision_recall_curve(y_test, probas_[:, 1])
    prec, recall, thresholds = prec[::-1], recall[::-1], thresholds[::-1]
    prec_rec_auc = auc(recall, prec)
    ave_prec = metrics.average_precision_score(y_test, probas_[:, 1])

    # Feature Importance Estimates
    fi = clf.get_final_attribute_specificity_list()

    return [metricList, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, fi]

def objective_XCS(trial, est, x_train, y_train, randSeed, hype_cv, param_grid, scoring_metric):
    params = {'learning_iterations': trial.suggest_categorical('learning_iterations', param_grid['learning_iterations']),
        'N': trial.suggest_categorical('N', param_grid['N']),
        'nu': trial.suggest_categorical('nu', param_grid['nu'])}
    return hyper_eval(est, x_train, y_train, randSeed, hype_cv, params, scoring_metric)

def run_XCS_full(x_train, y_train, x_test, y_test,randSeed,i,param_grid,n_trials,timeout,do_plot,full_path):
    isSingle = True
    for key, value in param_grid.items():
        if len(value) > 1:
            isSingle = False
    est = XCS()
    if not isSingle:
        # Run Hyperparameter sweep
        sampler = optuna.samplers.TPESampler(seed=randSeed)  # Make the sampler behave in a deterministic way.
        study = optuna.create_study(direction='maximize', sampler=sampler)
        optuna.logging.set_verbosity(optuna.logging.CRITICAL)
        study.optimize(lambda trial: objective_XCS(trial, est, x_train, y_train, randSeed, 3, param_grid, 'balanced_accuracy'),n_trials=n_trials, timeout=timeout, catch=(ValueError,))

        if do_plot == 'True':
            fig = optuna.visualization.plot_parallel_coordinate(study)
            fig.write_image(full_path+'/training/XCS_ParamOptimization_'+str(i)+'.png')

        best_trial = study.best_trial

        # Train model using 'best' hyperparameters
        clf = clone(est).set_params(**best_trial.params)
    else:
        params = copy.deepcopy(param_grid)
        for key, value in param_grid.items():
            params[key] = value[0]
        clf = clone(est).set_params(**params)
    setattr(clf, 'random_state', randSeed)
    model = clf.fit(x_train, y_train)

    # Save model
    pickle.dump(model, open(full_path+'/training/pickledModels/XCS_'+str(i), 'wb'))

    # Prediction evaluation
    y_pred = clf.predict(x_test)

    metricList = classEval(y_test, y_pred)

    # Determine probabilities of class predictions for each test instance (this will be used much later in calculating an ROC curve)
    probas_ = model.predict_proba(x_test)

    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = metrics.roc_curve(y_test, probas_[:, 1])
    roc_auc = auc(fpr, tpr)

    # Compute Precision/Recall curve and AUC
    prec, recall, thresholds = metrics.precision_recall_curve(y_test, probas_[:, 1])
    prec, recall, thresholds = prec[::-1], recall[::-1], thresholds[::-1]
    prec_rec_auc = auc(recall, prec)
    ave_prec = metrics.average_precision_score(y_test, probas_[:, 1])

    # Feature Importance Estimates
    fi = clf.get_final_attribute_specificity_list()

    return [metricList, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, fi]

def objective_ExSTraCS(trial, est, x_train, y_train, randSeed, hype_cv, param_grid, scoring_metric):
    params = {'learning_iterations': trial.suggest_categorical('learning_iterations', param_grid['learning_iterations']),
        'N': trial.suggest_categorical('N', param_grid['N']),
        'nu': trial.suggest_categorical('nu', param_grid['nu'])}
    return hyper_eval(est, x_train, y_train, randSeed, hype_cv, params, scoring_metric)

def run_ExSTraCS_full(x_train, y_train, x_test, y_test,randSeed,i,param_grid,n_trials,timeout,do_plot,full_path):
    isSingle = True
    for key, value in param_grid.items():
        if len(value) > 1:
            isSingle = False

    est = ExSTraCS()
    if not isSingle:
        # Run Hyperparameter sweep
        sampler = optuna.samplers.TPESampler(seed=randSeed)  # Make the sampler behave in a deterministic way.
        study = optuna.create_study(direction='maximize', sampler=sampler)
        optuna.logging.set_verbosity(optuna.logging.CRITICAL)
        study.optimize(lambda trial: objective_ExSTraCS(trial, est, x_train, y_train, randSeed, 3, param_grid,'balanced_accuracy'), n_trials=n_trials, timeout=timeout,catch=(ValueError,))

        if do_plot == 'True':
            fig = optuna.visualization.plot_parallel_coordinate(study)
            fig.write_image(full_path+'/training/ExSTraCS_ParamOptimization_'+str(i)+'.png')

        best_trial = study.best_trial

        # Train model using 'best' hyperparameters
        clf = clone(est).set_params(**best_trial.params)
    else:
        params = copy.deepcopy(param_grid)
        for key, value in param_grid.items():
            params[key] = value[0]
        clf = clone(est).set_params(**params)
    setattr(clf, 'rule_compaction', None)
    setattr(clf, 'random_state', randSeed)

    # SET EXPERT Knowledge
    rbSample = np.random.choice(x_train.shape[0], min(2000, x_train.shape[0]), replace=False)
    newL = []
    for r in rbSample:
        newL.append(x_train[r])
    newL = np.array(newL)
    dataFeaturesR = np.delete(newL, -1, axis=1)
    dataPhenotypesR = newL[:, -1]

    relieff = ReliefF()
    relieff.fit(dataFeaturesR, dataPhenotypesR)
    scores = relieff.feature_importances_
    setattr(clf, 'expert_knowledge', scores)

    model = clf.fit(x_train, y_train)

    # Save model
    pickle.dump(model, open(full_path+'/training/pickledModels/ExSTraCS_'+str(i), 'wb'))

    # Prediction evaluation
    y_pred = clf.predict(x_test)

    metricList = classEval(y_test, y_pred)

    # Determine probabilities of class predictions for each test instance (this will be used much later in calculating an ROC curve)
    probas_ = model.predict_proba(x_test)

    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = metrics.roc_curve(y_test, probas_[:, 1])
    roc_auc = auc(fpr, tpr)

    # Compute Precision/Recall curve and AUC
    prec, recall, thresholds = metrics.precision_recall_curve(y_test, probas_[:, 1])
    prec, recall, thresholds = prec[::-1], recall[::-1], thresholds[::-1]
    prec_rec_auc = auc(recall, prec)
    ave_prec = metrics.average_precision_score(y_test, probas_[:, 1])

    # Feature Importance Estimates
    fi = clf.get_final_attribute_specificity_list()

    return [metricList, fpr, tpr, roc_auc, prec, recall, prec_rec_auc, ave_prec, fi]

def classEval(y_true, y_pred):
    # calculate and store evaluation metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    ac = accuracy_score(y_true, y_pred)
    bac = balanced_accuracy_score(y_true, y_pred)
    re = recall_score(y_true, y_pred)
    pr = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # calculate specificity
    if tn == 0 and fp == 0:
        sp = 0
    else:
        sp = tn / float(tn + fp)

    return [bac, ac, f1, re, sp, pr, tp, tn, fp, fn]

def computeImportances(clf, x_train, y_train, x_test, y_test, bac):
    #Reruns the model n times (once for each feature), and evaluates performance change as a standard of feature importance
    feature_count = len(x_train[0])
    FIbAccList = []
    for feature in range(feature_count):
        indexList = []
        indexList.extend(range(0, feature))
        indexList.extend(range(feature + 1, feature_count))

        #Create temporary training and testing sets
        tempTrain = pd.DataFrame(x_train)
        FIxTrainList = tempTrain.iloc[:, indexList].values

        tempTest = pd.DataFrame(x_test)
        FIxTestList = tempTest.iloc[:, indexList].values

        clf.fit(FIxTrainList, y_train)
        FIyPred = clf.predict(FIxTestList)

        FIbAccList.append(balanced_accuracy_score(y_test, FIyPred))

    #Lower balanced accuracy metric values suggest higher feature importance
    featureImpList = []
    for element in FIbAccList:
        if element > bac: #removal of feature yielded higher accuracy
            featureImpList.append(0) #worst importance
        else:
            featureImpList.append(bac - element)

    return featureImpList

def hyperparameters():
    param_grid = {}
    #######EDITABLE CODE################################################################################################
    # Logistic Regression
    param_grid_LR = {'penalty': ['l2', 'l1'],'C': [1e-5, 1e5],'dual': [True, False],
                     'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                     'class_weight': [None, 'balanced'],'max_iter': [10, 1000],'n_jobs': [-1]}

    # Decision Tree
    param_grid_DT = {'criterion': ['gini', 'entropy'],'splitter': ['best', 'random'],'max_depth': [1, 30],
                     'min_samples_split': [2, 50],'min_samples_leaf': [1, 50],'max_features': [None, 'auto', 'log2'],
                     'class_weight': [None, 'balanced']}

    # Random Forest
    param_grid_RF = {'n_estimators': [10, 1000],'criterion': ['gini', 'entropy'],'max_depth': [1, 30],
                     'min_samples_split': [2, 50],'min_samples_leaf': [1, 50],'max_features': [None, 'auto', 'log2'],
                     'bootstrap': [True],'oob_score': [False, True],'n_jobs': [-1],'class_weight': [None, 'balanced']}

    # XG Boost - note: class weight balance is included as option internally
    param_grid_XGB = {'booster': ['gbtree'],'objective': ['binary:logistic'],'verbosity': [0],'reg_lambda': [1e-8, 1.0],
                      'alpha': [1e-8, 1.0],'eta': [1e-8, 1.0],'gamma': [1e-8, 1.0],'max_depth': [1, 30],
                      'grow_policy': ['depthwise', 'lossguide'],'n_estimators': [10, 1000],'min_samples_split': [2, 50],
                      'min_samples_leaf': [1, 50],'subsample': [0.5, 1.0],'min_child_weight': [0.1, 10],
                      'colsample_bytree': [0.1, 1.0]}

    # LG Boost - note: class weight balance is included as option internally
    param_grid_LGB = {'objective': ['binary'],'metric': ['binary_logloss'],'verbosity': [-1],'boosting_type': ['gbdt'],
                      'num_leaves': [2, 256],'max_depth': [1, 30],'lambda_l1': [1e-8, 10.0],'lambda_l2': [1e-8, 10.0],
                      'feature_fraction': [0.4, 1.0],'bagging_fraction': [0.4, 1.0],'bagging_freq': [1, 7],
                      'min_child_samples': [5, 100],'n_estimators': [10, 1000]}

    # SVM
    param_grid_SVM = {'kernel': ['linear', 'poly', 'rbf'],'C': [0.1, 1000],'gamma': ['scale'],'degree': [1, 6],
                      'probability': [True],'class_weight': [None, 'balanced']}

    # ANN
    param_grid_ANN = {'n_layers': [1, 3],'layer_size': [1, 100],'activation': ['identity', 'logistic', 'tanh', 'relu'],
                      'learning_rate': ['constant', 'invscaling', 'adaptive'],'momentum': [.1, .9],
                      'solver': ['sgd', 'adam'],'batch_size': ['auto'],'alpha': [0.0001, 0.05],'max_iter': [200]}

    # ExSTraCS
    # param_grid_ExSTraCS = {'learning_iterations':[20000,50000,100000,200000],'N':[500,1000,2000],'nu':[1,5,10]}
    param_grid_ExSTraCS = {'learning_iterations': [5000,10000],'N': [1000],'nu': [1,10]}

    # eLCS
    # param_grid_eLCS = {'learning_iterations':[20000,50000,100000,200000],'N':[500,1000,2000],'nu':[1,5,10]}
    param_grid_eLCS = {'learning_iterations': [5000,10000],'N': [1000],'nu': [1,10]}

    # XCS
    # param_grid_XCS = {'learning_iterations':[20000,50000,100000,200000],'N':[500,1000,2000],'nu':[1,5,10]}
    param_grid_XCS = {'learning_iterations': [5000,10000],'N': [1000],'nu': [1,10]}

    # GB
    param_grid_GB = {'loss': ['deviance', 'exponential'], 'learning_rate': [1e-2, 1], 'min_samples_leaf': [1, 200],
                     'max_depth': [1, 10], 'max_leaf_nodes': [None], 'tol': [1e-7], 'n_iter_no_change': [1, 20],
                     'validation_fraction': [0.01, 0.31, 0.01]}

    # KN
    param_grid_KN = {'n_neighbors': [1, 100], 'weights': ['uniform', 'distance'], 'p': [1, 5],
                     'metric': ['euclidean', 'minkowski']}

    ####################################################################################################################
    param_grid['logistic_regression'] = param_grid_LR
    param_grid['decision_tree'] = param_grid_DT
    param_grid['random_forest'] = param_grid_RF
    param_grid['XGB'] = param_grid_XGB
    param_grid['LGB'] = param_grid_LGB
    param_grid['SVM'] = param_grid_SVM
    param_grid['ANN'] = param_grid_ANN
    param_grid['ExSTraCS'] = param_grid_ExSTraCS
    param_grid['eLCS'] = param_grid_eLCS
    param_grid['XCS'] = param_grid_XCS
    param_grid['naive_bayes'] = {}
    param_grid['gradient_boosting'] = param_grid_GB
    param_grid['k_neighbors'] = param_grid_KN
    return param_grid

if __name__ == '__main__':
    job(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],int(sys.argv[5]),int(sys.argv[6]),int(sys.argv[7]),sys.argv[8],sys.argv[9],sys.argv[10],int(sys.argv[11]),int(sys.argv[12]))