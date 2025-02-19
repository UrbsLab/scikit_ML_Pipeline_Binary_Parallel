import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as scs
import random
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from ExploratoryAnalysisJob import identifyCategoricalFeatures
import csv
import time
import pickle

'''Phase 2 of Machine Learning Analysis Pipeline:'''

def job(cv_train_path,cv_test_path,experiment_path,scale_data,impute_data,overwrite_cv,categorical_cutoff,class_label,instance_label,random_state):

    ##EDITABLE CODE#####################################################################################################
    categorical_attribute_headers = []
    ####################################################################################################################

    job_start_time = time.time()
    random.seed(random_state)
    np.random.seed(random_state)

    #Grab path name components
    dataset_name = cv_train_path.split('/')[-3]
    cvCount = cv_train_path.split('/')[-1].split("_")[-2]

    if not os.path.exists(experiment_path + '/' + dataset_name + '/exploratory/scale_impute'):
        os.mkdir(experiment_path + '/' + dataset_name + '/exploratory/scale_impute')

    #Load datasets
    data_train = pd.read_csv(cv_train_path,na_values='NA',sep=',')
    data_test = pd.read_csv(cv_test_path,na_values='NA',sep=',')
    #data_train[class_label] = data_train[class_label].astype(dtype='int64')
    #data_test[class_label] = data_test[class_label].astype(dtype='int64')

    header = data_train.columns.values.tolist()
    header.remove(class_label)
    if instance_label != 'None':
        header.remove(instance_label)

    #Identify categorical variables in dataset
    if len(categorical_attribute_headers) == 0:
        if instance_label == "None":
            x_data = data_train.drop([class_label],axis=1)
        else:
            x_data = data_train.drop([class_label,instance_label], axis=1)
        categorical_variables = identifyCategoricalFeatures(x_data,categorical_cutoff)
    else:
        categorical_variables = categorical_attribute_headers

    scale_data = scale_data == 'True'
    impute_data = impute_data == 'True'

    #Scale Data
    if scale_data:
        data_train,data_test, scaler = dataScaling(data_train,data_test,class_label,instance_label,header)
        outfile = open(experiment_path + '/' + dataset_name + '/exploratory/scale_impute/scaler_cv'+str(cvCount), 'wb')
        pickle.dump(scaler, outfile)
        outfile.close()

    #Impute Missing Values in Training and Testing Data
    if impute_data:
        data_train,data_test,imputer,mode_dict = imputeCVData(class_label,instance_label,categorical_variables,data_train,data_test,random_state,header)
        outfile = open(experiment_path + '/' + dataset_name + '/exploratory/scale_impute/ordinal_imputer_cv' + str(cvCount),'wb')
        pickle.dump(imputer, outfile)
        outfile.close()
        outfile = open(experiment_path + '/' + dataset_name + '/exploratory/scale_impute/categorical_imputer_cv' + str(cvCount),'wb')
        pickle.dump(mode_dict, outfile)
        outfile.close()

    if overwrite_cv == 'True':
        #Remove old CV files
        os.remove(cv_train_path)
        os.remove(cv_test_path)
    else:
        #Rename old CV files
        os.rename(cv_train_path,experiment_path + '/' + dataset_name + '/CVDatasets/'+dataset_name+'_CVOnly_' + str(cvCount) +"_Train.csv")
        os.rename(cv_test_path,experiment_path + '/' + dataset_name + '/CVDatasets/'+dataset_name+'_CVOnly_' + str(cvCount) +"_Test.csv")

    #Write new CV files
    with open(cv_train_path,mode='w') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(data_train.columns.values.tolist())
        for row in data_train.values:
            writer.writerow(row)
    file.close()

    with open(cv_test_path,mode='w') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(data_test.columns.values.tolist())
        for row in data_test.values:
            writer.writerow(row)
    file.close()

    #Save Runtime
    runtime_file = open(experiment_path + '/' + dataset_name + '/runtime/runtime_preprocessing_'+str(cvCount)+'.txt','w')
    runtime_file.write(str(time.time()-job_start_time))
    runtime_file.close()

    #Print completion
    print(dataset_name+" phase 2 complete")
    job_file = open(experiment_path + '/jobsCompleted/job_preprocessing_'+dataset_name+'_'+str(cvCount)+'.txt', 'w')
    job_file.write('complete')
    job_file.close()

###################################
def dataScaling(df,data_test,class_label,instance_label,header):
    scale_train_df = None
    scale_test_df = None

    if instance_label == None or instance_label == 'None':
        x_train = df.drop([class_label], axis=1)
    else:
        x_train = df.drop([class_label, instance_label], axis=1)
        inst_train = df[instance_label]  # pull out instance labels in case they include text
    y_train = df[class_label]

    # Scale features (x)
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train_scaled = pd.DataFrame(scaler.transform(x_train), columns=x_train.columns)

    # Recombine x and y
    if instance_label == None or instance_label == 'None':
        scale_train_df = pd.concat([pd.DataFrame(y_train, columns=[class_label]), pd.DataFrame(x_train_scaled, columns=header)],axis=1, sort=False)
    else:
        scale_train_df = pd.concat([pd.DataFrame(y_train, columns=[class_label]), pd.DataFrame(inst_train, columns=[instance_label]),pd.DataFrame(x_train_scaled, columns=header)], axis=1, sort=False)

    # Scale corresponding testing dataset
    df = data_test
    if instance_label == None or instance_label == 'None':
        x_test = df.drop([class_label], axis=1)
    else:
        x_test = df.drop([class_label, instance_label], axis=1)
        inst_test = df[instance_label]  # pull out instance labels in case they include text
    y_test = df[class_label]

    # Scale features (x)
    x_test_scaled = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns)

    # Recombine x and y
    if instance_label == None or instance_label == 'None':
        scale_test_df = pd.concat([pd.DataFrame(y_test, columns=[class_label]), pd.DataFrame(x_test_scaled, columns=header)],axis=1, sort=False)
    else:
        scale_test_df = pd.concat([pd.DataFrame(y_test, columns=[class_label]), pd.DataFrame(inst_test, columns=[instance_label]),pd.DataFrame(x_test_scaled, columns=header)], axis=1, sort=False)

    return scale_train_df, scale_test_df, scaler

###################################
def imputeCVData(class_label,instance_label,categorical_variables,data_train,data_test,random_state,header):
    # Begin by imputing categorical variables with simple 'mode' imputation
    mode_dict = {}
    for c in data_train.columns:
        if c in categorical_variables:
            train_mode = data_train[c].mode().iloc[0]
            data_train[c].fillna(train_mode, inplace=True)
            mode_dict[c] = train_mode
    for c in data_test.columns:
        if c in categorical_variables:
            data_test[c].fillna(mode_dict[c], inplace=True)

    # Now impute remaining ordinal variables
    if instance_label == None or instance_label == 'None':
        x_train = data_train.drop([class_label], axis=1).values
        x_test = data_test.drop([class_label], axis=1).values
    else:
        x_train = data_train.drop([class_label, instance_label], axis=1).values
        x_test = data_test.drop([class_label, instance_label], axis=1).values

        inst_train = data_train[instance_label].values  # pull out instance labels in case they include text
        inst_test = data_test[instance_label].values

    y_train = data_train[class_label].values
    y_test = data_test[class_label].values

    # Impute features (x)
    imputer = IterativeImputer(random_state=random_state,max_iter=30).fit(x_train)
    x_new_train = imputer.transform(x_train)
    x_new_test = imputer.transform(x_test)

    # Recombine x and y
    if instance_label == None or instance_label == 'None':
        data_train = pd.concat([pd.DataFrame(y_train, columns=[class_label]), pd.DataFrame(x_new_train, columns=header)],axis=1, sort=False)
        data_test = pd.concat([pd.DataFrame(y_test, columns=[class_label]), pd.DataFrame(x_new_test, columns=header)], axis=1, sort=False)
    else:
        data_train = pd.concat([pd.DataFrame(y_train, columns=[class_label]), pd.DataFrame(inst_train, columns=[instance_label]),pd.DataFrame(x_new_train, columns=header)], axis=1, sort=False)
        data_test = pd.concat([pd.DataFrame(y_test, columns=[class_label]), pd.DataFrame(inst_test, columns=[instance_label]), pd.DataFrame(x_new_test, columns=header)], axis=1, sort=False)

    return data_train,data_test,imputer,mode_dict

###################################

if __name__ == '__main__':
    job(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[6],int(sys.argv[7]),sys.argv[8],sys.argv[9],int(sys.argv[10]))
