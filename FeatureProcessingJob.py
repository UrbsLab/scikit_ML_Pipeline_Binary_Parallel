import sys
import random
import numpy as np
import time
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from skrebate import MultiSURF
import csv
import pickle
import os

###Mutual Information###################################################################################################
def miJob(trainfile_path,random_state,experiment_path,class_label,instance_label):
    job_start_time = time.time()
    random.seed(random_state)
    np.random.seed(random_state)

    dataset_name = trainfile_path.split('/')[-3]
    data = pd.read_csv(trainfile_path,sep=',')
    dataFeatures = data.drop(class_label,axis=1).values
    dataPhenotypes = data[class_label].values
    headers = data.columns.values.tolist()
    headers.remove(class_label)
    if instance_label != 'None':
        headers.remove(instance_label)
    cvCount = trainfile_path.split('/')[-1].split("_")[-2]
    scores,scoreDict,score_sorted_features = run_mi(dataFeatures,dataPhenotypes,cvCount,dataset_name,experiment_path,random_state,headers)

    #Save CV MI Scores to pickled file
    if not os.path.exists(experiment_path + '/' + dataset_name + "/MutualInformation/pickledForPhase3"):
        os.mkdir(experiment_path + '/' + dataset_name + "/MutualInformation/pickledForPhase3")

    outfile = open(experiment_path + '/' + dataset_name + "/MutualInformation/pickledForPhase3/"+str(cvCount),'wb')
    pickle.dump([scores,scoreDict,score_sorted_features],outfile)
    outfile.close()

    #Save Runtime
    runtime_file = open(experiment_path + '/' + dataset_name + '/runtime/runtime_MutualInformation_CV_'+str(cvCount)+'.txt', 'w')
    runtime_file.write(str(time.time() - job_start_time))
    runtime_file.close()

    # Print completion
    print(dataset_name+" CV"+str(cvCount)+" phase 2 Mutual Information Evaluation complete")

def sort_save_fi_scores(scores, ordered_feature_names, filename):
    # Put list of scores in dictionary
    scoreDict = {}
    i = 0
    for each in ordered_feature_names:
        scoreDict[each] = scores[i]
        i += 1

    # Sort features by decreasing score
    score_sorted_features = sorted(scoreDict, key=lambda x: scoreDict[x], reverse=True)

    # Save scores to 'formatted' file
    with open(filename,mode='w') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["Sorted Mutual Information Scores"])
        for k in score_sorted_features:
            writer.writerow([k,scoreDict[k]])
    file.close()

    return scoreDict, score_sorted_features

def run_mi(xTrain, yTrain, cv_count, data_name, output_folder, randSeed, ordered_feature_names):
    # Run mutual information
    filename = output_folder + '/' + data_name + "/MutualInformation/scores_cv_" + str(cv_count) + '.csv'
    scores = mutual_info_classif(xTrain, yTrain, random_state=randSeed)

    scoreDict, score_sorted_features = sort_save_fi_scores(scores, ordered_feature_names, filename)

    return scores, scoreDict, score_sorted_features

###MultiSURF############################################################################################################
def msJob(trainfile_path,instance_subset,random_state,experiment_path,class_label,instance_label):
    job_start_time = time.time()
    random.seed(random_state)
    np.random.seed(random_state)

    dataset_name = trainfile_path.split('/')[-3]
    data = pd.read_csv(trainfile_path, sep=',')
    dataFeatures = data.drop(class_label, axis=1).values
    dataPhenotypes = data[class_label].values
    headers = data.columns.values.tolist()
    headers.remove(class_label)
    if instance_label != 'None':
        headers.remove(instance_label)
    cvCount = trainfile_path.split('/')[-1].split("_")[-2]

    formatted = np.insert(dataFeatures, dataFeatures.shape[1], dataPhenotypes, 1)
    choices = np.random.choice(formatted.shape[0],min(instance_subset,formatted.shape[0]),replace=False)
    newL = []
    for i in choices:
        newL.append(formatted[i])
    formatted = np.array(newL)
    dataFeatures = np.delete(formatted,-1,axis=1)
    dataPhenotypes = formatted[:,-1]

    scores,scoreDict,score_sorted_features = run_multisurf(dataFeatures,dataPhenotypes,cvCount,dataset_name,experiment_path,headers)

    # Save CV MS Scores to pickled file
    if not os.path.exists(experiment_path + '/' + dataset_name + "/MultiSURF/pickledForPhase3"):
        os.mkdir(experiment_path + '/' + dataset_name + "/MultiSURF/pickledForPhase3")

    outfile = open(experiment_path + '/' + dataset_name + "/MultiSURF/pickledForPhase3/" + str(cvCount),'wb')
    pickle.dump([scores, scoreDict, score_sorted_features], outfile)
    outfile.close()

    # Save Runtime
    runtime_file = open(experiment_path + '/' + dataset_name + '/runtime/runtime_MultiSURF_CV_' + str(cvCount) + '.txt', 'w')
    runtime_file.write(str(time.time() - job_start_time))
    runtime_file.close()

    # Print completion
    print(dataset_name + " CV" + str(cvCount) + " phase 2 MultiSURF Evaluation complete")

def run_multisurf(xTrain, yTrain, cv_count, data_name, output_folder, ordered_feature_names):
    # Run multisurf
    filename = output_folder + '/' + data_name + "/MultiSURF/scores_cv_" + str(cv_count) + '.csv'

    clf = MultiSURF().fit(xTrain, yTrain)
    scores = clf.feature_importances_

    scoreDict, score_sorted_features = sort_save_fi_scores(scores, ordered_feature_names, filename)

    return scores, scoreDict, score_sorted_features

########################################################################################################################
if __name__ == '__main__':
    if sys.argv[-1] == 'mi':
        miJob(sys.argv[1],int(sys.argv[2]),sys.argv[3],sys.argv[4],sys.argv[5])
    if sys.argv[-1] == 'ms':
        msJob(sys.argv[1],sys.argv[2],int(sys.argv[3]),sys.argv[4],sys.argv[5],sys.argv[6])