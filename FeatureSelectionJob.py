import time
import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle
import copy
import pandas as pd
import os
import csv
import sys

def job(full_path,do_mutual_info,do_multiSURF,max_features_to_keep,filter_poor_features,top_results,graph_scores,class_label,instance_label):
    job_start_time = time.time()

    selected_feature_lists = {}
    meta_feature_ranks = {}
    cvPartitions = int(len(glob.glob(full_path+'/CVDatasets/*.csv'))/2)
    algorithms = []

    #Graph Average FI scores for MI
    if do_mutual_info == 'TRUE' or do_mutual_info == 'True':
        algorithms.append('Mutual Information')
        counter = 0
        cv_keep_list = []
        feature_name_ranks = []
        for scoreInfo in glob.glob(full_path+"/MutualInformation/pickledForPhase3/*"):
            file = open(scoreInfo, 'rb')
            rawData = pickle.load(file)
            file.close()

            scoreDict = rawData[1]
            score_sorted_features = rawData[2]
            feature_name_ranks.append(score_sorted_features)

            if counter == 0:
                scoreSum = copy.deepcopy(scoreDict)
            else:
                for each in rawData[1]:
                    scoreSum[each] += scoreDict[each]
            counter += 1

            keep_list = []
            for each in scoreDict:
                if scoreDict[each] > 0:
                    keep_list.append(each)
            cv_keep_list.append(keep_list)
        selected_feature_lists['Mutual Information'] = cv_keep_list
        meta_feature_ranks['Mutual Information'] = feature_name_ranks

        if graph_scores == 'True':
            reportTopFS(scoreSum,"Mutual Information",cvPartitions,top_results,full_path)

    # Graph Average FI scores for MS
    if do_multiSURF == 'TRUE' or do_multiSURF == 'True':
        algorithms.append('MultiSURF')
        counter = 0
        cv_keep_list = []
        feature_name_ranks = []
        for scoreInfo in glob.glob(full_path + "/MultiSURF/pickledForPhase3/*"):
            file = open(scoreInfo, 'rb')
            rawData = pickle.load(file)
            file.close()

            scoreDict = rawData[1]
            score_sorted_features = rawData[2]
            feature_name_ranks.append(score_sorted_features)

            if counter == 0:
                scoreSum = copy.deepcopy(scoreDict)
            else:
                for each in rawData[1]:
                    scoreSum[each] += scoreDict[each]
            counter += 1

            keep_list = []
            for each in scoreDict:
                if scoreDict[each] > 0:
                    keep_list.append(each)
            cv_keep_list.append(keep_list)
        selected_feature_lists['MultiSURF'] = cv_keep_list
        meta_feature_ranks['MultiSURF'] = feature_name_ranks

        if graph_scores == 'True':
            reportTopFS(scoreSum, "MultiSURF", cvPartitions, top_results, full_path)

    #Filter Scores and replace old CV files
    cv_selected_list = selectFeatures(algorithms,cvPartitions,selected_feature_lists,max_features_to_keep,meta_feature_ranks)
    dataset_name = full_path.split('/')[-1]
    if filter_poor_features:
        genFilteredDatasets(cv_selected_list,class_label,instance_label,cvPartitions,full_path+'/CVDatasets',dataset_name)

    # Save Runtime
    runtime_file = open(full_path + '/runtime/runtime_FeatureSelection.txt', 'w')
    runtime_file.write(str(time.time() - job_start_time))
    runtime_file.close()

    # Print completion
    print(dataset_name + " phase 3 complete")
    experiment_path = '/'.join(full_path.split('/')[:-1])
    job_file = open(experiment_path + '/jobsCompleted/job_featureSelection_' + dataset_name + '.txt', 'w')
    job_file.write('complete')
    job_file.close()

def reportTopFS(scoreSum, algorithm, cv_partitions, topResults,full_path):
    # Make the sum of scores an average
    for v in scoreSum:
        scoreSum[v] = scoreSum[v] / float(cv_partitions)

    # Sort averages (decreasing order and print top 'n' and plot top 'n'
    f_names = []
    f_scores = []
    for each in scoreSum:
        f_names.append(each)
        f_scores.append(scoreSum[each])

    names_scores = {'Names': f_names, 'Scores': f_scores}
    ns = pd.DataFrame(names_scores)
    ns = ns.sort_values(by='Scores', ascending=False)

    # Select top 'n' to report and plot
    ns = ns.head(topResults)

    # Visualize sorted feature scores
    ns['Scores'].plot(kind='barh', figsize=(6, 12))
    plt.ylabel('Features')
    plt.xlabel(str(algorithm) + ' Score')
    plt.yticks(np.arange(len(ns['Names'])), ns['Names'])
    plt.title('Sorted ' + str(algorithm) + ' Scores')
    if algorithm == 'Mutual Information':
        algorithm = "MutualInformation"
    plt.savefig((full_path+"/"+algorithm+"/AverageScores.png"), bbox_inches="tight")
    plt.close('all')

def selectFeatures(algorithms, cv_partitions, selectedFeatureLists, maxFeaturesToKeep, metaFeatureRanks):
    cv_Selected_List = []  # list of selected features for each cv
    numAlgorithms = len(algorithms)
    if numAlgorithms > 1:  # 'Interesting' features determined by union of feature selection results (from different algorithms)
        for i in range(cv_partitions):
            unionList = selectedFeatureLists[algorithms[0]][i]  # grab first algorithm's lists
            # Determine union
            for j in range(1, numAlgorithms):  # number of union comparisons
                unionList = list(set(unionList) | set(selectedFeatureLists[algorithms[j]][i]))

            if len(unionList) > maxFeaturesToKeep:  # Apply further filtering if more than max features remains
                # Create score list dictionary with indexes in union list
                newFeatureList = []
                k = 0
                while len(newFeatureList) < maxFeaturesToKeep:
                    for each in metaFeatureRanks:
                        targetFeature = metaFeatureRanks[each][i][k]
                        if not targetFeature in newFeatureList:
                            newFeatureList.append(targetFeature)
                        if len(newFeatureList) < maxFeaturesToKeep:
                            break
                    k += 1
                unionList = newFeatureList
            unionList.sort()  # Added to ensure script random seed reproducibility
            cv_Selected_List.append(unionList)

    else:  # Only one algorithm applied
        for i in range(cv_partitions):
            featureList = selectedFeatureLists[algorithms[0]][i]  # grab first algorithm's lists

            if len(featureList) > maxFeaturesToKeep:  # Apply further filtering if more than max features remains
                # Create score list dictionary with indexes in union list
                newFeatureList = []
                k = 0
                while len(newFeatureList) < maxFeaturesToKeep:
                    targetFeature = metaFeatureRanks[algorithms[0]][i][k]
                    newFeatureList.append(targetFeature)
                    k += 1
                featureList = newFeatureList
            cv_Selected_List.append(featureList)

    return cv_Selected_List

def genFilteredDatasets(cv_Selected_List, outcomeLabel, instLabel,cv_partitions,path_to_csv,dataset_name):
    #create lists to hold training and testing set dataframes.
    trainList = []
    testList = []

    for i in range(cv_partitions):
        #Load training partition
        trainSet = pd.read_csv(path_to_csv+'/'+dataset_name+'_CV_' + str(i) +"_Train.csv", na_values='NA', sep = ",")
        trainList.append(trainSet)

        #Load testing partition
        testSet = pd.read_csv(path_to_csv+'/'+dataset_name+'_CV_' + str(i) +"_Test.csv", na_values='NA', sep = ",")
        testList.append(testSet)

        #Training datasets
        labelList = [outcomeLabel]
        if instLabel != 'None':
            labelList.append(instLabel)
        labelList = labelList + cv_Selected_List[i]

        td_train = trainList[i][labelList]
        td_test = testList[i][labelList]

        #Remove old CV files
        os.remove(path_to_csv+'/'+dataset_name+'_CV_' + str(i) +"_Train.csv")
        os.remove(path_to_csv+'/'+dataset_name+'_CV_' + str(i) + "_Test.csv")

        #Write new CV files
        with open(path_to_csv+'/'+dataset_name+'_CV_' + str(i) +"_Train.csv",mode='w') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(td_train.columns.values.tolist())
            for row in td_train.values:
                writer.writerow(row)
        file.close()

        with open(path_to_csv+'/'+dataset_name+'_CV_' + str(i) +"_Test.csv",mode='w') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(td_test.columns.values.tolist())
            for row in td_test.values:
                writer.writerow(row)
        file.close()

if __name__ == '__main__':
    job(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], int(sys.argv[6]), int(sys.argv[7]),sys.argv[8], sys.argv[9])
