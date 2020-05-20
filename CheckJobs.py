

import sys
import os
import argparse
import glob
import DataPreprocessingMain
import FeatureProcessingMain
import FeatureSelectionMain
import ModelMain
import StatsMain
import pandas as pd
import shutil

'''Sample Run Command (This module is designed to run independently from the 5 phases. Thus, it does not depend on metadata.csv. Arguments must be restated):
python CheckJobs.py 0 --output-path /Users/robert/Desktop/outputs --experiment-name test1
'''

def printIncompleteJobs(argv):
    # Parse arguments
    default_arg = 'True'
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--output-path', dest='output_path', type=str, help='path to output directory')
    parser.add_argument('--experiment-name', dest='experiment_name', type=str, help='name of experiment (no spaces)')

    parser.add_argument('--cv', dest='cv_partitions', type=int, help='number of CV partitions',default=3)

    parser.add_argument('--do-mutual-info', dest='do_mutual_info', type=str,default="True")
    parser.add_argument('--do-multiSURF', dest='do_multiSURF', type=str, default="True")

    parser.add_argument('--do-LR', dest='do_LR', type=str, default=default_arg)
    parser.add_argument('--do-DT', dest='do_DT', type=str, default=default_arg)
    parser.add_argument('--do-RF', dest='do_RF', type=str, default=default_arg)
    parser.add_argument('--do-NB', dest='do_NB', type=str, default=default_arg)
    parser.add_argument('--do-XGB', dest='do_XGB', type=str, default=default_arg)
    parser.add_argument('--do-LGB', dest='do_LGB', type=str, default=default_arg)
    parser.add_argument('--do-SVM', dest='do_SVM', type=str, default=default_arg)
    parser.add_argument('--do-ANN', dest='do_ANN', type=str, default=default_arg)
    parser.add_argument('--do-ExSTraCS', dest='do_ExSTraCS', type=str, default=default_arg)
    parser.add_argument('--do-eLCS', dest='do_eLCS', type=str, default=default_arg)
    parser.add_argument('--do-XCS', dest='do_XCS', type=str, default=default_arg)

    options = parser.parse_args(argv[2:])
    output_path = options.output_path
    experiment_name = options.experiment_name
    do_mutual_info = options.do_mutual_info
    do_multiSURF = options.do_multiSURF
    cv_partitions = options.cv_partitions

    algorithms = []
    if options.do_LR == 'True':
        algorithms.append("logistic_regression")
    if options.do_DT == 'True':
        algorithms.append("decision_tree")
    if options.do_RF == 'True':
        algorithms.append('random_forest')
    if options.do_NB == 'True':
        algorithms.append('naive_bayes')
    if options.do_XGB == 'True':
        algorithms.append('XGB')
    if options.do_LGB == 'True':
        algorithms.append('LGB')
    if options.do_SVM == 'True':
        algorithms.append('SVM')
    if options.do_ANN == 'True':
        algorithms.append('ANN')
    if options.do_ExSTraCS == 'True':
        algorithms.append('ExSTraCS')
    if options.do_eLCS == 'True':
        algorithms.append('eLCS')
    if options.do_XCS == 'True':
        algorithms.append('XCS')

    abbrev = {'logistic_regression':'LR','decision_tree':'DT','random_forest':'RF','naive_bayes':'NB','XGB':'XGB','LGB':'LGB','ANN':'ANN','SVM':'SVM','ExSTraCS':'ExSTraCS','eLCS':'eLCS','XCS':'XCS'}

    # Argument checks
    if not os.path.exists(output_path):
        raise Exception("Output path must exist (from phase 1) before check can begin")

    if not os.path.exists(output_path + '/' + experiment_name):
        raise Exception("Experiment must exist (from phase 1) before check can begin")

    datasets = os.listdir(output_path + "/" + experiment_name)
    datasets.remove('logs')
    datasets.remove('jobs')
    datasets.remove('jobsCompleted')
    if 'metadata.csv' in datasets:
        datasets.remove('metadata.csv')
    if 'DatasetComparisons' in datasets:
        datasets.remove('DatasetComparisons')

    phase1Jobs = []
    for dataset in datasets:
        phase1Jobs.append('job_preprocessing_'+dataset+'.txt')

    phase2Jobs = []
    for dataset in datasets:
        for cv in range(cv_partitions):
            if do_multiSURF:
                phase2Jobs.append('job_multiSURF_' + dataset + '_' + str(cv) + '.txt')
            if do_mutual_info:
                phase2Jobs.append('job_mutualInformation_' + dataset + '_' + str(cv) + '.txt')

    phase3Jobs = []
    for dataset in datasets:
        phase3Jobs.append('job_featureSelection_' + dataset + '.txt')

    phase4Jobs = []
    for dataset in datasets:
        for cv in range(cv_partitions):
            for algorithm in algorithms:
                phase4Jobs.append('job_model_' + dataset + '_' + str(cv) +'_' +abbrev[algorithm]+'.txt')

    phase5Jobs = []
    for dataset in datasets:
        phase5Jobs.append('job_stats_' + dataset + '.txt')

    print("Phase 1 Jobs Not Completed:")
    for filename in glob.glob(output_path + "/" + experiment_name+'/jobsCompleted/job_preprocessing*'):
        ref = filename.split('/')[-1]
        phase1Jobs.remove(ref)
    for job in phase1Jobs:
        print(job)
    if len(phase1Jobs) == 0:
        print("All Phase 1 Jobs Completed")
    print()

    print("Phase 2 Jobs Not Completed:")
    for filename in glob.glob(output_path + "/" + experiment_name + '/jobsCompleted/job_mu*'):
        ref = filename.split('/')[-1]
        phase2Jobs.remove(ref)
    for job in phase2Jobs:
        print(job)
    if len(phase2Jobs) == 0:
        print("All Phase 2 Jobs Completed")
    print()

    print("Phase 3 Jobs Not Completed:")
    for filename in glob.glob(output_path + "/" + experiment_name + '/jobsCompleted/job_featureSelection*'):
        ref = filename.split('/')[-1]
        phase3Jobs.remove(ref)
    for job in phase3Jobs:
        print(job)
    if len(phase3Jobs) == 0:
        print("All Phase 3 Jobs Completed")
    print()

    print("Phase 4 Jobs Not Completed:")
    for filename in glob.glob(output_path + "/" + experiment_name + '/jobsCompleted/job_model*'):
        ref = filename.split('/')[-1]
        phase4Jobs.remove(ref)
    for job in phase4Jobs:
        print(job)
    if len(phase4Jobs) == 0:
        print("All Phase 4 Jobs Completed")
    print()

    print("Phase 5 Jobs Not Completed:")
    for filename in glob.glob(output_path + "/" + experiment_name + '/jobsCompleted/job_stats*'):
        ref = filename.split('/')[-1]
        phase5Jobs.remove(ref)
    for job in phase5Jobs:
        print(job)
    if len(phase5Jobs) == 0:
        print("All Phase 5 Jobs Completed")
    print()


'''Sample Run Command (This module is designed to AFTER phase 1 has been run. Thus, it may depend on metadata.csv):
python CheckJobs.py 1 --data-path /Users/robert/Desktop/Datasets/Multiplexer6.csv --output-path /Users/robert/Desktop/outputs --experiment-name test1
'''
def runPhase1Job(argv):
    # Parse arguments
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--data-path', dest='data_path', type=str, help='path to dataset in question')
    parser.add_argument('--output-path', dest='output_path', type=str, help='path to output directory')
    parser.add_argument('--experiment-name', dest='experiment_name', type=str, help='name of experiment (no spaces)')
    parser.add_argument('--cv', dest='cv_partitions', type=int, help='number of CV partitions', default=3)
    parser.add_argument('--partition-method', dest='partition_method', type=str, help='S or R or M', default="S")
    parser.add_argument('--scale-data', dest='scale_data', type=str, default="True")
    parser.add_argument('--impute-data', dest='impute_data', type=str, default="True")
    parser.add_argument('--categorical-cutoff', dest='categorical_cutoff', type=int, default=10)
    parser.add_argument('--class-label', dest='class_label', type=str, help='outcome label of all datasets',
                        default="Class")
    parser.add_argument('--instance-label', dest='instance_label', type=str, default="")
    parser.add_argument('--match-label', dest='match_label', type=str, default="")
    parser.add_argument('--export-initial-analysis', dest='export_initial_analysis', type=str, default="True")
    parser.add_argument('--export-feature-correlations', dest='export_feature_correlations', type=str, default="True")
    parser.add_argument('--export-univariate', dest='export_univariate', type=str, default="True")
    parser.add_argument('--random-state', dest='random_state', type=int, default=42)

    options = parser.parse_args(argv[2:])
    data_path = options.data_path
    output_path = options.output_path
    experiment_name = options.experiment_name
    cv_partitions = options.cv_partitions
    partition_method = options.partition_method
    scale_data = options.scale_data
    impute_data = options.impute_data
    categorical_cutoff = options.categorical_cutoff
    class_label = options.class_label
    if options.instance_label == '':
        instance_label = 'None'
    else:
        instance_label = options.instance_label
    if options.match_label == '':
        match_label = 'None'
    else:
        match_label = options.match_label
    export_initial_analysis = options.export_initial_analysis
    export_feature_correlations = options.export_feature_correlations
    export_univariate = options.export_univariate
    random_state = options.random_state
    DataPreprocessingMain.submitClusterJob(data_path,output_path+'/'+experiment_name,cv_partitions,partition_method,scale_data,impute_data,categorical_cutoff,
                                           export_initial_analysis,export_feature_correlations,export_univariate,class_label,instance_label,match_label,random_state)

'''Sample Run Command (This module is designed to AFTER phase 2 has been run and assumes phase 1 ran correctly. Thus, it may depend on metadata.csv):
python CheckJobs.py 2a --output-path /Users/robert/Desktop/outputs --experiment-name test1 --cv 0 --dataset-name Multiplexer6
'''
def runPhase2aJob(argv): #Mutual Information
    # Parse arguments
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--output-path', dest='output_path', type=str, help='path to output directory')
    parser.add_argument('--experiment-name', dest='experiment_name', type=str, help='name of experiment (no spaces)')
    parser.add_argument('--do-mutual-info', dest='do_mutual_info', type=str, help='do mutual information analysis',default="True")
    parser.add_argument('--do-multiSURF', dest='do_multiSURF', type=str, help='do multiSURF analysis', default="True")
    parser.add_argument('--instance-subset', dest='instance_subset', type=int,help='sample subset size to use with multiSURF', default=2000)
    parser.add_argument('--cv', dest='cv', type=int)
    parser.add_argument('--dataset-name', dest='dataset_name', type=str)

    options = parser.parse_args(argv[2:])
    output_path = options.output_path
    experiment_name = options.experiment_name
    do_mutual_info = options.do_mutual_info
    do_multiSURF = options.do_multiSURF
    instance_subset = options.instance_subset
    dataset_name = options.dataset_name
    cv = options.cv

    metadata = pd.read_csv(output_path + '/' + experiment_name + '/' + 'metadata.csv').values
    random_state = int(metadata[2, 1])
    class_label = metadata[0, 1]
    instance_label = metadata[1, 1]

    full_path = output_path+"/"+experiment_name+"/"+dataset_name
    cv_filename = glob.glob(full_path + "/CVDatasets/*"+str(cv)+"_Train.csv")[0]
    FeatureProcessingMain.submitClusterMIJob(cv_filename,random_state,output_path + '/' + experiment_name,class_label,instance_label)

'''Sample Run Command (This module is designed to AFTER phase 2 has been run and assumes phase 1 ran correctly. Thus, it may depend on metadata.csv):
python CheckJobs.py 2b --output-path /Users/robert/Desktop/outputs --experiment-name test1 --cv 0 --dataset-name Multiplexer6
'''
def runPhase2bJob(argv): #MultiSURF
    # Parse arguments
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--output-path', dest='output_path', type=str, help='path to output directory')
    parser.add_argument('--experiment-name', dest='experiment_name', type=str, help='name of experiment (no spaces)')
    parser.add_argument('--instance-subset', dest='instance_subset', type=int,help='sample subset size to use with multiSURF', default=2000)
    parser.add_argument('--cv', dest='cv', type=int)
    parser.add_argument('--dataset-name', dest='dataset_name', type=str)

    options = parser.parse_args(argv[2:])
    output_path = options.output_path
    experiment_name = options.experiment_name
    instance_subset = options.instance_subset
    dataset_name = options.dataset_name
    cv = options.cv

    metadata = pd.read_csv(output_path + '/' + experiment_name + '/' + 'metadata.csv').values
    random_state = int(metadata[2, 1])
    class_label = metadata[0, 1]
    instance_label = metadata[1, 1]

    full_path = output_path + "/" + experiment_name + "/" + dataset_name
    cv_filename = glob.glob(full_path + "/CVDatasets/*" + str(cv) + "_Train.csv")[0]
    FeatureProcessingMain.submitClusterMSJob(cv_filename, instance_subset,random_state, output_path + '/' + experiment_name,class_label, instance_label)

'''Sample Run Command (This module is designed to AFTER phase 3 has been run and assumes phase 2 ran correctly. Thus, it may depend on metadata.csv):
python CheckJobs.py 3 --output-path /Users/robert/Desktop/outputs --experiment-name test1 --dataset-name Multiplexer6
'''
def runPhase3Job(argv):
    # Parse arguments
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--output-path', dest='output_path', type=str, help='path to output directory')
    parser.add_argument('--experiment-name', dest='experiment_name', type=str, help='name of experiment (no spaces)')
    parser.add_argument('--max-features-to-keep', dest='max_features_to_keep', type=int,help='max features to keep. None if no max', default=2000)
    parser.add_argument('--filter-poor-features', dest='filter_poor_features', type=str, default='True')
    parser.add_argument('--top-results', dest='top_results', type=int, help='# top features to illustrate in figures',default=20)
    parser.add_argument('--graph-scores', dest='graph_scores', type=str, default='True')
    parser.add_argument('--dataset-name', dest='dataset_name', type=str)

    options = parser.parse_args(argv[2:])
    output_path = options.output_path
    experiment_name = options.experiment_name
    max_features_to_keep = options.max_features_to_keep
    filter_poor_features = options.filter_poor_features
    top_results = options.top_results
    graph_scores = options.graph_scores
    dataset_name = options.dataset_name

    metadata = pd.read_csv(output_path + '/' + experiment_name + '/' + 'metadata.csv').values

    class_label = metadata[0, 1]
    instance_label = metadata[1, 1]
    do_mutual_info = metadata[3, 1]
    do_multiSURF = metadata[4, 1]

    full_path = output_path + "/" + experiment_name + "/" + dataset_name
    FeatureSelectionMain.submitClusterJob(full_path,do_mutual_info,do_multiSURF,max_features_to_keep,filter_poor_features,top_results,
                                          graph_scores,class_label,instance_label,output_path + "/" + experiment_name)

'''Sample Run Command (This module is designed to AFTER phase 4 has been run and assumes phase 3 ran correctly. Thus, it may depend on metadata.csv):
python CheckJobs.py 4 --output-path /Users/robert/Desktop/outputs --experiment-name test1 --dataset-name Multiplexer6 --cv 0 --algorithm logistic_regression
'''
def runPhase4Job(argv):
    # Parse arguments
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--output-path', dest='output_path', type=str, help='path to output directory')
    parser.add_argument('--experiment-name', dest='experiment_name', type=str, help='name of experiment (no spaces)')
    parser.add_argument('--n-trials', dest='n_trials', type=int,help='# of bayesian hyperparameter optimization trials', default=100)
    parser.add_argument('--timeout', dest='timeout', type=int, help='seconds until hp sweep stops', default=300)
    parser.add_argument('--lcs-timeout', dest='lcs_timeout', type=int,help='seconds until hp sweep stops for LCS algorithms', default=300)
    parser.add_argument('--plot-hyperparam-sweep', dest='plot_hyperparam_sweep', type=str, default='True')
    parser.add_argument('--cv', dest='cv', type=int)
    parser.add_argument('--dataset-name', dest='dataset_name', type=str)
    parser.add_argument('--algorithm', dest='algorithm', type=str)

    options = parser.parse_args(argv[2:])
    output_path = options.output_path
    experiment_name = options.experiment_name
    n_trials = options.n_trials
    timeout = options.timeout
    lcs_timeout = options.lcs_timeout
    plot_hyperparam_sweep = options.plot_hyperparam_sweep
    dataset_name = options.dataset_name
    cv = options.cv
    algorithm = options.algorithm

    metadata = pd.read_csv(output_path + '/' + experiment_name + '/' + 'metadata.csv').values

    class_label = metadata[0, 1]
    instance_label = metadata[1, 1]
    random_state = int(metadata[2, 1])

    if not algorithm in ['logistic_regression','decision_tree','random_forest','naive_bayes','XGB','ANN','SVM','eLCS','XCS','ExSTraCS']:
        raise Exception('Invalid algorithm')
    full_path = output_path + "/" + experiment_name + "/" + dataset_name
    train_file_path = full_path + '/CVDatasets/' + dataset_name + "_CV_" + str(cv) + "_Train.csv"
    test_file_path = full_path + '/CVDatasets/' + dataset_name + "_CV_" + str(cv) + "_Test.csv"

    ModelMain.submitClusterJob(algorithm,train_file_path,test_file_path,full_path,n_trials,timeout,lcs_timeout,plot_hyperparam_sweep,instance_label,class_label,random_state,output_path+'/'+experiment_name,cv)

'''Sample Run Command (This module is designed to AFTER phase 5 has been run and assumes phase 4 ran correctly. Thus, it may depend on metadata.csv):
python CheckJobs.py 5 --output-path /Users/robert/Desktop/outputs --experiment-name test1 --dataset-name Multiplexer6
'''
def runPhase5Job(argv):
    # Parse arguments
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--output-path', dest='output_path', type=str, help='path to output directory')
    parser.add_argument('--experiment-name', dest='experiment_name', type=str, help='name of experiment (no spaces)')
    parser.add_argument('--plot-ROC', dest='plot_ROC', type=str, default='True')
    parser.add_argument('--plot-PRC', dest='plot_PRC', type=str, default='True')
    parser.add_argument('--plot-FI', dest='plot_FI', type=str, default='True')
    parser.add_argument('--dataset-name', dest='dataset_name', type=str)

    options = parser.parse_args(argv[2:])
    output_path = options.output_path
    experiment_name = options.experiment_name
    plot_ROC = options.plot_ROC
    plot_PRC = options.plot_PRC
    plot_FI = options.plot_FI
    dataset_name = options.dataset_name

    metadata = pd.read_csv(output_path + '/' + experiment_name + '/' + 'metadata.csv').values

    class_label = metadata[0, 1]
    instance_label = metadata[1, 1]

    do_LR = metadata[5, 1]
    do_DT = metadata[6, 1]
    do_RF = metadata[7, 1]
    do_NB = metadata[8, 1]
    do_XGB = metadata[9, 1]
    do_LGB = metadata[10, 1]
    do_SVM = metadata[11, 1]
    do_ANN = metadata[12, 1]
    do_ExSTraCS = metadata[13, 1]
    do_eLCS = metadata[14, 1]
    do_XCS = metadata[15, 1]

    encodedAlgos = ''
    encodedAlgos = StatsMain.encode(do_LR, encodedAlgos)
    encodedAlgos = StatsMain.encode(do_DT, encodedAlgos)
    encodedAlgos = StatsMain.encode(do_RF, encodedAlgos)
    encodedAlgos = StatsMain.encode(do_NB, encodedAlgos)
    encodedAlgos = StatsMain.encode(do_XGB, encodedAlgos)
    encodedAlgos = StatsMain.encode(do_LGB, encodedAlgos)
    encodedAlgos = StatsMain.encode(do_ANN, encodedAlgos)
    encodedAlgos = StatsMain.encode(do_SVM, encodedAlgos)
    encodedAlgos = StatsMain.encode(do_ExSTraCS, encodedAlgos)
    encodedAlgos = StatsMain.encode(do_eLCS, encodedAlgos)
    encodedAlgos = StatsMain.encode(do_XCS, encodedAlgos)

    full_path = output_path + "/" + experiment_name + "/" + dataset_name

    StatsMain.submitClusterJob(full_path,encodedAlgos,plot_ROC,plot_PRC,plot_FI,class_label,instance_label,output_path+'/'+experiment_name)

'''Sample Run Command:
python CheckJobs.py clean --output-path /Users/robert/Desktop/outputs --experiment-name test1
'''
def runCleanJob(argv):
    # Parse arguments
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--output-path', dest='output_path', type=str, help='path to output directory')
    parser.add_argument('--experiment-name', dest='experiment_name', type=str, help='name of experiment (no spaces)')
    options = parser.parse_args(argv[2:])
    output_path = options.output_path
    experiment_name = options.experiment_name

    datasets = os.listdir(output_path + "/" + experiment_name)
    datasets.remove('logs')
    datasets.remove('jobs')
    datasets.remove('jobsCompleted')
    if 'metadata.csv' in datasets:
        datasets.remove('metadata.csv')
    if 'DatasetComparisons' in datasets:
        datasets.remove('DatasetComparisons')

    for dataset in datasets:
        full_path = full_path = output_path + "/" + experiment_name + "/" + dataset
        if os.path.exists(full_path + "/MutualInformation/pickledForPhase3"):
            shutil.rmtree(full_path + "/MutualInformation/pickledForPhase3")
        if os.path.exists(full_path + "/MultiSURF/pickledForPhase3"):
            shutil.rmtree(full_path + "/MultiSURF/pickledForPhase3")
        shutil.rmtree(full_path+'/runtime')
        os.remove(full_path + '/preprocessing/OriginalHeaders.csv')
        for file in glob.glob(full_path+'/training/*_metrics'):
            os.remove(file)
    os.remove(output_path + '/' + experiment_name + '/metadata.csv')
    shutil.rmtree(output_path + "/" + experiment_name + '/jobsCompleted')

if __name__ == '__main__':
    if sys.argv[1] == '0':
        sys.exit(printIncompleteJobs(sys.argv))
    if sys.argv[1] == '1':
        sys.exit(runPhase1Job(sys.argv))
    if sys.argv[1] == '2a':
        sys.exit(runPhase2aJob(sys.argv))
    if sys.argv[1] == '2b':
        sys.exit(runPhase2bJob(sys.argv))
    if sys.argv[1] == '3':
        sys.exit(runPhase3Job(sys.argv))
    if sys.argv[1] == '4':
        sys.exit(runPhase4Job(sys.argv))
    if sys.argv[1] == '5':
        sys.exit(runPhase5Job(sys.argv))
    if sys.argv[1] == 'clean':
        sys.exit(runCleanJob(sys.argv))