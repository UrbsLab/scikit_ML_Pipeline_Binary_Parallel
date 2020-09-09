
import argparse
import os
import sys
import pandas as pd
import glob
import ModelJob
import time
import csv

'''Sample Run Command:
python ModelMain.py --output-path /Users/robert/Desktop/outputs --experiment-name test1
'''

def main(argv):
    #Parse arguments
    default_arg = 'True'
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--output-path', dest='output_path', type=str, help='path to output directory')
    parser.add_argument('--experiment-name', dest='experiment_name', type=str, help='name of experiment (no spaces)')
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
    parser.add_argument('--do-KN', dest='do_KN', type=str, default=default_arg)
    parser.add_argument('--do-GB', dest='do_GB', type=str, default=default_arg)
    parser.add_argument('--n-trials', dest='n_trials', type=int,help='# of bayesian hyperparameter optimization trials', default=100)
    parser.add_argument('--timeout', dest='timeout', type=int,help='seconds until hp sweep stops', default=300)
    parser.add_argument('--lcs-timeout', dest='lcs_timeout', type=int, help='seconds until hp sweep stops for LCS algorithms', default=300)
    parser.add_argument('--plot-hyperparam-sweep', dest='plot_hyperparam_sweep', type=str, default='True')

    options = parser.parse_args(argv[1:])
    output_path = options.output_path
    experiment_name = options.experiment_name

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
    if options.do_GB == 'True':
        algorithms.append('gradient_boosting')
    if options.do_KN == 'True':
        algorithms.append('k_neighbors')

    n_trials = options.n_trials
    timeout = options.timeout
    lcs_timeout = options.lcs_timeout
    plot_hyperparam_sweep = options.plot_hyperparam_sweep

    # Argument checks
    if not os.path.exists(output_path):
        raise Exception("Output path must exist (from phase 1) before phase 4 can begin")

    if not os.path.exists(output_path + '/' + experiment_name):
        raise Exception("Experiment must exist (from phase 1) before phase 4 can begin")

    metadata = pd.read_csv(output_path + '/' + experiment_name + '/' + 'metadata.csv').values

    class_label = metadata[0, 1]
    instance_label = metadata[1, 1]
    random_state = int(metadata[2,1])

    dataset_paths = os.listdir(output_path + "/" + experiment_name)
    dataset_paths.remove('logs')
    dataset_paths.remove('jobs')
    dataset_paths.remove('jobsCompleted')
    dataset_paths.remove('metadata.csv')
    for dataset_directory_path in dataset_paths:
        full_path = output_path + "/" + experiment_name + "/" + dataset_directory_path
        if not os.path.exists(full_path+'/training'):
            os.mkdir(full_path+'/training')
        if not os.path.exists(full_path+'/training/pickledModels'):
            os.mkdir(full_path+'/training/pickledModels')
        cvPartitions = int(len(glob.glob(full_path + '/CVDatasets/*.csv')) / 2)
        for cvCount in range(cvPartitions):
            train_file_path = full_path+'/CVDatasets/'+dataset_directory_path+"_CV_"+str(cvCount)+"_Train.csv"
            test_file_path = full_path + '/CVDatasets/' + dataset_directory_path + "_CV_" + str(cvCount) + "_Test.csv"
            for algorithm in algorithms:
                submitLocalJob(algorithm,train_file_path,test_file_path,full_path,n_trials,timeout,lcs_timeout,plot_hyperparam_sweep,instance_label,class_label,random_state,cvCount)
                #submitClusterJob(algorithm,train_file_path,test_file_path,full_path,n_trials,timeout,lcs_timeout,plot_hyperparam_sweep,instance_label,class_label,random_state,output_path+'/'+experiment_name,cvCount)

    # Update metadata
    if metadata.shape[0] == 5:  # Only update if metadata below hasn't been added before (i.e. in a previous phase 4 run)
        with open(output_path + '/' + experiment_name + '/' + 'metadata.csv', mode='a') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["LR", options.do_LR])
            writer.writerow(["DT", options.do_DT])
            writer.writerow(["RF", options.do_RF])
            writer.writerow(["NB", options.do_NB])
            writer.writerow(["XGB", options.do_XGB])
            writer.writerow(["LGB", options.do_LGB])
            writer.writerow(["SVM", options.do_SVM])
            writer.writerow(["ANN", options.do_ANN])
            writer.writerow(["ExSTraCS", options.do_ExSTraCS])
            writer.writerow(["eLCS", options.do_eLCS])
            writer.writerow(["XCS", options.do_XCS])
            writer.writerow(["GB", options.do_GB])
            writer.writerow(["KN", options.do_KN])
        file.close()

def submitLocalJob(algorithm,train_file_path,test_file_path,full_path,n_trials,timeout,lcs_timeout,plot_hyperparam_sweep,instance_label,class_label,random_state,cvCount):
    ModelJob.job(algorithm,train_file_path,test_file_path,full_path,n_trials,timeout,lcs_timeout,plot_hyperparam_sweep,instance_label,class_label,random_state,cvCount)

def submitClusterJob(algorithm,train_file_path,test_file_path,full_path,n_trials,timeout,lcs_timeout,plot_hyperparam_sweep,instance_label,class_label,random_state,experiment_path,cvCount):
    job_ref = str(time.time())
    job_name = experiment_path + '/jobs/' + job_ref + '_run.sh'
    sh_file = open(job_name, 'w')
    sh_file.write('#!/bin/bash\n')
    sh_file.write('#BSUB -J ' + job_ref + '\n')
    sh_file.write('#BSUB -o ' + experiment_path + '/logs/' + job_ref + '.o\n')
    sh_file.write('#BSUB -e ' + experiment_path + '/logs/' + job_ref + '.e\n')

    this_file_path = os.path.dirname(os.path.realpath(__file__))
    sh_file.write('python '+this_file_path+'/ModelMain.py '+algorithm+" "+train_file_path+" "+test_file_path+" "+full_path+" "+
                  str(n_trials)+" "+str(timeout)+" "+str(lcs_timeout)+" "+plot_hyperparam_sweep+" "+instance_label+" "+class_label+" "+
                  str(random_state)+str(cvCount)+'\n')
    sh_file.close()
    os.system('bsub < ' + job_name)
    pass

if __name__ == '__main__':
    sys.exit(main(sys.argv))