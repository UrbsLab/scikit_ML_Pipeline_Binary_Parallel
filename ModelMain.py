
import argparse
import os
import sys
import pandas as pd
import glob
import ModelJob
import time
import csv

'''Phase 5 of Machine Learning Analysis Pipeline:
Sample Run Command:
python ModelMain.py --output-path /Users/robert/Desktop/outputs --experiment-name test1
'''

def main(argv):
    #Parse arguments
    default_arg = 'True'
    parser = argparse.ArgumentParser(description='')
    #No defaults
    parser.add_argument('--output-path', dest='output_path', type=str, help='path to output directory')
    parser.add_argument('--experiment-name', dest='experiment_name', type=str, help='name of experiment (no spaces)')
    #ML modeling algorithms: Defaults available
    parser.add_argument('--do-LR', dest='do_LR', type=str, help='run logistic regression modeling',default=default_arg)
    parser.add_argument('--do-DT', dest='do_DT', type=str, help='run decision tree modeling',default=default_arg)
    parser.add_argument('--do-RF', dest='do_RF', type=str, help='run random forest modeling',default=default_arg)
    parser.add_argument('--do-NB', dest='do_NB', type=str, help='run naive bayes modeling',default=default_arg)
    parser.add_argument('--do-XGB', dest='do_XGB', type=str, help='run XGBoost modeling',default=default_arg)
    parser.add_argument('--do-LGB', dest='do_LGB', type=str, help='run LGBoost modeling',default=default_arg)
    parser.add_argument('--do-SVM', dest='do_SVM', type=str, help='run support vector machine modeling',default=default_arg)
    parser.add_argument('--do-ANN', dest='do_ANN', type=str, help='run artificial neural network modeling',default=default_arg)
    parser.add_argument('--do-ExSTraCS', dest='do_ExSTraCS', type=str, help='run ExSTraCS modeling (a learning classifier system designed for biomedical data mining)',default=default_arg)
    parser.add_argument('--do-eLCS', dest='do_eLCS', type=str, help='run eLCS modeling (a basic supervised-learning learning classifier system)',default=default_arg)
    parser.add_argument('--do-XCS', dest='do_XCS', type=str, help='run XCS modeling (a supervised-learning-only implementation of the best studied learning classifier system)',default=default_arg)
    parser.add_argument('--do-KN', dest='do_KN', type=str, help='run k-neighbors classifier modeling',default=default_arg)
    parser.add_argument('--do-GB', dest='do_GB', type=str, help='run gradient boosting modeling',default=default_arg)
    #Defaults available
    parser.add_argument('--n-trials', dest='n_trials', type=int,help='# of bayesian hyperparameter optimization trials using optuna', default=100)
    parser.add_argument('--timeout', dest='timeout', type=int,help='seconds until hyperparameter sweep stops running new trials (Note: it may run longer to finish last trial started)', default=300)
    parser.add_argument('--lcs-timeout', dest='lcs_timeout', type=int, help='seconds until hyperparameter sweep stops for LCS algorithms', default=1200)
    parser.add_argument('--export-hyper-sweep', dest='export_hyper_sweep_plots', type=str, default='True')
    parser.add_argument('--run-parallel',dest='run_parallel',type=str,help='path to directory containing datasets',default="True")
    parser.add_argument('--res-mem', dest='reserved_memory', type=int, help='reserved memory for the job (in Gigabytes)',default=4)
    parser.add_argument('--max-mem', dest='maximum_memory', type=int, help='maximum memory before the job is automatically terminated',default=15)
    parser.add_argument('-c','--do-check',dest='do_check', help='Boolean: Specify whether to check for existence of all output files.', action='store_true')

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
    export_hyper_sweep_plots = options.export_hyper_sweep_plots
    run_parallel = options.run_parallel
    reserved_memory = options.reserved_memory
    maximum_memory = options.maximum_memory
    do_check = options.do_check

    # Argument checks
    if not os.path.exists(output_path):
        raise Exception("Output path must exist (from phase 1) before phase 5 can begin")

    if not os.path.exists(output_path + '/' + experiment_name):
        raise Exception("Experiment must exist (from phase 1) before phase 5 can begin")

    metadata = pd.read_csv(output_path + '/' + experiment_name + '/' + 'metadata.csv').values

    class_label = metadata[0, 1]
    instance_label = metadata[1, 1]
    random_state = int(metadata[2,1])
    cv_partitions = int(metadata[4,1])
    filter_poor_features = metadata[10,1]

    if not do_check:
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

            for cvCount in range(cv_partitions):
                train_file_path = full_path+'/CVDatasets/'+dataset_directory_path+"_CV_"+str(cvCount)+"_Train.csv"
                test_file_path = full_path + '/CVDatasets/' + dataset_directory_path + "_CV_" + str(cvCount) + "_Test.csv"
                for algorithm in algorithms:
                    if run_parallel:
                        submitClusterJob(algorithm,train_file_path,test_file_path,full_path,n_trials,timeout,lcs_timeout,export_hyper_sweep_plots,instance_label,class_label,random_state,output_path+'/'+experiment_name,cvCount,filter_poor_features,reserved_memory,maximum_memory)
                    else:
                        submitLocalJob(algorithm,train_file_path,test_file_path,full_path,n_trials,timeout,lcs_timeout,export_hyper_sweep_plots,instance_label,class_label,random_state,cvCount,filter_poor_features)

        # Update metadata
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

    else: #run job checks
        abbrev = {'logistic_regression':'LR','decision_tree':'DT','random_forest':'RF','naive_bayes':'NB','XGB':'XGB','LGB':'LGB','ANN':'ANN','SVM':'SVM','ExSTraCS':'ExSTraCS','eLCS':'eLCS','XCS':'XCS','gradient_boosting':'GB','k_neighbors':'KN'}

        datasets = os.listdir(output_path + "/" + experiment_name)
        datasets.remove('logs')
        datasets.remove('jobs')
        datasets.remove('jobsCompleted')
        if 'metadata.csv' in datasets:
            datasets.remove('metadata.csv')
        if 'DatasetComparisons' in datasets:
            datasets.remove('DatasetComparisons')

        phase5Jobs = []
        for dataset in datasets:
            for cv in range(cv_partitions):
                for algorithm in algorithms:
                    phase5Jobs.append('job_model_' + dataset + '_' + str(cv) +'_' +abbrev[algorithm]+'.txt')

        for filename in glob.glob(output_path + "/" + experiment_name + '/jobsCompleted/job_model*'):
            ref = filename.split('/')[-1]
            phase5Jobs.remove(ref)
        for job in phase5Jobs:
            print(job)
        if len(phase5Jobs) == 0:
            print("All Phase 5 Jobs Completed")
        else:
            print("Above Phase 5 Jobs Not Completed")
        print()

def submitLocalJob(algorithm,train_file_path,test_file_path,full_path,n_trials,timeout,lcs_timeout,export_hyper_sweep_plots,instance_label,class_label,random_state,cvCount,filter_poor_features):
    ModelJob.job(algorithm,train_file_path,test_file_path,full_path,n_trials,timeout,lcs_timeout,export_hyper_sweep_plots,instance_label,class_label,random_state,cvCount,filter_poor_features)

def submitClusterJob(algorithm,train_file_path,test_file_path,full_path,n_trials,timeout,lcs_timeout,export_hyper_sweep_plots,instance_label,class_label,random_state,experiment_path,cvCount,filter_poor_features,reserved_memory,maximum_memory):
    job_ref = str(time.time())
    job_name = experiment_path+'/jobs/P5_'+str(algorithm)+'_'+str(cvCount)+'_'+job_ref+'_run.sh'
    sh_file = open(job_name,'w')
    sh_file.write('#!/bin/bash\n')
    sh_file.write('#BSUB -q doi_normal'+'\n')
    sh_file.write('#BSUB -J '+job_ref+'\n')
    sh_file.write('#BSUB -R "rusage[mem='+str(reserved_memory)+'G]"'+'\n')
    sh_file.write('#BSUB -M '+str(maximum_memory)+'GB'+'\n')
    sh_file.write('#BSUB -o ' + experiment_path+'/logs/P5_'+str(algorithm)+'_'+str(cvCount)+'_'+job_ref+'.o\n')
    sh_file.write('#BSUB -e ' + experiment_path+'/logs/P5_'+str(algorithm)+'_'+str(cvCount)+'_'+job_ref+'.e\n')

    this_file_path = os.path.dirname(os.path.realpath(__file__))
    sh_file.write('python '+this_file_path+'/ModelJob.py '+algorithm+" "+train_file_path+" "+test_file_path+" "+full_path+" "+
                  str(n_trials)+" "+str(timeout)+" "+str(lcs_timeout)+" "+export_hyper_sweep_plots+" "+instance_label+" "+class_label+" "+
                  str(random_state)+" "+str(cvCount)+" "+str(filter_poor_features)+'\n')
    sh_file.close()
    os.system('bsub < ' + job_name)
    pass

if __name__ == '__main__':
    sys.exit(main(sys.argv))
