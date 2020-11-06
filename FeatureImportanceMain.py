
import argparse
import os
import sys
import glob
import FeatureImportanceJob
import time
import pandas as pd
import csv

'''Phase 3 of Machine Learning Analysis Pipeline:
Sample Run Command:
python FeatureImportanceMain.py --output-path /Users/robert/Desktop/outputs --experiment-name test1
'''

def main(argv):
    #Parse arguments
    parser = argparse.ArgumentParser(description='')
    #No defaults
    parser.add_argument('--output-path', dest='output_path', type=str, help='path to output directory')
    parser.add_argument('--experiment-name', dest='experiment_name', type=str, help='name of experiment (no spaces)')
    #Defaults available
    parser.add_argument('--do-mutual-info', dest='do_mutual_info', type=str, help='do mutual information analysis',default="True")
    parser.add_argument('--do-multisurf', dest='do_multisurf', type=str, help='do multiSURF analysis',default="True")
    parser.add_argument('--n-jobs', dest='n_jobs', type=int, help='number of course dedicated to running algorithm; setting to -1 will use all available cores', default=1)
    parser.add_argument('--instance-subset', dest='instance_subset', type=int, help='sample subset size to use with multiSURF',default=2000)
    parser.add_argument('--run-parallel',dest='run_parallel',type=str,help='path to directory containing datasets',default="True")
    parser.add_argument('--res-mem', dest='reserved_memory', type=int, help='reserved memory for the job (in Gigabytes)',default=4)
    parser.add_argument('--max-mem', dest='maximum_memory', type=int, help='maximum memory before the job is automatically terminated',default=15)
    parser.add_argument('-c','--do-check',dest='do_check', help='Boolean: Specify whether to check for existence of all output files.', action='store_true')

    options = parser.parse_args(argv[1:])

    output_path = options.output_path
    experiment_name = options.experiment_name
    do_mutual_info = options.do_mutual_info
    do_multisurf = options.do_multisurf
    n_jobs = options.n_jobs
    instance_subset = options.instance_subset
    run_parallel = options.run_parallel
    reserved_memory = options.reserved_memory
    maximum_memory = options.maximum_memory
    do_check = options.do_check

    # Argument checks
    if not os.path.exists(output_path):
        raise Exception("Output path must exist (from phase 1) before phase 3 can begin")

    if not os.path.exists(output_path + '/' + experiment_name):
        raise Exception("Experiment must exist (from phase 1) before phase 3 can begin")

    metadata = pd.read_csv(output_path+'/'+experiment_name + '/' + 'metadata.csv').values

    class_label = metadata[0, 1]
    instance_label = metadata[1,1]
    random_state = int(metadata[2, 1])
    categorical_cutoff = int(metadata[3,1])
    cv_partitions = int(metadata[4,1])

    if not do_check:
        #Iterate through datasets, ignoring common folders
        dataset_paths = os.listdir(output_path+"/"+experiment_name)
        dataset_paths.remove('logs')
        dataset_paths.remove('jobs')
        dataset_paths.remove('jobsCompleted')
        dataset_paths.remove('metadata.csv')
        for dataset_directory_path in dataset_paths:
            full_path = output_path+"/"+experiment_name+"/"+dataset_directory_path
            experiment_path = output_path+'/'+experiment_name

            if do_mutual_info == 'True':
                if not os.path.exists(full_path+"/mutualinformation"):
                    os.mkdir(full_path+"/mutualinformation")
                for cv_train_path in glob.glob(full_path+"/CVDatasets/*_CV_*Train.csv"):
                    command_text = '/FeatureImportanceJob.py ' + cv_train_path+" "+experiment_path+" "+str(random_state)+" "+class_label+" "+instance_label+" "+str(categorical_cutoff)+" " +str(instance_subset)+" mi"
                    if run_parallel:
                        submitClusterJob(command_text, experiment_path,reserved_memory,maximum_memory)
                    else:
                        submitLocalJob(cv_train_path,experiment_path,random_state,class_label,instance_label,categorical_cutoff,instance_subset,'mi')

            if do_multisurf == 'True':
                if not os.path.exists(full_path+"/multisurf"):
                    os.mkdir(full_path+"/multisurf")
                for cv_train_path in glob.glob(full_path+"/CVDatasets/*_CV_*Train.csv"):
                    command_text = '/FeatureImportanceJob.py ' + cv_train_path+" "+experiment_path+" "+str(random_state)+" "+class_label+" "+instance_label+" "+str(categorical_cutoff)+" " +str(instance_subset)+" ms"
                    if run_parallel:
                        submitClusterJob(command_text, experiment_path,reserved_memory,maximum_memory)
                    else:
                        submitLocalJob(cv_train_path,experiment_path,random_state,class_label,instance_label,categorical_cutoff,instance_subset,'mi')

        #Update metadata
        if metadata.shape[0] == 7: #Only update if metadata below hasn't been added before (i.e. in a previous phase 2 run)
            with open(output_path + '/' + experiment_name + '/' + 'metadata.csv',mode='a') as file:
                writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(["Do Mutual Info",do_mutual_info])
                writer.writerow(["Do MultiSURF", do_multisurf])
            file.close()

    else: #run job checks
        datasets = os.listdir(output_path + "/" + experiment_name)
        datasets.remove('logs')
        datasets.remove('jobs')
        datasets.remove('jobsCompleted')
        if 'metadata.csv' in datasets:
            datasets.remove('metadata.csv')
        if 'DatasetComparisons' in datasets:
            datasets.remove('DatasetComparisons')

        phase3Jobs = []
        for dataset in datasets:
            for cv in range(cv_partitions):
                if do_multisurf:
                    phase3Jobs.append('job_multisurf_' + dataset + '_' + str(cv) + '.txt')
                if do_mutual_info:
                    phase3Jobs.append('job_mutualinformation_' + dataset + '_' + str(cv) + '.txt')

        for filename in glob.glob(output_path + "/" + experiment_name + '/jobsCompleted/job_mu*'):
            ref = filename.split('/')[-1]
            phase3Jobs.remove(ref)
        for job in phase3Jobs:
            print(job)
        if len(phase3Jobs) == 0:
            print("All Phase 3 Jobs Completed")
        else:
            print("Above Phase 3 Jobs Not Completed")
        print()

def submitLocalJob(cv_train_path,experiment_path,random_state,class_label,instance_label,categorical_cutoff,instance_subset,algorithm):
    FeatureProcessingJob.job(cv_train_path,experiment_path,random_state,class_label,instance_label,categorical_cutoff,instance_subset,algorithm)

def submitClusterJob(command_text,experiment_path,reserved_memory,maximum_memory):
    job_ref = str(time.time())
    job_name = experiment_path+'/jobs/P3_'+job_ref+'_run.sh'
    sh_file = open(job_name,'w')
    sh_file.write('#!/bin/bash\n')
    sh_file.write('#BSUB -q i2c2_normal'+'\n')
    sh_file.write('#BSUB -J '+job_ref+'\n')
    sh_file.write('#BSUB -R "rusage[mem='+str(reserved_memory)+'G]"'+'\n')
    sh_file.write('#BSUB -M '+str(maximum_memory)+'GB'+'\n')
    sh_file.write('#BSUB -o ' + experiment_path+'/logs/P3_'+job_ref+'.o\n')
    sh_file.write('#BSUB -e ' + experiment_path+'/logs/P3_'+job_ref+'.e\n')

    this_file_path = os.path.dirname(os.path.realpath(__file__))
    sh_file.write('python ' + this_file_path + command_text+'\n')
    sh_file.close()
    os.system('bsub < ' + job_name)
    pass

if __name__ == '__main__':
    sys.exit(main(sys.argv))
