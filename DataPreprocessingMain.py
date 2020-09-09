
import sys
import os
import argparse
import glob
import pandas as pd
import DataPreprocessingJob
import time
import csv

'''Phase 2 of Machine Learning Analysis Pipeline:
Sample Run Command:
python DataPreprocessingMain.py --output-path /Users/robert/Desktop/outputs --experiment-name test1
'''

def main(argv):
    #Parse arguments
    parser = argparse.ArgumentParser(description="")
    #No defaults
    parser.add_argument('--output-path',dest='output_path',type=str,help='path to output directory')
    parser.add_argument('--experiment-name', dest='experiment_name',type=str, help='name of experiment output folder (no spaces)')
    #Defaults available
    parser.add_argument('--run-parallel',dest='run_parallel',type=str,help='path to directory containing datasets',default="True")
    parser.add_argument('--scale-data',dest='scale_data',type=str,help='perform data scaling?',default="True")
    parser.add_argument('--impute-data', dest='impute_data',type=str,help='perform missing value data imputation? (required for most ML algorithms if missing data is present)',default="True")
    parser.add_argument('--overwrite-cv', dest='overwrite_cv',type=str,help='overwrites earlier cv datasets with new scaled/imputed ones',default="True")

    options = parser.parse_args(argv[1:])
    output_path = options.output_path
    experiment_name = options.experiment_name
    run_parallel = options.run_parallel
    scale_data = options.scale_data
    impute_data = options.impute_data
    overwrite_cv = options.overwrite_cv

    # Argument checks
    if not os.path.exists(output_path):
        raise Exception("Output path must exist (from phase 1) before phase 2 can begin")

    if not os.path.exists(output_path + '/' + experiment_name):
        raise Exception("Experiment must exist (from phase 1) before phase 2 can begin")

    metadata = pd.read_csv(output_path+'/'+experiment_name + '/' + 'metadata.csv').values
    class_label = metadata[0, 1]
    instance_label = metadata[1,1]
    random_state = int(metadata[2, 1])
    categorical_cutoff = int(metadata[3,1])

    #Iterate through datasets, ignoring common folders
    dataset_paths = os.listdir(output_path+"/"+experiment_name)
    dataset_paths.remove('logs')
    dataset_paths.remove('jobs')
    dataset_paths.remove('jobsCompleted')
    dataset_paths.remove('metadata.csv')
    for dataset_directory_path in dataset_paths:
        full_path = output_path+"/"+experiment_name+"/"+dataset_directory_path
        cv_count = 0
        for cv_train_path in glob.glob(full_path+"/CVDatasets/*Train.csv"):
            cv_test_path = cv_train_path.replace("Train.csv","Test.csv")
            if run_parallel:
                submitClusterJob(cv_train_path,cv_test_path,output_path+'/'+experiment_name,scale_data,impute_data,overwrite_cv,categorical_cutoff,class_label,instance_label,random_state)
            else:
                submitLocalJob(cv_train_name,cv_test_name,output_path+'/'+experiment_name,scale_data,impute_data,overwrite_cv,categorical_cutoff,class_label,instance_label,random_state)

    #Update metadata
    if metadata.shape[0] == 5: #Only update if metadata below hasn't been added before (i.e. in a previous phase 2 run)
        with open(output_path + '/' + experiment_name + '/' + 'metadata.csv',mode='a') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["Do Data Scaling",scale_data])
            writer.writerow(["Do Data Imputation",impute_data])
        file.close()


def submitLocalJob(cv_train_path,cv_test_path,experiment_path,scale_data,impute_data,overwrite_cv,categorical_cutoff,class_label,instance_label,random_state):
    DataPreprocessingJob.job(cv_train_path,cv_test_path,experiment_path,scale_data,impute_data,overwrite_cv,categorical_cutoff,class_label,instance_label,random_state)

def submitClusterJob(cv_train_path,cv_test_path,experiment_path,scale_data,impute_data,overwrite_cv,categorical_cutoff,class_label,instance_label,random_state):
    job_ref = str(time.time())
    job_name = experiment_path+'/jobs/P2_'+job_ref+'_run.sh'
    sh_file = open(job_name,'w')
    sh_file.write('#!/bin/bash\n')
    sh_file.write('#BSUB -q doi_normal'+'\n')
    sh_file.write('#BSUB -J '+job_ref+'\n')
    sh_file.write('#BSUB -R "rusage[mem=4G]"'+'\n')
    sh_file.write('#BSUB -M 15GB'+'\n')
    sh_file.write('#BSUB -o ' + experiment_path+'/logs/'+job_ref+'.o\n')
    sh_file.write('#BSUB -e ' + experiment_path+'/logs/'+job_ref+'.e\n')

    this_file_path = os.path.dirname(os.path.realpath(__file__))
    sh_file.write('python '+this_file_path+'/DataPreprocessingJob.py '+cv_train_path+" "+cv_test_path+" "+experiment_path+" "+scale_data+
                  " "+impute_data+" "+overwrite_cv+" "+str(categorical_cutoff)+" "+class_label+" "+instance_label+" "+str(random_state)+'\n')
    sh_file.close()
    os.system('bsub < '+job_name)
    pass

if __name__ == '__main__':
    sys.exit(main(sys.argv))
