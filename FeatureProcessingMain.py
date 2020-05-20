
import argparse
import os
import sys
import glob
import FeatureProcessingJob
import time
import pandas as pd
import csv

'''Sample Run Command:
python FeatureProcessingMain.py --output-path /Users/robert/Desktop/outputs --experiment-name test1
'''

def main(argv):
    #Parse arguments
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--output-path', dest='output_path', type=str, help='path to output directory')
    parser.add_argument('--experiment-name', dest='experiment_name', type=str, help='name of experiment (no spaces)')
    parser.add_argument('--do-mutual-info', dest='do_mutual_info', type=str, help='do mutual information analysis',default="True")
    parser.add_argument('--do-multiSURF', dest='do_multiSURF', type=str, help='do multiSURF analysis',default="True")
    parser.add_argument('--instance-subset', dest='instance_subset', type=int, help='sample subset size to use with multiSURF',default=2000)

    options = parser.parse_args(argv[1:])
    output_path = options.output_path
    experiment_name = options.experiment_name
    do_mutual_info = options.do_mutual_info
    do_multiSURF = options.do_multiSURF
    instance_subset = options.instance_subset

    # Argument checks
    if not os.path.exists(output_path):
        raise Exception("Output path must exist (from phase 1) before phase 2 can begin")

    if not os.path.exists(output_path + '/' + experiment_name):
        raise Exception("Experiment must exist (from phase 1) before phase 2 can begin")

    metadata = pd.read_csv(output_path+'/'+experiment_name + '/' + 'metadata.csv').values
    random_state = int(metadata[2, 1])
    class_label = metadata[0, 1]
    instance_label = metadata[1,1]

    #Iterate through datasets
    dataset_paths = os.listdir(output_path+"/"+experiment_name)
    dataset_paths.remove('logs')
    dataset_paths.remove('jobs')
    dataset_paths.remove('jobsCompleted')
    dataset_paths.remove('metadata.csv')
    for dataset_directory_path in dataset_paths:
        full_path = output_path+"/"+experiment_name+"/"+dataset_directory_path
        if do_mutual_info == 'True':
            if not os.path.exists(full_path+"/MutualInformation"):
                os.mkdir(full_path+"/MutualInformation")
            for cv_filename in glob.glob(full_path+"/CVDatasets/*Train.csv"):
                submitLocalMIJob(cv_filename,random_state,output_path+'/'+experiment_name,class_label,instance_label)
                #submitClusterMIJob(cv_filename,random_state,output_path+'/'+experiment_name,class_label,instance_label)
        if do_multiSURF == 'True':
            if not os.path.exists(full_path + "/MultiSURF"):
                os.mkdir(full_path + "/MultiSURF")
            for cv_filename in glob.glob(full_path+"/CVDatasets/*Train.csv"):
                submitLocalMSJob(cv_filename,instance_subset,random_state,output_path+'/'+experiment_name,class_label,instance_label)
                #submitClusterMSJob(cv_filename,instance_subset,random_state,output_path+'/'+experiment_name,class_label,instance_label)

    #Update metadata
    if metadata.shape[0] == 3: #Only update if metadata below hasn't been added before (i.e. in a previous phase 2 run)
        with open(output_path + '/' + experiment_name + '/' + 'metadata.csv',mode='a') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["Do Mutual Info",do_mutual_info])
            writer.writerow(["Do MultiSURF", do_multiSURF])
        file.close()

def submitLocalMIJob(trainfile_path,random_state,experiment_path,class_label,instance_label):
    FeatureProcessingJob.miJob(trainfile_path,random_state,experiment_path,class_label,instance_label)

def submitClusterMIJob(trainfile_path,random_state,experiment_path,class_label,instance_label):
    job_ref = str(time.time())
    job_name = experiment_path + '/jobs/' + job_ref + '_run.sh'
    sh_file = open(job_name, 'w')
    sh_file.write('#!/bin/bash\n')
    sh_file.write('#BSUB -J ' + job_ref + '\n')
    sh_file.write('#BSUB -o ' + experiment_path + '/logs/' + job_ref + '.o\n')
    sh_file.write('#BSUB -e ' + experiment_path + '/logs/' + job_ref + '.e\n')

    this_file_path = os.path.dirname(os.path.realpath(__file__))
    sh_file.write('python ' + this_file_path + '/FeatureProcessingJob.py ' + trainfile_path + " " + str(random_state) + " " + experiment_path + " " + class_label + " " + instance_label+" mi"'\n')
    sh_file.close()
    os.system('bsub < ' + job_name)
    pass

def submitLocalMSJob(trainfile_path,instance_subset,random_state,experiment_path,class_label,instance_label):
    FeatureProcessingJob.msJob(trainfile_path,instance_subset,random_state,experiment_path,class_label,instance_label)

def submitClusterMSJob(trainfile_path,instance_subset,random_state,experiment_path,class_label,instance_label):
    job_ref = str(time.time())
    job_name = experiment_path + '/jobs/' + job_ref + '_run.sh'
    sh_file = open(job_name, 'w')
    sh_file.write('#!/bin/bash\n')
    sh_file.write('#BSUB -J ' + job_ref + '\n')
    sh_file.write('#BSUB -o ' + experiment_path + '/logs/' + job_ref + '.o\n')
    sh_file.write('#BSUB -e ' + experiment_path + '/logs/' + job_ref + '.e\n')

    this_file_path = os.path.dirname(os.path.realpath(__file__))
    sh_file.write('python '+this_file_path+'/FeatureProcessingJob.py '+trainfile_path+" "+str(instance_subset)+" "+str(random_state)+" "+experiment_path+ " " + class_label + " " + instance_label+" ms"'\n')
    sh_file.close()
    os.system('bsub < ' + job_name)
    pass

if __name__ == '__main__':
    sys.exit(main(sys.argv))