
import sys
import os
import argparse
import glob
import DataPreprocessingJob
import time
import csv

'''Sample Run Command:
python DataPreprocessingMain.py --data-path /Users/robert/Desktop/Datasets --output-path /Users/robert/Desktop/outputs --experiment-name test1
'''

def main(argv):
    #Parse arguments
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--data-path',dest='data_path',type=str,help='path to directory containing datasets')
    parser.add_argument('--output-path',dest='output_path',type=str,help='path to output directory')
    parser.add_argument('--experiment-name', dest='experiment_name',type=str, help='name of experiment (no spaces)')
    parser.add_argument('--cv',dest='cv_partitions',type=int,help='number of CV partitions',default=3)
    parser.add_argument('--partition-method',dest='partition_method',type=str,help='S or R or M',default="S")
    parser.add_argument('--scale-data',dest='scale_data',type=str,default="True")
    parser.add_argument('--impute-data', dest='impute_data',type=str,default="True")
    parser.add_argument('--categorical-cutoff', dest='categorical_cutoff', type=int, default=10)
    parser.add_argument('--class-label', dest='class_label', type=str, help='outcome label of all datasets', default="Class")
    parser.add_argument('--instance-label', dest='instance_label', type=str, default="")
    parser.add_argument('--match-label', dest='match_label', type=str, default="")
    parser.add_argument('--export-initial-analysis', dest='export_initial_analysis', type=str, default="True")
    parser.add_argument('--export-feature-correlations', dest='export_feature_correlations', type=str, default="True")
    parser.add_argument('--export-univariate', dest='export_univariate', type=str, default="True")
    parser.add_argument('--random-state', dest='random_state', type=int, default=42)

    options = parser.parse_args(argv[1:])
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

    #Check to make sure data_path exists and experiment name is valid & unique
    if not os.path.exists(data_path):
        raise Exception("Provided data_path does not exist")

    if os.path.exists(output_path+'/'+experiment_name):
        raise Exception("Experiment Name must be unique")

    for char in experiment_name:
        if not char in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890':
            raise Exception('Experiment Name must be alphanumeric')

    #Create output folder if it doesn't already exist
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    #Create Experiment folder, with log and job folders
    os.mkdir(output_path+'/'+experiment_name)
    os.mkdir(output_path+'/'+experiment_name+'/jobs')
    os.mkdir(output_path+'/'+experiment_name+'/logs')
    os.mkdir(output_path+'/'+ experiment_name+'/jobsCompleted')

    #Check that there is at least 1 dataset
    if len(glob.glob(data_path+'/*.csv')) == 0:
        raise Exception("There must be at least one csv dataset in data_path directory")

    #Iterate through datasets
    for datasetFilename in glob.glob(data_path+'/*.csv'):
        submitLocalJob(datasetFilename,output_path+'/'+experiment_name,cv_partitions,partition_method,scale_data,impute_data,categorical_cutoff,export_initial_analysis,export_feature_correlations,export_univariate,class_label,instance_label,match_label,random_state)
        #submitClusterJob(datasetFilename,output_path+'/'+experiment_name,cv_partitions,partition_method,scale_data,impute_data,categorical_cutoff,export_initial_analysis,export_feature_correlations,export_univariate,random_state)

    # Save metadata to file
    with open(output_path+'/'+experiment_name+'/'+'metadata.csv',mode='w') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["DATA LABEL", "VALUE"])
        writer.writerow(["class label",class_label])
        writer.writerow(["instance label", instance_label])
        writer.writerow(["random state",random_state])
    file.close()

def submitLocalJob(dataset_path,experiment_path,cv_partitions,partition_method,scale_data,impute_data,categorical_cutoff,export_initial_analysis,export_feature_correlations,export_univariate,class_label,instance_label,match_label,random_state):
    DataPreprocessingJob.job(dataset_path,experiment_path,cv_partitions,partition_method,scale_data,impute_data,categorical_cutoff,export_initial_analysis,export_feature_correlations,export_univariate,class_label,instance_label,match_label,random_state)

def submitClusterJob(dataset_path,experiment_path,cv_partitions,partition_method,scale_data,impute_data,categorical_cutoff,export_initial_analysis,export_feature_correlations,export_univariate,class_label,instance_label,match_label,random_state):
    job_ref = str(time.time())
    job_name = experiment_path+'/jobs/'+job_ref+'_run.sh'
    sh_file = open(job_name,'w')
    sh_file.write('#!/bin/bash\n')
    sh_file.write('#BSUB -J '+job_ref+'\n')
    sh_file.write('#BSUB -o ' + experiment_path+'/logs/'+job_ref+'.o\n')
    sh_file.write('#BSUB -e ' + experiment_path+'/logs/'+job_ref+'.e\n')

    this_file_path = os.path.dirname(os.path.realpath(__file__))
    sh_file.write('python '+this_file_path+'/DataPreprocessingJob.py '+dataset_path+" "+experiment_path+" "+str(cv_partitions)+
                  " "+partition_method+" "+scale_data+" "+impute_data+" "+str(categorical_cutoff)+" "+export_initial_analysis+
                  " "+export_feature_correlations+" "+export_univariate+" "+class_label+" "+instance_label+" "+match_label+
                  " "+str(random_state)+'\n')
    sh_file.close()
    os.system('bsub < '+job_name)
    pass

if __name__ == '__main__':
    sys.exit(main(sys.argv))