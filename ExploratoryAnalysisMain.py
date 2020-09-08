
import sys
import os
import argparse
import glob
import ExploratoryAnalysisJob
import time
import csv

'''Phase 1 of Machine Learning Analysis Pipeline:
Sample Run Command:
python ExploratoryAalysisMain.py --data-path /Users/robert/Desktop/Datasets --output-path /Users/robert/Desktop/outputs --experiment-name test1
'''

def main(argv):
    #Parse arguments
    parser = argparse.ArgumentParser(description="")
    #No defaults
    parser.add_argument('--data-path',dest='data_path',type=str,help='path to directory containing datasets')
    parser.add_argument('--output-path',dest='output_path',type=str,help='path to output directory')
    parser.add_argument('--experiment-name', dest='experiment_name',type=str, help='name of experiment output folder (no spaces)')
    #Defaults available (but critical to check)
    parser.add_argument('--data-ext',dest='data_ext',type=str,help='name of datafile extension; only txt and csv permitted', default="txt")
    parser.add_argument('--run-parallel',dest='run_parallel',type=str,help='path to directory containing datasets',default="True")
    parser.add_argument('--class-label', dest='class_label', type=str, help='outcome label of all datasets', default="Class")
    parser.add_argument('--instance-label', dest='instance_label', type=str, default="")
    #Defaults available (but less critical to check)
    parser.add_argument('--cv',dest='cv_partitions',type=int,help='number of CV partitions',default=3)
    parser.add_argument('--partition-method',dest='partition_method',type=str,help='S or R or M for stratified, random, or matched, respectively',default="S")
    parser.add_argument('--match-label', dest='match_label', type=str, help='only applies when M selected for partition-method; indicates column with matched instance ids', default="")
    parser.add_argument('--categorical-cutoff', dest='categorical_cutoff', type=int,help='number of unique values after which a variable is considered to be quantitative vs categorical', default=10)
    parser.add_argument('--export-ea', dest='export_exploratory_analysis', type=str, help='run and export basic exploratory analysis files, i.e. unique value counts, missingness counts, class balance barplot',default="True")
    parser.add_argument('--export-fc', dest='export_feature_correlations', type=str, help='run and export feature correlation analysis (yields correlation heatmap)',default="True")
    parser.add_argument('--export-up', dest='export_univariate_plots', type=str, help='export univariate analysis plots (note: univariate analysis still output by default)',default="True")
    parser.add_argument('--random-state', dest='random_state', type=int, help='sets a specific random seed for reproducible results',default=42)

    options = parser.parse_args(argv[1:])
    data_path = options.data_path
    output_path = options.output_path
    experiment_name = options.experiment_name

    data_ext = options.data_ext
    run_parallel = options.run_parallel
    class_label = options.class_label
    if options.instance_label == '':
        instance_label = 'None'
    else:
        instance_label = options.instance_label

    cv_partitions = options.cv_partitions
    partition_method = options.partition_method
    if options.match_label == '':
        match_label = 'None'
    else:
        match_label = options.match_label
    categorical_cutoff = options.categorical_cutoff
    export_exploratory_analysis = options.export_exploratory_analysis
    export_feature_correlations = options.export_feature_correlations
    export_univariate_plots = options.export_univariate_plots
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

    #Run all datasets in target folder with specified file extension
    if data_ext == 'txt':
        if len(glob.glob(data_path+'/*.txt')) == 0: #Check that there is at least 1 dataset
            raise Exception("There must be at least one txt dataset in data_path directory")
        for datasetFilename in glob.glob(data_path+'/*.txt'): #Iterate through datasets
            if run_parallel:
                submitClusterJob(datasetFilename,output_path+'/'+experiment_name,cv_partitions,partition_method,categorical_cutoff,export_exploratory_analysis,export_feature_correlations,export_univariate_plots,class_label,instance_label,match_label,random_state)
            else:
                submitLocalJob(datasetFilename,output_path+'/'+experiment_name,cv_partitions,partition_method,categorical_cutoff,export_exploratory_analysis,export_feature_correlations,export_univariate_plots,class_label,instance_label,match_label,random_state)

    elif data_ext == 'csv':
        if len(glob.glob(data_path+'/*.csv')) == 0: #Check that there is at least 1 dataset
            raise Exception("There must be at least one csv dataset in data_path directory")
        for datasetFilename in glob.glob(data_path+'/*.csv'): #Iterate through datasets
            if run_parallel:
                submitClusterJob(datasetFilename,output_path+'/'+experiment_name,cv_partitions,partition_method,categorical_cutoff,export_exploratory_analysis,export_feature_correlations,export_univariate_plots,class_label,instance_label,match_label,random_state)
            else:
                submitLocalJob(datasetFilename,output_path+'/'+experiment_name,cv_partitions,partition_method,categorical_cutoff,export_exploratory_analysis,export_feature_correlations,export_univariate_plots,class_label,instance_label,match_label,random_state)
    else:
        raise Exception("File extension not recognized (only .txt or .csv permitted)")

    # Save metadata to file
    with open(output_path+'/'+experiment_name+'/'+'metadata.csv',mode='w') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["DATA LABEL", "VALUE"])
        writer.writerow(["class label",class_label])
        writer.writerow(["instance label", instance_label])
        writer.writerow(["random state",random_state])
        writer.writerow(["categorical cutoff",categorical_cutoff])
    file.close()

def submitLocalJob(dataset_path,experiment_path,cv_partitions,partition_method,categorical_cutoff,export_exploratory_analysis,export_feature_correlations,export_univariate_plots,class_label,instance_label,match_label,random_state):
    DataPreprocessingJob.job(dataset_path,experiment_path,cv_partitions,partition_method,categorical_cutoff,export_exploratory_analysis,export_feature_correlations,export_univariate_plots,class_label,instance_label,match_label,random_state)

def submitClusterJob(dataset_path,experiment_path,cv_partitions,partition_method,categorical_cutoff,export_exploratory_analysis,export_feature_correlations,export_univariate_plots,class_label,instance_label,match_label,random_state):
    job_ref = str(time.time())
    job_name = experiment_path+'/jobs/'+job_ref+'_run.sh'
    sh_file = open(job_name,'w')
    sh_file.write('#!/bin/bash\n')
    sh_file.write('#BSUB -q doi_normal'+'\n')
    sh_file.write('#BSUB -J '+job_ref+'\n')
    sh_file.write('bsub -R "rusage[mem=4G]"'+'\n')
    sh_file.write('bsub -M 15GB'+'\n')
    sh_file.write('#BSUB -o ' + experiment_path+'/logs/'+job_ref+'.o\n')
    sh_file.write('#BSUB -e ' + experiment_path+'/logs/'+job_ref+'.e\n')

    this_file_path = os.path.dirname(os.path.realpath(__file__))
    sh_file.write('python '+this_file_path+'/ExploratoryAnalysisJob.py '+dataset_path+" "+experiment_path+" "+str(cv_partitions)+
                  " "+partition_method+" "+str(categorical_cutoff)+" "+export_exploratory_analysis+
                  " "+export_feature_correlations+" "+export_univariate_plots+" "+class_label+" "+instance_label+" "+match_label+
                  " "+str(random_state)+'\n')
    sh_file.close()
    os.system('bsub < '+job_name)
    pass

if __name__ == '__main__':
    sys.exit(main(sys.argv))
