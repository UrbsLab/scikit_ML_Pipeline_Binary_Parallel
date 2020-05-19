
import argparse
import os
import sys
import pandas as pd
import FeatureSelectionJob
import time

'''Sample Run Command:
python FeatureSelectionMain.py --output-path /Users/robert/Desktop/outputs --experiment-name test1
'''

def main(argv):
    #Parse arguments
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--output-path', dest='output_path', type=str, help='path to output directory')
    parser.add_argument('--experiment-name', dest='experiment_name', type=str, help='name of experiment (no spaces)')
    parser.add_argument('--max-features-to-keep', dest='max_features_to_keep', type=int,help='max features to keep. None if no max', default=2000)
    parser.add_argument('--filter-poor-features', dest='filter_poor_features', type=str, default='True')
    parser.add_argument('--top-results', dest='top_results', type=int,help='# top features to illustrate in figures', default=20)
    parser.add_argument('--graph-scores', dest='graph_scores', type=str, default='True')

    options = parser.parse_args(argv[1:])
    output_path = options.output_path
    experiment_name = options.experiment_name
    max_features_to_keep = options.max_features_to_keep
    filter_poor_features = options.filter_poor_features
    top_results = options.top_results
    graph_scores = options.graph_scores

    metadata = pd.read_csv(output_path + '/' + experiment_name + '/' + 'metadata.csv').values

    class_label = metadata[0, 1]
    instance_label = metadata[1, 1]
    do_mutual_info = metadata[3,1]
    do_multiSURF = metadata[4,1]

    # Argument checks
    if not os.path.exists(output_path):
        raise Exception("Output path must exist (from phase 1) before phase 3 can begin")

    if not os.path.exists(output_path + '/' + experiment_name):
        raise Exception("Experiment must exist (from phase 1) before phase 3 can begin")

    dataset_paths = os.listdir(output_path + "/" + experiment_name)
    dataset_paths.remove('logs')
    dataset_paths.remove('jobs')
    dataset_paths.remove('metadata.csv')
    for dataset_directory_path in dataset_paths:
        full_path = output_path + "/" + experiment_name + "/" + dataset_directory_path
        submitLocalJob(full_path,do_mutual_info,do_multiSURF,max_features_to_keep,filter_poor_features,top_results,graph_scores,class_label,instance_label)
        #submitClusterJob(full_path,do_mutual_info,do_multiSURF,max_features_to_keep,filter_poor_features,top_results,graph_scores,class_label,instance_label,output_path+'/'+experiment_name)

def submitLocalJob(full_path,do_mutual_info,do_multiSURF,max_features_to_keep,filter_poor_features,top_results,graph_scores,class_label,instance_label):
    FeatureSelectionJob.job(full_path,do_mutual_info,do_multiSURF,max_features_to_keep,filter_poor_features,top_results,graph_scores,class_label,instance_label)

def submitClusterJob(full_path,do_mutual_info,do_multiSURF,max_features_to_keep,filter_poor_features,top_results,graph_scores,class_label,instance_label,experiment_path):
    job_ref = str(time.time())
    job_name = experiment_path + '/jobs/' + job_ref + '_run.sh'
    sh_file = open(job_name, 'w')
    sh_file.write('#!/bin/bash\n')
    sh_file.write('#BSUB -J ' + job_ref + '\n')
    sh_file.write('#BSUB -o ' + experiment_path + '/logs/' + job_ref + '.o\n')
    sh_file.write('#BSUB -e ' + experiment_path + '/logs/' + job_ref + '.e\n')

    this_file_path = os.path.dirname(os.path.realpath(__file__))
    sh_file.write('python '+this_file_path+'/FeatureSelectionMain.py '+full_path+" "+do_mutual_info+" "+do_multiSURF+" "+
                  str(max_features_to_keep)+" "+filter_poor_features+" "+str(top_results)+" "+graph_scores+" "+class_label+" "+instance_label+'\n')
    sh_file.close()
    os.system('bsub < ' + job_name)
    pass

if __name__ == '__main__':
    sys.exit(main(sys.argv))