
import argparse
import os
import sys
import pandas as pd
import FeatureSelectionJob
import time

'''Phase 4 of Machine Learning Analysis Pipeline:
Sample Run Command:
python FeatureSelectionMain.py --output-path /Users/robert/Desktop/outputs --experiment-name test1
'''

def main(argv):
    #Parse arguments
    parser = argparse.ArgumentParser(description='')
    #No defaults
    parser.add_argument('--output-path', dest='output_path', type=str, help='path to output directory')
    parser.add_argument('--experiment-name', dest='experiment_name', type=str, help='name of experiment (no spaces)')
    #Defaults available
    parser.add_argument('--run-parallel',dest='run_parallel',type=str,help='path to directory containing datasets',default="True")
    parser.add_argument('--max-features', dest='max_features_to_keep', type=int,help='max features to keep. None if no max', default=2000)
    parser.add_argument('--filter-features', dest='filter_poor_features', type=str, help='filter out the worst performing features prior to modeling',default='True')
    parser.add_argument('--top-results', dest='top_results', type=int,help='number of top features to illustrate in figures', default=20)
    parser.add_argument('--export-scores', dest='export_scores', type=str,help='export figure summarizing average feature importance scores over cv partitions', default='True')
    parser.add_argument('--overwrite-cv', dest='overwrite_cv',type=str,help='overwrites working cv datasets with new feature subset datasets',default="True")

    options = parser.parse_args(argv[1:])
    output_path = options.output_path
    experiment_name = options.experiment_name
    run_parallel = options.run_parallel
    max_features_to_keep = options.max_features_to_keep
    filter_poor_features = options.filter_poor_features
    top_results = options.top_results
    export_scores = options.export_scores
    overwrite_cv = options.overwrite_cv

    metadata = pd.read_csv(output_path + '/' + experiment_name + '/' + 'metadata.csv').values

    class_label = metadata[0, 1]
    instance_label = metadata[1, 1]
    cv_partitions = int(metadata[4,1])
    do_mutual_info = metadata[7,1]
    do_multisurf = metadata[8,1]

    # Argument checks
    if not os.path.exists(output_path):
        raise Exception("Output path must exist (from phase 1) before phase 4 can begin")

    if not os.path.exists(output_path + '/' + experiment_name):
        raise Exception("Experiment must exist (from phase 1) before phase 4 can begin")

    dataset_paths = os.listdir(output_path + "/" + experiment_name)
    dataset_paths.remove('logs')
    dataset_paths.remove('jobs')
    dataset_paths.remove('jobsCompleted')
    dataset_paths.remove('metadata.csv')
    for dataset_directory_path in dataset_paths:
        full_path = output_path + "/" + experiment_name + "/" + dataset_directory_path
        if run_parallel:
            submitClusterJob(full_path,output_path+'/'+experiment_name,do_mutual_info,do_multisurf,max_features_to_keep,filter_poor_features,top_results,export_scores,class_label,instance_label,cv_partitions,overwrite_cv)
        else:
            submitLocalJob(full_path,output_path+'/'+experiment_name,do_mutual_info,do_multisurf,max_features_to_keep,filter_poor_features,top_results,export_scores,class_label,instance_label,cv_partitions,overwrite_cv)

def submitLocalJob(full_path,experiment_path,do_mutual_info,do_multisurf,max_features_to_keep,filter_poor_features,top_results,export_scores,class_label,instance_label,cv_partitions,overwrite_cv):
    FeatureSelectionJob.job(full_path,experiment_path,do_mutual_info,do_multisurf,max_features_to_keep,filter_poor_features,top_results,export_scores,class_label,instance_label,cv_partitions,overwrite_cv)

def submitClusterJob(full_path,experiment_path,do_mutual_info,do_multisurf,max_features_to_keep,filter_poor_features,top_results,export_scores,class_label,instance_label,cv_partitions,overwrite_cv):
    job_ref = str(time.time())
    job_name = experiment_path+'/jobs/P4_'+job_ref+'_run.sh'
    sh_file = open(job_name,'w')
    sh_file.write('#!/bin/bash\n')
    sh_file.write('#BSUB -q doi_normal'+'\n')
    sh_file.write('#BSUB -J '+job_ref+'\n')
    sh_file.write('#BSUB -R "rusage[mem=4G]"'+'\n')
    sh_file.write('#BSUB -M 15GB'+'\n')
    sh_file.write('#BSUB -o ' + experiment_path+'/logs/'+job_ref+'.o\n')
    sh_file.write('#BSUB -e ' + experiment_path+'/logs/'+job_ref+'.e\n')

    this_file_path = os.path.dirname(os.path.realpath(__file__))
    sh_file.write('python '+this_file_path+'/FeatureSelectionJob.py '+full_path+" "+experiment_path+" "+do_mutual_info+" "+do_multisurf+" "+
                  str(max_features_to_keep)+" "+filter_poor_features+" "+str(top_results)+" "+export_scores+" "+class_label+" "+instance_label+" "+str(cv_partitions)+" "+overwrite_cv+'\n')
    sh_file.close()
    os.system('bsub < ' + job_name)
    pass

if __name__ == '__main__':
    sys.exit(main(sys.argv))
