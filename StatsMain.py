
import argparse
import os
import sys
import time
import pandas as pd
import StatsJob

'''Sample Run Command:
python StatsMain.py --output-path /Users/robert/Desktop/outputs --experiment-name test1
'''

def main(argv):
    #Parse arguments
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--output-path', dest='output_path', type=str, help='path to output directory')
    parser.add_argument('--experiment-name', dest='experiment_name', type=str, help='name of experiment (no spaces)')
    parser.add_argument('--plot-ROC', dest='plot_ROC', type=str, default='True')
    parser.add_argument('--plot-PRC', dest='plot_PRC', type=str, default='True')
    parser.add_argument('--plot-FI', dest='plot_FI', type=str, default='True')

    options = parser.parse_args(argv[1:])
    output_path = options.output_path
    experiment_name = options.experiment_name
    plot_ROC = options.plot_ROC
    plot_PRC = options.plot_PRC
    plot_FI = options.plot_FI

    # Argument checks
    if not os.path.exists(output_path):
        raise Exception("Output path must exist (from phase 1) before phase 3 can begin")

    if not os.path.exists(output_path + '/' + experiment_name):
        raise Exception("Experiment must exist (from phase 1) before phase 3 can begin")

    metadata = pd.read_csv(output_path + '/' + experiment_name + '/' + 'metadata.csv').values

    class_label = metadata[0, 1]
    instance_label = metadata[1, 1]

    do_LR = metadata[5,1]
    do_DT = metadata[6,1]
    do_RF = metadata[7,1]
    do_NB = metadata[8,1]
    do_XGB = metadata[9,1]
    do_LGB = metadata[10,1]
    do_SVM = metadata[11,1]
    do_ANN = metadata[12,1]
    do_ExSTraCS = metadata[13,1]
    do_eLCS = metadata[14,1]
    do_XCS = metadata[15,1]

    encodedAlgos = ''
    encodedAlgos = encode(do_LR,encodedAlgos)
    encodedAlgos = encode(do_DT, encodedAlgos)
    encodedAlgos = encode(do_RF, encodedAlgos)
    encodedAlgos = encode(do_NB, encodedAlgos)
    encodedAlgos = encode(do_XGB, encodedAlgos)
    encodedAlgos = encode(do_LGB, encodedAlgos)
    encodedAlgos = encode(do_ANN, encodedAlgos)
    encodedAlgos = encode(do_SVM, encodedAlgos)
    encodedAlgos = encode(do_ExSTraCS, encodedAlgos)
    encodedAlgos = encode(do_eLCS, encodedAlgos)
    encodedAlgos = encode(do_XCS, encodedAlgos)

    # Iterate through datasets
    dataset_paths = os.listdir(output_path + "/" + experiment_name)
    dataset_paths.remove('logs')
    dataset_paths.remove('jobs')
    dataset_paths.remove('metadata.csv')
    for dataset_directory_path in dataset_paths:
        full_path = output_path + "/" + experiment_name + "/" + dataset_directory_path
        submitLocalJob(full_path,encodedAlgos,plot_ROC,plot_PRC,plot_FI,class_label,instance_label)
        #submitClusterJob(full_path,encodedAlgos,plot_ROC,plot_PRC,plot_FI,output_path+'/'+experiment_name,class_label,instance_label)

    #Clean up files
    os.remove(output_path+'/'+experiment_name+'/metadata.csv')

def submitLocalJob(full_path,encoded_algos,plot_ROC,plot_PRC,plot_FI,class_label,instance_label):
    StatsJob.job(full_path,encoded_algos,plot_ROC,plot_PRC,plot_FI,class_label,instance_label)

def submitClusterJob(full_path,encoded_algos,plot_ROC,plot_PRC,plot_FI,class_label,instance_label,experiment_path):
    job_ref = str(time.time())
    job_name = experiment_path + '/jobs/' + job_ref + '_run.sh'
    sh_file = open(job_name, 'w')
    sh_file.write('#!/bin/bash\n')
    sh_file.write('#BSUB -J ' + job_ref + '\n')
    sh_file.write('#BSUB -o ' + experiment_path + '/logs/' + job_ref + '.o\n')
    sh_file.write('#BSUB -e ' + experiment_path + '/logs/' + job_ref + '.e\n')

    this_file_path = os.path.dirname(os.path.realpath(__file__))
    sh_file.write('python '+this_file_path+'/StatsMain.py '+full_path+" "+encoded_algos+" "+plot_ROC+" "+plot_PRC+" "+plot_FI+" "+class_label+" "+instance_label+'\n')
    sh_file.close()
    os.system('bsub < ' + job_name)
    pass

def encode(do_algo,encodedAlgos):
    if do_algo == "True" or do_algo == "TRUE":
        encodedAlgos += '1'
    else:
        encodedAlgos += '0'
    return encodedAlgos

if __name__ == '__main__':
    sys.exit(main(sys.argv))