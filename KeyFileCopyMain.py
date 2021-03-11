from distutils.dir_util import copy_tree
import argparse
import sys
import os
import glob
import shutil

def main(argv):
    #Parse arguments
    parser = argparse.ArgumentParser(description="")
    #No defaults
    parser.add_argument('--data-path',dest='data_path',type=str,help='path to directory containing datasets')
    parser.add_argument('--output-path',dest='output_path',type=str,help='path to output directory')
    parser.add_argument('--experiment-name', dest='experiment_name',type=str, help='name of experiment output folder (no spaces)')

    options = parser.parse_args(argv[1:])
    data_path = options.data_path
    output_path = options.output_path
    experiment_name = options.experiment_name

    if not os.path.exists(data_path):
        raise Exception("Provided data_path does not exist")

    #Create copied file summary folder
    os.mkdir(output_path+'/'+experiment_name+'/KeyFileCopy')

    #Copy Dataset comparisons if present
    if os.path.exists(output_path+'/'+experiment_name+'/DatasetComparisons'):
        #Make corresponding data folder
        os.mkdir(output_path+'/'+experiment_name+'/KeyFileCopy'+'/DatasetComparisons')
        copy_tree(output_path+'/'+experiment_name+'/DatasetComparisons', output_path+'/'+experiment_name+'/KeyFileCopy'+'/DatasetComparisons')

    #Create dataset name folders
    for datasetFilename in glob.glob(data_path+'/*'):
        dataset_name = datasetFilename.split('/')[-1].split('.')[0]
        if not os.path.exists(output_path+'/'+experiment_name+'/KeyFileCopy'+ '/' + dataset_name):
            os.mkdir(output_path+'/'+experiment_name+'/KeyFileCopy'+ '/' + dataset_name)
            os.mkdir(output_path+'/'+experiment_name+'/KeyFileCopy'+ '/' + dataset_name+'/results')
            #copy respective results folder
            copy_tree(output_path+'/'+experiment_name+ '/' + dataset_name+'/training'+'/results/', output_path+'/'+experiment_name+'/KeyFileCopy'+ '/' + dataset_name+'/results/')
            #Copy class balance
            shutil.copy(output_path+'/'+experiment_name+ '/' + dataset_name+'/exploratory/'+'ClassCounts.png', output_path+'/'+experiment_name+'/KeyFileCopy'+ '/' + dataset_name +'/' +'ClassCounts.png')
            shutil.copy(output_path+'/'+experiment_name+ '/' + dataset_name+'/exploratory/'+'ClassCounts.csv', output_path+'/'+experiment_name+'/KeyFileCopy'+ '/' + dataset_name +'/' +'ClassCounts.csv')
    #Copy metafile
    shutil.copy(output_path+'/'+experiment_name+ '/metadata.csv',output_path+'/'+experiment_name+'/KeyFileCopy'+ '/metadata.csv')

if __name__ == '__main__':
    sys.exit(main(sys.argv))
