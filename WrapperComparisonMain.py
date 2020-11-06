'''
Only applied if there are multiple datasets withing experiment folder. Works under the assumption that the file structure generated by phases 1 to 6 have not changed.
'''

import argparse
import os
import sys
import time
import WrapperComparisonJob

'''Phase 7 of Machine Learning Analysis Pipeline:
Sample Run Command:
python WrapperComparisonMain.py --output-path /Users/robert/Desktop/outputs --experiment-name test1
'''

def main(argv):
    # Parse arguments
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--output-path', dest='output_path', type=str, help='path to output directory')
    parser.add_argument('--experiment-name', dest='experiment_name', type=str, help='name of experiment (no spaces)')
    parser.add_argument('--run-parallel',dest='run_parallel',type=str,help='path to directory containing datasets',default="True")
    parser.add_argument('--res-mem', dest='reserved_memory', type=int, help='reserved memory for the job (in Gigabytes)',default=4)
    parser.add_argument('--max-mem', dest='maximum_memory', type=int, help='maximum memory before the job is automatically terminated',default=15)

    options = parser.parse_args(argv[1:])
    output_path = options.output_path
    experiment_name = options.experiment_name
    run_parallel = options.run_parallel
    reserved_memory = options.reserved_memory
    maximum_memory = options.maximum_memory

    if run_parallel:
        submitClusterJob(output_path+'/'+experiment_name,reserved_memory,maximum_memory)
    else:
        submitLocalJob(output_path+'/'+experiment_name)

def submitLocalJob(experiment_path):
    WrapperComparisonJob.job(experiment_path)

def submitClusterJob(experiment_path,reserved_memory,maximum_memory):
    job_ref = str(time.time())
    job_name = experiment_path + '/jobs/P7_' + job_ref + '_run.sh'
    sh_file = open(job_name,'w')
    sh_file.write('#!/bin/bash\n')
    sh_file.write('#BSUB -q i2c2_normal'+'\n')
    sh_file.write('#BSUB -J '+job_ref+'\n')
    sh_file.write('#BSUB -R "rusage[mem='+str(reserved_memory)+'G]"'+'\n')
    sh_file.write('#BSUB -M '+str(maximum_memory)+'GB'+'\n')
    sh_file.write('#BSUB -o ' + experiment_path+'/logs/P7_'+job_ref+'.o\n')
    sh_file.write('#BSUB -e ' + experiment_path+'/logs/P7_'+job_ref+'.e\n')

    this_file_path = os.path.dirname(os.path.realpath(__file__))
    sh_file.write('python ' + this_file_path + '/WrapperComparisonJob.py ' + experiment_path + '\n')
    sh_file.close()
    os.system('bsub < ' + job_name)
    pass

if __name__ == '__main__':
    sys.exit(main(sys.argv))
