import sys
import os
sys.path.append (os.path.dirname(os.path.abspath("aa.tx")));
import shutil
from core.project.project import Project
import subprocess
from subprocess import call
import glob

#This script prepares a bash for execution of parallelization.py in  the cluster. It also modifies a copy of .fproj to point to correct working directory, project and video paths in the cluster.

#These parameters we get from command line, they must change with every project
#local_working_directory = sys.argv[1];
#cluster_working_directory = sys.argv[2];
#cluster_video_path = sys.argv[3];
#project_name = sys.argv[4];

# count lines in limits.txt form  https://gist.github.com/zed/0ac760859e614cd03652
def wccount(filename):
   out = subprocess.Popen(['wc', '-l', filename],
						stdout=subprocess.PIPE,
						stderr=subprocess.STDOUT
						).communicate()[0]
   return int(out.partition(b' ')[0])

# ONLY FOR TESTS
def prepareForClusterDummy(local_working_directory,cluster_working_directory,cluster_video_path,project_name):

	clusterScriptHeader = '#!/bin/bash\n#$ -S /bin/bash\n#$ -v TST=abc\n#$ -M casillas@ist.ac.at\n#$ -m ea\n#$-l mf=4000M\n#$ -l h_vmem=4000M\n#$ -pe openmp 4\n\nulimit -c 0'

	local_limits_path = local_working_directory+'limits.txt';
	local_project_path   = local_working_directory+project_name+'.fproj';
	local_temp_path = local_working_directory+'copy_for_cluster/'
	local_temp_project_path = local_temp_path+project_name+'.fproj';

	if not os.path.exists(local_temp_path):
	   os.mkdir(local_temp_path)

	 # copy project file and limits file
	shutil.copy(local_project_path,local_temp_project_path)
	shutil.copy(local_limits_path,local_temp_path)


	numLines = wccount(local_limits_path)

	# create bash script
	clusterScriptHeader += '\n#$ -t 1-'+str(numLines)+':1'

	clusterScriptBody = 'LIMIT=$(awk \"NR==$SGE_TASK_ID\" '+ cluster_working_directory+'/copy_for_cluster/limits.txt)\n'
	clusterScriptBody += 'python '+  'core/parallelization.py ' + cluster_working_directory + '/copy_for_cluster/ ' +  project_name + ' $LIMIT'

	scriptFile = open(local_temp_path+'run_ferda_parallel.sh','w');
	scriptFile.write(clusterScriptHeader+'\n\n'+clusterScriptBody);
	scriptFile.close()




def prepareAssemblyForCluster(numFiles,remoteParallelizationFile,localAssemblyScript):
	clusterScriptHeader = '#!/bin/bash\n#$ -S /bin/bash\n#$ -v TST=abc\n#$ -M casillas@ist.ac.at\n#$ -m a\n#$ -l mf=4000M\n#$ -l h_vmem=6000M\n#$ -l h_rt=4:00:00 \n#$ -pe openmp 1\n\nulimit -c 0'
	clusterScriptHeader += '\n#$ -t 1-'+str(numFiles)+':1'
	#clusterScriptBody = 'LIMIT=$(awk \"NR==$SGE_TASK_ID\" '+ remoteParallelizationFile+')\n'
	clusterScriptBody = 'LIMIT=$(awk \"NR==$SGE_TASK_ID\" '+ remoteParallelizationFile+')\n'
	clusterScriptBody +='export PYTHONPATH=/cluster/home/casillas/ferda/'+'\n'
	clusterScriptBody += 'module load graph-tool/2.10 \n'
	clusterScriptBody += 'python -m core.cluster_bg_computer ' + ' $LIMIT\n'



	scriptFile = open(localAssemblyScript,'w');
	scriptFile.write(clusterScriptHeader+'\n\n'+clusterScriptBody);
	scriptFile.close()


def prepareForCluster(local_working_directory,cluster_working_directory,cluster_video_path,project_name):

	#These parameters are fixed for a given cluster installation, therefore it does not make much sense to get them for every project
	cluster_Ferda_Dir = '/cluster/home/casillas/ferda/'
	clusterScriptHeader = '#!/bin/bash\n#$ -S /bin/bash\n#$ -v TST=abc\n#$ -M casillas@ist.ac.at\n#$ -m a\n#$ -l mf=4000M\n#$ -l h_vmem=6000M\n#$ -l h_rt=4:00:00 \n#$ -pe openmp 1\n\nulimit -c 0'

	local_limits_path = local_working_directory+'limits.txt';
	local_project_path   = local_working_directory+project_name+'.fproj';
	local_temp_path = local_working_directory+'copy_for_cluster/'
	local_temp_project_path = local_temp_path+project_name+'.fproj';
	print(local_temp_project_path)

	# create temporary directory with all things that have to be moved to the cluster
	if not os.path.exists(local_temp_path):
	   os.mkdir(local_temp_path)

	# copy project file and limits file and pkl files
	shutil.copy(local_project_path,local_temp_project_path)
	shutil.copy(local_limits_path,local_temp_path)
	for fileN in glob.glob(local_working_directory+"*.pkl"):
		shutil.copy(fileN,local_temp_path)

	# update fproj file -------------------
	p = Project()
	p.load(local_temp_project_path)
	p.video_paths = [cluster_video_path]
	p.working_directory = cluster_working_directory;
	for it in p.log.data_:
	   print it.action_name, it.data
	print("local_temp_path[:-1]   "+local_temp_path[:-1])
	p.save(local_temp_path[:-1])
	# save in tmp directory, not working directory.
	#-1 removes the trailing / since project.save does not expect it


	numLines = wccount(local_limits_path)

	# create bash script
        clusterScriptHeader += '\n#$ -t 1-'+str(numLines)+':1'

        clusterScriptBody = 'LIMIT=$(awk \"NR==$SGE_TASK_ID\" '+ cluster_working_directory+'copy_for_cluster/limits.txt)\n'
        clusterScriptBody +='module load mahotas \n'
        clusterScriptBody +='module load graph-tool/2.12 \n'
        clusterScriptBody += 'python '+ cluster_Ferda_Dir+ 'core/parallelization.py ' + cluster_working_directory + 'copy_for_cluster/ ' +  project_name + ' $LIMIT'

	scriptFile = open(local_temp_path+'run_ferda_parallel.sh','w');
	scriptFile.write(clusterScriptHeader+'\n\n'+clusterScriptBody);
	scriptFile.close()

	return numLines;
