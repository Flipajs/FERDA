import sys
import os
sys.path.append (os.path.dirname(os.path.abspath("aa.tx")));
from core.project.project import Project
import subprocess
from subprocess import call
import glob
from change_property import change_pr
import shutil
import numpy as np

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
	clusterScriptHeader = '#!/bin/bash\n#$ -S /bin/bash\n#$ -v TST=abc\n#$ -M casillas@ist.ac.at\n#$ -m ea\n#$-l mf=4000M\n#$ -l h=!gpu62\n#$ -l h_vmem=4000M\n#$ -pe openmp 4\n\nulimit -c 0'

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
	clusterScriptHeader = '#!/bin/bash\n#$ -S /bin/bash\n#$ -v TST=abc\n#$ -M casillas@ist.ac.at\n#$ -m a\n#$ -l mf=8000M\n#$ -q bea.q\n#$ -l h_vmem=8000M\n#$ -l h_rt=4:00:00 \n#$ -pe openmp 1\n\nulimit -c 0'
	clusterScriptHeader += '\n#$ -t 1-'+str(numFiles)+':1'
	clusterScriptBody = 'LIMIT=$(awk \"NR==$SGE_TASK_ID\" '+ remoteParallelizationFile+')\n'
	clusterScriptBody +='export PYTHONPATH=/cluster/home/casillas/ferda/'+'\n'
	clusterScriptBody += 'module load graph-tool/2.10 \n'
	clusterScriptBody += 'python -m core.cluster_bg_computer ' + ' $LIMIT\n'

	scriptFile = open(localAssemblyScript,'w');
	scriptFile.write(clusterScriptHeader+'\n\n'+clusterScriptBody);
	scriptFile.close()

def prepareForCluster(local_working_directory,cluster_working_directory,cluster_video_path,project_name,sizeMats,msdThr=1.3,singleThr=1.8,arenaDiameter=90,cluster_Ferda_Dir = '/cluster/home/casillas/ferda/',clusterMatlabDirectory='/cluster/home/casillas/colony_analysis/'):
	#These parameters are fixed for a given cluster installation, therefore it does not make much sense to get them for every project

	clusterScriptHeader = '#!/bin/bash\n#$ -S /bin/bash\n#$ -v TST=abc\n#$ -M casillas@ist.ac.at\n#$ -m a\n#$ -l mf=8000M\n#$ -q bea.q\n#$ -l h_vmem=8000M\n#$ -l h_rt=4:00:00 \n#$ -pe openmp 1\n\nulimit -c 0'

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
	# already copied above, now we unpickle and change two of its properties ## optimisation, make sure optimisation is used
	change_pr(local_temp_project_path,cluster_video_path,'video_paths');
	change_pr(local_temp_project_path,cluster_working_directory,'working_directory');
	change_pr(local_temp_project_path,'TRUE','optimisation');
	change_pr(local_temp_project_path,55,'min_area'); 


	numLines = wccount(local_limits_path)

	#TRACK create bash script
	clusterScriptBody = '\n#$ -N '+project_name+'TRACK \n#$ -t 1-'+str(numLines)+':1\n'
	clusterScriptBody += 'LIMIT=$(awk \"NR==$SGE_TASK_ID\" '+ cluster_working_directory+'copy_for_cluster/limits.txt)\n'
	clusterScriptBody +='module load mahotas \n'
	clusterScriptBody +='module load graph-tool/2.12 \n'
	clusterScriptBody += 'python '+ cluster_Ferda_Dir+ 'core/parallelization.py ' + cluster_working_directory + 'copy_for_cluster/ ' +  project_name + ' $LIMIT'

	scriptFile = open(local_temp_path+'run_ferda_parallel.sh','w');
	scriptFile.write(clusterScriptHeader+'\n\n'+clusterScriptBody);  #!!!
	scriptFile.close()

	#EXPORT create bash script
	num_exportedParts=int(np.ceil(float(numLines)/sizeMats))
	clusterScriptBody = '\n#$ -N '+project_name+'EXPORT  \n#$ -t 1-'+str(num_exportedParts)+':1\n'
	clusterScriptBody += 'LIMIT=$(awk \"NR==$SGE_TASK_ID\" '+ cluster_working_directory+'copy_for_cluster/exportLimits.txt)\n'
	clusterScriptBody +='module load graph-tool/2.12 \n'
	clusterScriptBody += 'cd '+ cluster_Ferda_Dir+'\n'
	clusterScriptBody += 'python -m scripts.export.export_part '+ cluster_working_directory + 'copy_for_cluster/ ' + cluster_working_directory + 'copy_for_cluster/ ' + ' $LIMIT'

	scriptFile = open(local_temp_path+'run_ferda_export.sh','w');
	scriptFile.write(clusterScriptHeader+'\n\n'+clusterScriptBody); #!!!
	scriptFile.close()

	#CONVERT create bash script
	clusterScriptHeader ='#    !/bin/bash   \n'+'#\n'+'#$ -S /bin/bash\n'+'#$ -M casillas@ist.ac.at\n'+'#$ -m ea\n';
	clusterScriptHeader +='#$ -N '+project_name+'CONVERT\n'+'#$ -l mf=16G\n'+'#$ -l h_vmem=16G\n'+'#$ -q gasper.q\n';
	clusterScriptHeader +='##$ -q big.q\n'+'##$ -P bigserver\n'+'##$ -cwd\n'+'#$ -l matlab=1\n'+'##$ -q big.q@bignick21.ista.local\n';
	clusterScriptHeader +='#$ -t 1-1\n';

	clusterScriptBody = 'cd '+clusterMatlabDirectory+'\n'
	clusterScriptBody +='matlab -nojvm -nodisplay -singleCompThread -r "createTracksObject(\''+cluster_working_directory + 'copy_for_cluster/\','+str(num_exportedParts)+','+str(sizeMats)+','+str(msdThr)+','+str(singleThr)+','+str(arenaDiameter)+',1); exit;"'

	scriptFile = open(local_temp_path+'run_matlab_convert.sh','w');
	scriptFile.write(clusterScriptHeader+'\n\n'+clusterScriptBody);
	scriptFile.close()



	return numLines;

def write_exportLimits (local_temp_path, numLines,sizeMats=1):
    exportLimits = open(local_temp_path+'exportLimits.txt','w');
    min_tracklet_length=1
    pts_export=1

    num_exportedParts=int(np.ceil(float(numLines)/sizeMats))
    current_start=0
    for i in range (num_exportedParts):
        if (i+1)*sizeMats<=numLines:
            current_partNum=sizeMats
        else:
            current_partNum=numLines-current_start

        exportLimits.write(str(current_start)+' '+str(current_partNum)+' '+str(min_tracklet_length)+' '+ str(pts_export) +'\n')
        current_start+=sizeMats
    exportLimits.close()
