import sys
import os
import datetime
import shutil
import auxForRunSeveral as ars
import prepare_for_cluster as pfc
import subprocess

# Usage:
# 0) these scripts (runSeveralOnCluster, auxForRunSeveral and prepare_for_cluster) must be saved in the root of the ferda file
# 0) set up paswordless login into the cluster
# 0) set the correct values in the "CONFIG" section of this script
#
# 1) Initialize the ferda projects
# 2) Copy the video files to the cluster
# 3) Create a file (localProjectListFile) that has one line per project, formated as follows:
#
#      projectName,localProjectPath,remoteVideoPath
#
#    the separator (,) can be modified in the CONFIG section if need be
# 4) run this script as follows:
#
#      python runSeveralOnCluster.py localProjectListFile clusterWD
#
#    This will a) create on the cluster the project files, with proper paths
#              b) use clusterRunCommand to submit the ferda jobs to the cluster
#              c) create a log on the local machine of what was sent to run
#              d) create a script on the local machine, afterFinish+date+.sh on the same folder as locelProjectListFile, that must be run when all the executions are finished on the cluster
#
# 5) run afterFinish+date+.sh on the local machine, to retrieve all the temp files created on the cluster back to the local machine
# 6) open each project in ferda and clean up the graph.
#
# IMPORTANT!
# * If a project or path name includes spaces, it will be skipped!!
# * The relevant project and video files, both local and on the cluster, must not change during the entire 1-6 steps
# * Any changes to the parameters that control the execution on the cluster must be made to prepare_for_cluster.py


## ---------------------------------
# ---- CONFIG BEGIN  -----
remoteUser = "casillas"
remoteHost = "bjoern22.ista.local";
# Ferda files will be copied inside this directory in the cluster and retrieved from there
remoteWorkingPath = ars.fixPath("/cluster/home/casillas/COLONIESWD2/");  #!!!!
remoteFerdaPath = "/cluster/home/casillas/new_ferda/"

# Where to save THIS SCRIPT'S logs
localLogPath = ars.fixPath("/home/casillas/Documents/COLONIESWD/cluster/logs/");

# normally qsub  use cat for testing purposes only
clusterRunCommand = "qsub"  # qsub after testing

separatorInFile = ',';
numPartsPerMat = 10;
# ---- CONFIG END  -----


localProjectListFile = sys.argv[1];

## ---------------------------------
##Initializations
listOfProjectsToAssemble = ''  # Textfile containing the arguments for cluster_bg_computer
remoteCommandQueue = [];  # Here we will put the commands to copy the projs to the cluster, create all necesseary folders

date = ars.longDate();

# We will save a log of what and when was sent to the cluster
ars.exists_local_directory(localLogPath);
logFile = localLogPath + "ClusterRun" + date + ".log"
logText = "On " + date + " the execution was started to run on the cluster located in " + remoteHost + "using the files provided in " + localProjectListFile + "\n";
# We will write a script that is to be run when the cluster is finished, to retrieve all the files
localWorkingPath = os.path.dirname(os.path.abspath(localProjectListFile));
localAfterFinishScriptFile = localWorkingPath + "/" + "afterFinish" + date + ".sh";
parallelizationListFile = "parallelization" + date + ".txt";
localParallelizationListFile = localWorkingPath + "/" + parallelizationListFile
localAfterFinishScript = "";

## ---------------------------------
# we read the different project locations and names from the file, one by one
numberSkipped = 0;
totNumLines = 0;
lines = [line.rstrip('\n') for line in open(localProjectListFile)]
numFiles = 0;  # This is the actual number of files that will be submitted
for l in lines:
    lsp = l.split(separatorInFile);
    if (len(lsp) != 3):
        continue
    totNumLines += 1;

    projectName = lsp[0];
    localProjectPath = ars.fixPath(lsp[1]);
    remoteVideoPath = lsp[2];

    if (" " in projectName + localProjectPath):
        print(
        "\nE: project " + projectName + " Contains some space either in its name or in one of its paths, this is not supported, SKIPPING");
        numberSkipped += 1;
        continue

    ##for each, we see if: a) the proj exists b) the videos are on the cluster c) the folder is not on the cluster

    # We check if the local project file is here
    localProjectFile = localProjectPath + projectName + ".fproj"
    if not os.path.exists(localProjectFile):
        print(
        "\nE: project " + projectName + " was supossed to be in " + localProjectPath + " but wasn't found there. This project is SKIPPED");
        numberSkipped += 1;
        continue

    # We find out what the video path exists in the cluster
    if not ars.exists_remote(remoteUser + "@" + remoteHost, remoteVideoPath):
        print(
        "\nE: project " + projectName + " was supossed to have it's videos in " + remoteVideoPath + " but the path wasn't there. This project is SKIPPED");
        numberSkipped += 1;
        continue

    # We check that the project files are not already on the cluster, and warn if they are, to avoid unwanted overwrites
    remoteProjectPath = remoteWorkingPath + projectName + "WD/";
    if ars.exists_remote(remoteUser + "@" + remoteHost, remoteProjectPath):
        print(
        "\nW: project " + projectName + "is already on the cluster. Maybe it was submitted before. Do you wish to submit it again? ATTENTION: this will overwrite all work done on that project.");
        answer = raw_input('Submit again (y/n)');
        if not ((answer == 'y') or (answer == 'Y')):
            print("will not overwrite, skipping project");
            numberSkipped += 1;
            continue
    else:
        addFolderCommand = "ssh " + remoteUser + "@" + remoteHost + " mkdir " + remoteProjectPath
        remoteCommandQueue = [(addFolderCommand,numFiles)] + remoteCommandQueue;

    # If all are satisfied, we: 1) generate the temp for cluster 2) enque the scp command 3) add the corresponding qsub line to runAll.sh  4) add the corresponding line to the log 5) add the lines to scp the temp files back to the afterFinish.sh

    logText += "\nWILL PROCESS: " + projectName + " : " + localProjectPath + " -> " + remoteProjectPath + "," + remoteVideoPath;

    print (
    "\nWILL PROCESS: " + projectName + " : " + localProjectPath + " -> " + remoteProjectPath + "," + remoteVideoPath);

    # prepare the copy for cluster
    numParts = pfc.prepareForCluster(localProjectPath, remoteProjectPath, remoteVideoPath, projectName,
                                     sizeMats=numPartsPerMat, cluster_Ferda_Dir=remoteFerdaPath)
    pfc.write_exportLimits (localProjectPath+ "copy_for_cluster/", numParts,sizeMats=numPartsPerMat)
    # enqueue the command that will copy the project to the cluster
    scpCommand = "scp -r " + localProjectPath + "copy_for_cluster/" + " " + remoteUser + "@" + remoteHost + ":" + remoteProjectPath;
    remoteCommandQueue = [(scpCommand,numFiles)] + remoteCommandQueue

    # write what cluster_bg_computer need to listOfProjectsToAssemble
    listOfProjectsToAssemble += remoteProjectPath + "copy_for_cluster/" + " " + projectName + " " + str(numParts) + "\n"

    # enqueue the command that will submit the job for execution in the cluster
    submitCommand = "ssh " + remoteUser + "@" + remoteHost + " " + clusterRunCommand + " " + remoteProjectPath + "copy_for_cluster/run_ferda_parallel.sh"
    remoteCommandQueue = [(submitCommand,numFiles)] + remoteCommandQueue
    submitCommand = "ssh " + remoteUser + "@" + remoteHost + " " + clusterRunCommand + " " + remoteProjectPath + "copy_for_cluster/run_ferda_export.sh"
    remoteCommandQueue = [(submitCommand,numFiles)] + remoteCommandQueue
    submitCommand = "ssh " + remoteUser + "@" + remoteHost + " " + clusterRunCommand + " " + remoteProjectPath + "copy_for_cluster/run_matlab_convert.sh"
    remoteCommandQueue = [(submitCommand,numFiles)] + remoteCommandQueue


    numFiles += 1

    localVideoPath = ars.get_videopath(localProjectFile)

    # and add the command to the afterfinish script to retrieve all that this produces
#localAfterFinishScript += "scp  " + remoteUser + "@" + remoteHost + ":" + remoteProjectPath + "copy_for_cluster/*.fproj" + " " + localProjectPath + "\n";
    localAfterFinishScript += "scp  " + remoteUser + "@" + remoteHost + ":" + remoteProjectPath + "copy_for_cluster/*.mat" + " " + localProjectPath + "copy_for_cluster/"+"\n";
    #localAfterFinishScript += "scp  " + remoteUser + "@" + remoteHost + ":" + remoteProjectPath + "copy_for_cluster/*.pkl" + " " + localProjectPath + "\n";
    #localAfterFinishScript += "scp  " + remoteUser + "@" + remoteHost + ":" + remoteProjectPath + "copy_for_cluster/*.sqlite3" + " " + localProjectPath + "\n";
    #localAfterFinishScript += "python change_property.py " + localProjectFile + " " + localProjectPath + " working_directory\n";
    #localAfterFinishScript += "python change_property.py " + localProjectFile + " " + localVideoPath[
    #    0] + " video_paths\n";

# Write and copy the file containing the list of projects to assemble
F = open(localParallelizationListFile, "w");
F.write(listOfProjectsToAssemble)
F.close()
scpCommand = "scp " + localParallelizationListFile + " " + remoteUser + "@" + remoteHost + ":" + remoteFerdaPath;


remoteParallelizationFile = remoteFerdaPath + parallelizationListFile;



## ---------------------------------
# Once we've finished reading all the lines, we do all the  remote commands
logText += "\nSUMMARY: of " + str(totNumLines) + " lines in the file " + str(numberSkipped) + " where skipped\n"
print("\nSUMMARY of " + str(totNumLines) + " in the project file " + str(
    numberSkipped) + " were skipped. The rest are ready to be copied & submitted for execution.\n")
answer = raw_input('Proceed (y/n)');
if ((answer == 'y') or (answer == 'Y')):
    lastFile = -1;
    while len(remoteCommandQueue) > 0:
        (com,numFile) = remoteCommandQueue.pop();
        if numFile != lastFile:
            jobNumber = 0;
            lastFile = numFile;
        comSplit = com.split(" ");

        #Include hook-job number, if this is a job submission
        if (jobNumber != 0) and (clusterRunCommand in comSplit):
            comPos = comSplit.index(clusterRunCommand)+1;
            newCom = comSplit[:comPos]+["-hold_jid",str(jobNumber)]+comSplit[comPos:];
            comSplit = newCom;
            print("N");

        print(numFile)
        print(comSplit)
        r = subprocess.check_output(comSplit);

        pos = r.find("Your job-array ");
        if pos != -1:  #This was a job submission
            print("FOUND "+str(pos)+" .. \t"),
            r2 = r[pos+15:];
            jobNumber = int(r2[:r2.find('.')]);
            firstJobSubmitted = True;
            print("Job number = "+str(jobNumber)+"\n")



    # We write the script that is to be run when execution in the cluster is finished
    F = open(localAfterFinishScriptFile, "w");
    F.write(localAfterFinishScript)
    F.close()

    #print("Once the tracking jobs are finished, put the following script on the cluster's que:  " + remoteAssemblyScript)
    print("\n Once the above is finished, run the following on the local machine to get the files back:  " + localAfterFinishScript)

## ---------------------------------
logText += "\n\n Finished execution of script at " + ars.longDate();
# We write the log
F = open(logFile, "w");
F.write(logText);
F.close();
