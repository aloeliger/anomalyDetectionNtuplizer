# A utility class for managing samples that we want to submit to condor for ntuplization.

import os
import copy
import subprocess
import hashlib

class condorSample():
    def __init__ (self, name: str, listOfSamples: list[str], outputFileName: str, configuration: str):
        self.name = name
        self.listOfSamples = listOfSamples
        self.outputFileName = f'{os.environ["CMSSW_BASE"]}/src/anomalyDetectionNtuplizer/samples/metaData/{outputFileName}'
        self.configuration = configuration
    
    @staticmethod
    def fromCondorSample(theOtherSample):
        newCondorSample = copy.deepcopy(theOtherSample)
        return newCondorSample
    
    def getFilesForSample(self, sample):
        dasQuery= f'file dataset={sample}'
        overallCommand = f'dasgoclient --query="{dasQuery}"'

        theProcess = subprocess.run(
            [overallCommand],
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        theFiles = theProcess.stdout.decode()
        theFiles = theFiles.split('\n')
        theFiles.remove('')

        return theFiles

    def getListOfAllFiles(self):
        allFiles = []
        for sample in self.listOfSamples:
            allFiles += self.getFilesForSample(sample)
        return allFiles
    
    def writeAllFiles(self,theFiles):
        with open(self.outputFileName, 'w') as theFile:
            theFile.write('\n'.join(theFiles))
    
    # returns true if our checked files match the file hash for the stuff on disk
    def currentFilesMatchHash(self, newFiles):
        with open(self.outputFileName, 'r') as theFile:
            oldFileContent = theFile.read()
        oldContentHash = hashlib.md5(oldFileContent.encode(), usedforsecurity=False)
        oldContentHash = oldContentHash.hexdigest()
        newContentHash = hashlib.md5('\n'.join(newFiles).encode(), usedforsecurity=False)
        newContentHash = newContentHash.hexdigest()

        return newContentHash == oldContentHash

    # Here's where the magic happens
    def getFileList(self):
        # no matter what happens we'll call to dasgo client to look up our files
        theFiles = self.getListOfAllFiles()
        if not os.path.exists(self.outputFileName): # new file, we need to write it.
            self.writeAllFiles(theFiles)
            return self.outputFileName
        else: #The file exists, let's check if it needs updating
            if self.currentFilesMatchHash(theFiles): #the files on disk do not need updating
                return self.outputFileName
            else:
                self.writeAllFiles(theFiles)
                return self.outputFileName
        return self.outputFileName #default, should never be called
    
    def createSubmissionCommand(self, submitDir, dagLocation, outputDir, jobName):
        
        command = [
            'farmoutAnalysisJobs',
            '--memory-requirements=4000',
            '--infer-cmssw-path',
            '--input-dir=/',
            '--assume-input-files-exist',
            f'--submit-dir={submitDir}',
            f'--output-dag-file={dagLocation}/dag',
            f'--output-dir={outputDir}',
            f'--input-file-list={self.getFileList()}', #will create or update the file if it doesn't exist
            '--use-singularity CentOS7',
            f'{jobName}',
            f'{self.configuration}',
            "'inputFiles=$inputFileNames'",
            "'outputFile=$outputFileName'",
        ]
        finalCommand = ' \\\n'.join(command)
        return finalCommand