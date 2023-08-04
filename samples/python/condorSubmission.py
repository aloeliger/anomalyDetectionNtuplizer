# A utility class for representing the entire condor submission process

import os
import datetime
import pickle
from anomalyDetectionNtuplizer.samples.condorSample import condorSample

class condorSubmission():
    def __init__(
        self, 
        name: str,
        outputFileName: str,
        submissionPath: str = f'/nfs_scratch/{os.environ["USER"]}/',
        submissionName = datetime.datetime.now().strftime('%d%b%Y_%H%M_AnomalyBumpHuntNtuples'),
        listOfCondorSamples: list[condorSample] = [],
    ):
        # The creation info for this particular submission
        self.name = name
        self.creationTime = datetime.datetime.now()
        self.outputFileLocation = f'{os.environ["CMSSW_BASE"]}/src/anomalyDetectionNtuplizer/samples/metaData/{outputFileName}'

        # The samples associated with this submission
        self.listOfCondorSamples = listOfCondorSamples

        # Things for deriving the sample submissions
        self.submissionPath = f'{submissionPath}/{submissionName}'
        self.outputDir = f'/store/user/aloeliger/{submissionName}/' #don't get this confused with output file location
        self.submissionName = submissionName
    
    @staticmethod
    def loadFromPickle(pickleLocation):
        with open(pickleLocation, 'rb') as theFile:
            theObj = theFile.load(theFile)
        return theObj

    def saveToPickle(self):
        with open(self.outputFileLocation, 'wb') as theFile:
            pickle.dump(self, theFile)
    
    def createSubmissionFile(self):
        with open(f'{self.submissionName}_finalSubmission.sh', 'w') as finalSubmissionScript:
            for sample in self.listOfCondorSamples:
                sampleDagLocation = self.submissionPath+f'/{sample.name}/dags'
                sampleSubmissionLocation = self.submissionPath+f'/{sample.name}/submit'
                sampleOutputDir = self.outputDir+f'/{sample.name}/'
                sampleJobName = f'{sample.name}_{self.submissionName}'

                os.makedirs(sampleDagLocation, exist_ok=True)

                finalSubmissionScript.write(f'# Submission for sample name: {sample.name}')
                finalSubmissionScript.write('\n')
                finalSubmissionScript.write(
                    sample.createSubmissionCommand(
                        submitDir=f'{sampleSubmissionLocation}',
                        dagLocation=f'{sampleDagLocation}',
                        outputDir=f'{sampleOutputDir}',
                        jobName=sampleJobName
                    )
                )
                finalSubmissionScript.write('\n')
                finalSubmissionScript.write('\n')
    
    def createSubmission(self):
        print('Creating submission file')
        self.createSubmissionFile()
        print('Pickling self for posterity...')
        self.saveToPickle()

