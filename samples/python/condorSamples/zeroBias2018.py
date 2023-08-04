from anomalyDetectionNtuplizer.samples.condorSample import condorSample
import os

ZeroBias2018_Even = condorSample(
    name='ZeroBias2018_Even',
    listOfSamples=[
        '/ZeroBias/Run2018A-UL2018_MiniAODv2-v1/MINIAOD',
        '/ZeroBias/Run2018B-UL2018_MiniAODv2-v1/MINIAOD',
        '/ZeroBias/Run2018C-UL2018_MiniAODv2-v1/MINIAOD',
        '/ZeroBias/Run2018D-UL2018_MiniAODv2-v1/MINIAOD',
    ],
    outputFileName='ZeroBias2018_Even.txt',
    configuration=os.environ['CMSSW_BASE']+'/src/anomalyDetectionNtuplizer/overall/python/randomEvenEventNtuplization_cff.py'
)

ZeroBias2018_Odd = condorSample(
    name='ZeroBias2018_Odd',
    listOfSamples=[
        '/ZeroBias/Run2018A-UL2018_MiniAODv2-v1/MINIAOD',
        '/ZeroBias/Run2018B-UL2018_MiniAODv2-v1/MINIAOD',
        '/ZeroBias/Run2018C-UL2018_MiniAODv2-v1/MINIAOD',
        '/ZeroBias/Run2018D-UL2018_MiniAODv2-v1/MINIAOD',
    ],
    outputFileName='ZeroBias2018_Odd.txt',
    configuration=os.environ['CMSSW_BASE']+'/src/anomalyDetectionNtuplizer/overall/python/randomOddEventNtuplization_cff.py'
)