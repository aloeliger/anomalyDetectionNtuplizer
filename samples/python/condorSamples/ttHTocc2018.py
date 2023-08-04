from anomalyDetectionNtuplizer.samples.condorSample import condorSample
import os

ttHTocc2018 = condorSample(
    name='ttHTocc2018',
    listOfSamples=[
        '/ttHTocc_M125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM',
    ],
    outputFileName='ttHTocc2018.txt',
    configuration=os.environ['CMSSW_BASE']+'/src/anomalyDetectionNtuplizer/overall/python/completeNtuplizationMC_cff.py'
)