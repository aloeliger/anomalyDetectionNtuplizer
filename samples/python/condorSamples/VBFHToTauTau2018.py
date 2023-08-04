from anomalyDetectionNtuplizer.samples.condorSample import condorSample
import os

VBFHToTauTau2018 = condorSample(
    name='VBFHToTauTau2018',
    listOfSamples=[
        '/VBFHToTauTau_M125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM',
    ],
    outputFileName='VBFHToTauTau2018.txt',
    configuration=os.environ['CMSSW_BASE']+'/src/anomalyDetectionNtuplizer/overall/python/completeNtuplizationMC_cff.py'
)