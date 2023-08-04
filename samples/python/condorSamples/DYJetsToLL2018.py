from anomalyDetectionNtuplizer.samples.condorSample import condorSample
import os

DYJetsToLL2018 = condorSample(
    name='DYJetsToLL2018',
    listOfSamples=[
        '/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/MINIAODSIM',
    ],
    outputFileName='DYJetsToLL2018.txt',
    configuration=os.environ['CMSSW_BASE']+'/src/anomalyDetectionNtuplizer/overall/python/completeNtuplizationMC_cff.py'
)