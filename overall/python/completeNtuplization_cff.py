import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run2_2018_cff import Run2_2018

from FWCore.ParameterSet.VarParsing import VarParsing
options = VarParsing('analysis')
options.parseArguments()

process = cms.Process("ntuplizationTest", Run2_2018)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.RawToDigi_Data_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_data', '')

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(options.maxEvents) )
#process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(5) )

process.MessageLogger.cerr.FwkReport.reportEvery = 10000

process.options = cms.untracked.PSet()

#Input
#get the file and split it up

#process.source = cms.Source("PoolSource",
#                            fileNames=cms.untracked.vstring(options.inputFiles))
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(options.inputFiles)
)

#Output
process.TFileService = cms.Service("TFileService",
                                   fileName=cms.string(options.outputFile)
)

#Modules to be run
process.load("anomalyDetectionNtuplizer.basicEventInfo.basicEventInfo_cfi")
#process.load("anomalyDetectionNtuplizer.PFcandidateAnalyzer.PFcandSequence_cfi")
process.load("anomalyDetectionNtuplizer.recoObjectNtuplization.recoObjectNtuplizer_cfi")

#process.load("anomalyDetectionNtuplizer.randomSelectionFilter.randomSelectionFilter_cfi")
#process.randomSelectionFilter.reductionRate = 3000.0
#process.filterTask = cms.Task(process.randomSelectionFilter)
#process.filterPath = cms.Path(process.filterTask)
#process.filterPath = cms.Path(process.randomSelectionFilter)
#process.filterPath.associate(process.filterTask)

process.thePath = cms.Path(
    process.basicEventInfo +
    process.recoObjectNtuplesSequence
)

process.schedule = cms.Schedule(process.thePath)

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)

# Multi-threading
process.options.numberOfThreads=cms.untracked.uint32(1)
process.options.numberOfStreams=cms.untracked.uint32(0)
