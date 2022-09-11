import FWCore.ParameterSet.Config as cms

from anomalyDetectionNtuplizer.PFcandidateAnalyzer.chargedHadronPFcandidateAnalyzer_cfi import chargedHadronPFcandidateAnalyzer
from anomalyDetectionNtuplizer.PFcandidateAnalyzer.neutralHadronPFcandidateAnalyzer_cfi import neutralHadronPFcandidateAnalyzer
from anomalyDetectionNtuplizer.PFcandidateAnalyzer.muonPFcandidateAnalyzer_cfi import muonPFcandidateAnalyzer
from anomalyDetectionNtuplizer.PFcandidateAnalyzer.electronPFcandidateAnalyzer_cfi import electronPFcandidateAnalyzer
from anomalyDetectionNtuplizer.PFcandidateAnalyzer.gammaPFcandidateAnalyzer_cfi import gammaPFcandidateAnalyzer

PFcandSequence = cms.Sequence(chargedHadronPFcandidateAnalyzer +
                              neutralHadronPFcandidateAnalyzer +
                              muonPFcandidateAnalyzer +
                              electronPFcandidateAnalyzer +
                              gammaPFcandidateAnalyzer)
