#module setup config for charge PFCandidate analyzers
import FWCore.ParameterSet.Config as cms

chargedHadronPFcandidateAnalyzer = cms.EDAnalyzer('PFcandidateAnalyzer',
                                                  candidateSource = cms.InputTag('packedPFCandidates'),
                                                  candidateCode = cms.int32(211),
                                                  treeName = cms.string('chargedHadronPFcands'))
