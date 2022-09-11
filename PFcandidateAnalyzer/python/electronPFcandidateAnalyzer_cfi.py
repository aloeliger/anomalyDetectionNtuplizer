#module setup config for charge PFCandidate analyzers
import FWCore.ParameterSet.Config as cms

electronPFcandidateAnalyzer = cms.EDAnalyzer('PFcandidateAnalyzer',
                                             candidateSource = cms.InputTag('packedPFCandidates'),
                                             candidateCode = cms.int32(11),
                                             treeName = cms.string('electronPFcands'))
