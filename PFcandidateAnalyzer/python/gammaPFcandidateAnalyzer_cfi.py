#module setup config for charge PFCandidate analyzers
import FWCore.ParameterSet.Config as cms

gammaPFcandidateAnalyzer = cms.EDAnalyzer('PFcandidateAnalyzer',
                                          candidateSource = cms.InputTag('packedPFCandidates'),
                                          candidateCode = cms.int32(22),
                                          treeName = cms.string('gammaPFcands'))
