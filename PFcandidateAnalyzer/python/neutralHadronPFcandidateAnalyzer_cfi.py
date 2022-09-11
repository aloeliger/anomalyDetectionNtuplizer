#module setup config for charge PFCandidate analyzers
import FWCore.ParameterSet.Config as cms

neutralHadronPFcandidateAnalyzer = cms.EDAnalyzer('PFcandidateAnalyzer',
                                                  candidateSource = cms.InputTag('packedPFCandidates'),
                                                  candidateCode = cms.int32(130),
                                                  treeName = cms.string('neutralHadronPFcands'))
