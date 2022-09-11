#module setup config for charge PFCandidate analyzers
import FWCore.ParameterSet.Config as cms

muonPFcandidateAnalyzer = cms.EDAnalyzer('PFcandidateAnalyzer',
                                         candidateSource = cms.InputTag('packedPFCandidates'),
                                         candidateCode = cms.int32(13),
                                         treeName = cms.string('muonPFcands'))
