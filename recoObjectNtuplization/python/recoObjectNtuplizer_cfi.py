import FWCore.ParameterSet.Config as cms

electronNtuplizer = cms.EDAnalyzer(
    'electronNtuplizer',
    objectSrc = cms.InputTag('slimmedElectrons'),
    objectName = cms.untracked.string('Electron'),
)

jetNtuplizer = cms.EDAnalyzer(
    'jetNtuplizer',
    objectSrc = cms.InputTag('slimmedJets'),
    objectName = cms.untracked.string('Jet'),
)

fatJetNtuplizer = cms.EDAnalyzer(
    'jetNtuplizer',
    objectSrc = cms.InputTag('slimmedJetsAK8'),
    objectName = cms.untracked.string('FatJet')
)

muonNtuplizer = cms.EDAnalyzer(
    'muonNtuplizer',
    objectSrc = cms.InputTag('slimmedMuons'),
    objectName = cms.untracked.string('Muon'),
)

photonNtuplizer = cms.EDAnalyzer(
    'photonNtuplizer',
    objectSrc=cms.InputTag('slimmedPhotons'),
    objectName = cms.untracked.string('Photon'),
)

tauNtuplizer = cms.EDAnalyzer(
    'tauNtuplizer',
    objectSrc = cms.InputTag('slimmedTaus'),
    objectName = cms.untracked.string('Tau')
)

boostedTauNtuplizer = cms.EDAnalyzer(
    'tauNtuplizer',
    objectSrc = cms.InputTag('slimmedTausBoosted'),
    objectName = cms.untracked.string('BoostedTau')
)

recoObjectNtuplesSequence = cms.Sequence(
    electronNtuplizer +
    jetNtuplizer + 
    fatJetNtuplizer +
    muonNtuplizer + 
    photonNtuplizer + 
    tauNtuplizer +
    boostedTauNtuplizer
)