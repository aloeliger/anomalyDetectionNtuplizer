import FWCore.ParameterSet.Config as cms

randomSelectionFilter = cms.EDFilter('randomSelectionFilter',
                                     reductionRate = cms.double(100.0))
