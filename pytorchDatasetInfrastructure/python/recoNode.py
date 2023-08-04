# utility class for holding node information

import ROOT
import math

class recoNode():
    def __init__(self, pt, eta, phi, m, nodeType):
        self.lorentzVector = ROOT.Math.PtEtaPhiMVector(
            pt,
            eta,
            phi,
            m,
        )
        self.nodeType = nodeType
    
    @staticmethod
    def createListFromChain(chain, name):
        theList = []

        nObjects = getattr(chain, f'{name}_nObjects')
        ptVector = getattr(chain, f'{name}_ptVector')
        etaVector = getattr(chain, f'{name}_etaVector')
        phiVector = getattr(chain, f'{name}_phiVector')
        mVector = getattr(chain, f'{name}_massVector')
        types = [
            'Electron',
            'Jet',
            'FatJet',
            'Muon',
            'Photon',
            'Tau',
            'BoostedTau',
        ]
        nodeType = types.index(name)

        for i in range(nObjects):
            theList.append(
                recoNode(
                    ptVector[i],
                    etaVector[i],
                    phiVector[i],
                    mVector[i],
                    nodeType,
                )
            )

        return theList
    
    def DeltaR(self, secondNode):
        deltaEta = abs(self.lorentzVector.Eta() - secondNode.lorentzVector.Eta())
        deltaPhi = abs(self.lorentzVector.Phi() - secondNode.lorentzVector.Phi())
        if deltaPhi > math.pi:
            deltaPhi -= math.pi
        return math.sqrt(deltaEta**2+deltaPhi**2)

    def getFeatureVector(self):
        featureVector = [
            self.lorentzVector.Pt(),
            self.lorentzVector.Eta(),
            self.lorentzVector.Phi(),
            self.lorentzVector.M(),
        ]
        typeEncoding = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] #currently 7 types, but I don't like this...
        typeEncoding[self.nodeType] = 1.0
        featureVector+=typeEncoding
        return featureVector
