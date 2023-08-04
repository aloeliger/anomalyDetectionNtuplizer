// system include files
#include <memory>
#include <iostream>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "TTree.h"

#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Photon.h"
#include "DataFormats/PatCandidates/interface/Tau.h"

namespace recoObjectNtuplization
{
    template<class T>
    class recoObjectNtuplizer: public edm::one::EDAnalyzer< edm::one::SharedResources >
    {
        public:
            explicit recoObjectNtuplizer(const edm::ParameterSet&);
            ~recoObjectNtuplizer() override {};

        private:
            void beginJob() override {};
            void analyze(const edm::Event&, const edm::EventSetup&) override;
            void endJob() override {};

            edm::InputTag objectSrc;
            edm::Service<TFileService> theFileService;

            std::string objectName;
            
            TTree* outputTree;
            unsigned int nObjects = 0;
            std::vector<double> ptVector;
            std::vector<double> etaVector;
            std::vector<double> phiVector;
            std::vector<double> massVector;
            std::vector<double> etVector;
            std::vector<int> chargeVector;
            std::vector<double> mtVector;
            std::vector<double> vxVector;
            std::vector<double> vyVector;
            std::vector<double> vzVector;
            // std::vector<double> dxyVector;
            // std::vector<double> dzVector;
            std::vector<size_t> numberOfDaughtersVector;
            std::vector<double> vertexChi2Vector;
    };

    template <class T>
    recoObjectNtuplizer<T>::recoObjectNtuplizer(const edm::ParameterSet& iConfig):
        objectSrc(iConfig.getParameter<edm::InputTag>("objectSrc")),
        objectName(iConfig.getUntrackedParameter<std::string>("objectName"))
    {
        usesResource("TFileService");
        consumes<std::vector<T>>(objectSrc);

        outputTree = theFileService->make<TTree>((objectName+"_info").c_str(), "4 vector information");
        outputTree->Branch((objectName+"_nObjects").c_str(), &nObjects);
        outputTree->Branch((objectName+"_ptVector").c_str(), &ptVector);
        outputTree->Branch((objectName+"_etaVector").c_str(), &etaVector);
        outputTree->Branch((objectName+"_phiVector").c_str(), &phiVector);
        outputTree->Branch((objectName+"_massVector").c_str(), &massVector);
        outputTree->Branch((objectName+"_etVector").c_str(), &etVector);
        outputTree->Branch((objectName+"_chargeVector").c_str(), &chargeVector);
        outputTree->Branch((objectName+"_mtVector").c_str(), &mtVector);
        outputTree->Branch((objectName+"_vxVector").c_str(), &vxVector);
        outputTree->Branch((objectName+"_vyVector").c_str(), &vyVector);
        outputTree->Branch((objectName+"_vzVector").c_str(), &vzVector);
        // outputTree->Branch((objectName+"_dxyVector").c_str(), &dxyVector);
        // outputTree->Branch((objectName+"_dzVector").c_str(), &dzVector);
        outputTree->Branch((objectName+"_nDaughters").c_str(), &numberOfDaughtersVector);
        outputTree->Branch((objectName+"_vertexChi2Vector").c_str(), &vertexChi2Vector);
    }

    template<class T>
    void recoObjectNtuplizer<T>::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
    {
        edm::Handle<std::vector<T>> objectHandle;
        iEvent.getByLabel(objectSrc, objectHandle);

        nObjects = objectHandle->size();

        for(auto theObject = objectHandle->begin();
            theObject != objectHandle->end();
            theObject++)
        {
            ptVector.push_back(theObject->pt());
            etaVector.push_back(theObject->eta());
            phiVector.push_back(theObject->phi());
            massVector.push_back(theObject->mass());
            etVector.push_back(theObject->et());
            chargeVector.push_back(theObject->charge());
            mtVector.push_back(theObject->mt());
            vxVector.push_back(theObject->vx());
            vyVector.push_back(theObject->vy());
            vzVector.push_back(theObject->vz());
            // dxyVector.push_back(theObject->dxy());
            // dzVector.push_back(theObject->dz());
            numberOfDaughtersVector.push_back(theObject->numberOfDaughters());
            vertexChi2Vector.push_back(theObject->vertexChi2());
        }

        outputTree->Fill();

        nObjects = 0;
        ptVector.clear();
        etaVector.clear();
        phiVector.clear();
        massVector.clear();
        etVector.clear();     
        chargeVector.clear();
        mtVector.clear();
        vxVector.clear();
        vyVector.clear();
        vzVector.clear();
        // dxyVector.clear();
        // dzVector.clear();
        numberOfDaughtersVector.clear();
        vertexChi2Vector.clear();
    }
}

typedef recoObjectNtuplization::recoObjectNtuplizer<pat::Electron> electronNtuplizer;
typedef recoObjectNtuplization::recoObjectNtuplizer<pat::Jet> jetNtuplizer;
typedef recoObjectNtuplization::recoObjectNtuplizer<pat::Muon> muonNtuplizer;
typedef recoObjectNtuplization::recoObjectNtuplizer<pat::Photon> photonNtuplizer;
typedef recoObjectNtuplization::recoObjectNtuplizer<pat::Tau> tauNtuplizer;

DEFINE_FWK_MODULE(electronNtuplizer);
DEFINE_FWK_MODULE(jetNtuplizer);
DEFINE_FWK_MODULE(muonNtuplizer);
DEFINE_FWK_MODULE(photonNtuplizer);
DEFINE_FWK_MODULE(tauNtuplizer);