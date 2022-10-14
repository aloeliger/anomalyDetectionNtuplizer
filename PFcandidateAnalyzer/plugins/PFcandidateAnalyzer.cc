// -*- C++ -*-
//
// Package:    anomalyDetectionNtuplizer/PFcandidateAnalyzer
// Class:      PFcandidateAnalyzer
//
/**\class PFcandidateAnalyzer PFcandidateAnalyzer.cc anomalyDetectionNtuplizer/PFcandidateAnalyzer/plugins/PFcandidateAnalyzer.cc

   Description: [one line class summary]

   Implementation:
   [Notes on implementation]
*/
//
// Original Author:  Andrew Loeliger
//         Created:  Sat, 10 Sep 2022 19:19:30 GMT
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"

#include <TTree.h>
#include <string>
//
// class declaration
//

// If the analyzer does not use TFileService, please remove
// the template argument to the base class so the class inherits
// from  edm::one::EDAnalyzer<>
// This will improve performance in multithreaded jobs.


class PFcandidateAnalyzer : public edm::one::EDAnalyzer<edm::one::SharedResources>  {
public:
  explicit PFcandidateAnalyzer(const edm::ParameterSet&);
  ~PFcandidateAnalyzer();

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


private:
  virtual void beginJob() override;
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
  virtual void endJob() override;

  // ----------member data ---------------------------
  TTree* theTree;
  edm::Service< TFileService > theFileService;
  edm::EDGetTokenT< std::vector< pat::PackedCandidate > > candidateToken;
  int candidateCode;
  std::string treeName;
  

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
  std::vector<double> dxyVector;
  std::vector<double> dzVector;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
PFcandidateAnalyzer::PFcandidateAnalyzer(const edm::ParameterSet& iConfig):
  candidateToken( consumes<std::vector< pat::PackedCandidate > >(iConfig.getParameter< edm::InputTag >("candidateSource")))
{
  //now do what ever initialization is needed
  candidateCode = iConfig.getParameter<int>("candidateCode");
  treeName = iConfig.getParameter<std::string>("treeName");
  
  theTree = theFileService->make< TTree >(treeName.c_str(), "List of pf candidate 4 vector and charge info in the event");
  theTree->Branch("nObjects", &nObjects);
  theTree->Branch("ptVector", &ptVector);
  theTree->Branch("etaVector", &etaVector);
  theTree->Branch("phiVector", &phiVector);
  theTree->Branch("massVector", &massVector);
  theTree->Branch("etVector", &etVector);
  theTree->Branch("chargeVector", &chargeVector);
  theTree->Branch("mtVector", &mtVector);
  theTree->Branch("vxVector", &vxVector);
  theTree->Branch("vyVector", &vyVector);
  theTree->Branch("vzVector", &vzVector);
  theTree->Branch("dxyVector", &dxyVector);
  theTree->Branch("dzVector", &dzVector);
}


PFcandidateAnalyzer::~PFcandidateAnalyzer()
{

  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called for each event  ------------
void
PFcandidateAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;

  edm::Handle< std::vector<pat::PackedCandidate> > candidateHandle;
  iEvent.getByToken(candidateToken, candidateHandle);

  for(auto theCandidate = candidateHandle->begin();
      theCandidate != candidateHandle->end();
      theCandidate++)
    {
      if(std::abs(theCandidate->pdgId())!=candidateCode) continue;
      nObjects++;
      ptVector.push_back(theCandidate->pt());
      etaVector.push_back(theCandidate->eta());
      phiVector.push_back(theCandidate->phi());
      massVector.push_back(theCandidate->mass());
      etVector.push_back(theCandidate->et());
      chargeVector.push_back(theCandidate->charge());
      mtVector.push_back(theCandidate->mt());
      vxVector.push_back(theCandidate->vx());
      vyVector.push_back(theCandidate->vy());
      vzVector.push_back(theCandidate->vz());
      dxyVector.push_back(theCandidate->dxy());
      dzVector.push_back(theCandidate->dz());
    }
  theTree->Fill();

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
  dxyVector.clear();
  dzVector.clear();
}


// ------------ method called once each job just before starting event loop  ------------
void
PFcandidateAnalyzer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void
PFcandidateAnalyzer::endJob()
{
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
PFcandidateAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);

  //Specify that only 'tracks' is allowed
  //To use, remove the default given above and uncomment below
  //ParameterSetDescription desc;
  //desc.addUntracked<edm::InputTag>("tracks","ctfWithMaterialTracks");
  //descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(PFcandidateAnalyzer);
