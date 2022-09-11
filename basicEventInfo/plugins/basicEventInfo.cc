// -*- C++ -*-
//
// Package:    anomalyDetectionNtuplizer/basicEventInfo
// Class:      basicEventInfo
//
/**\class basicEventInfo basicEventInfo.cc anomalyDetectionNtuplizer/basicEventInfo/plugins/basicEventInfo.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Andrew Loeliger
//         Created:  Sat, 10 Sep 2022 21:59:56 GMT
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

#include <TTree.h>
//
// class declaration
//

// If the analyzer does not use TFileService, please remove
// the template argument to the base class so the class inherits
// from  edm::one::EDAnalyzer<>
// This will improve performance in multithreaded jobs.


class basicEventInfo : public edm::one::EDAnalyzer<edm::one::SharedResources>  {
   public:
      explicit basicEventInfo(const edm::ParameterSet&);
      ~basicEventInfo();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


   private:
      virtual void beginJob() override;
      virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() override;

      // ----------member data ---------------------------
edm::Service<TFileService> theFileService;
TTree* basicInfoTree;
unsigned int run;
unsigned int lumi;
unsigned int evt;

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
basicEventInfo::basicEventInfo(const edm::ParameterSet& iConfig)
{
   //now do what ever initialization is needed
basicInfoTree = theFileService->make< TTree >("basicInfo", "Basic event information");
basicInfoTree->Branch("run", &run);
basicInfoTree->Branch("lumi", &lumi);
basicInfoTree->Branch("evt", &evt);

}


basicEventInfo::~basicEventInfo()
{

   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called for each event  ------------
void
basicEventInfo::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

  run  = iEvent.id().run();
  lumi = iEvent.id().luminosityBlock();
  evt  = iEvent.id().event();

  basicInfoTree->Fill();
}


// ------------ method called once each job just before starting event loop  ------------
void
basicEventInfo::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void
basicEventInfo::endJob()
{
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
basicEventInfo::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
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
DEFINE_FWK_MODULE(basicEventInfo);
