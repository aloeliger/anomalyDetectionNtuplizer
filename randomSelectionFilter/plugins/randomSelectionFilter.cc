// -*- C++ -*-
//
// Package:    anomalyDetectionNtuplizer/randomSelectionFilter
// Class:      randomSelectionFilter
//
/**\class randomSelectionFilter randomSelectionFilter.cc anomalyDetectionNtuplizer/randomSelectionFilter/plugins/randomSelectionFilter.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Andrew Loeliger
//         Created:  Mon, 10 Oct 2022 17:26:08 GMT
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CLHEP/Random/RandomEngine.h"

//
// class declaration
//

class randomSelectionFilter : public edm::stream::EDFilter<> {
public:
  explicit randomSelectionFilter(const edm::ParameterSet&);
  ~randomSelectionFilter() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginStream(edm::StreamID) override;
  bool filter(edm::Event&, const edm::EventSetup&) override;
  void endStream() override;

  double reductionRate;

  //void beginRun(edm::Run const&, edm::EventSetup const&) override;
  //void endRun(edm::Run const&, edm::EventSetup const&) override;
  //void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
  //void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

  // ----------member data ---------------------------
  /*
#ifdef THIS_IS_AN_EVENT_EXAMPLE
  edm::EDGetTokenT<ExampleData> exampleToken_;
#endif
#ifdef THIS_IS_AN_EVENTSETUP_EXAMPLE
  edm::ESGetToken<SetupData, SetupRecord> setupToken_;
#endif
  */
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
randomSelectionFilter::randomSelectionFilter(const edm::ParameterSet& iConfig) {
  //now do what ever initialization is needed
  /*
#ifdef THIS_IS_AN_EVENT_EXAMPLE
  exampleToken_ = consumes<ExampleData>(iConfig.getParameter<edm::InputTag>("examples"));
#endif
#ifdef THIS_IS_EN_EVENTSETUP_EXAMPLE
  setupToken_ = esConsumes<SetupData, SetupRecord>();
#endif
  */
  reductionRate = iConfig.exists("reductionRate") ? iConfig.getParameter<double>("reductionRate"): 100.0;
}

randomSelectionFilter::~randomSelectionFilter() {
  // do anything here that needs to be done at destruction time
  // (e.g. close files, deallocate resources etc.)
  //
  // please remove this method altogether if it would be left empty
}

//
// member functions
//

// ------------ method called on each new Event  ------------
bool randomSelectionFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  /*
#ifdef THIS_IS_AN_EVENT_EXAMPLE
  ExampleData const& in = iEvent.get(exampleToken_);
#endif

#ifdef THIS_IS_AN_EVENTSETUP_EXAMPLE
  SetupData const& setup = iSetup.getData(setupToken_);
#endif
  */
  edm::Service<edm::RandomNumberGenerator> rng;
  CLHEP::HepRandomEngine& engine = rng->getEngine(iEvent.streamID());
  bool acceptance = engine.flat() <= (1.0/reductionRate);
  //std::cout<<acceptance<<std::endl;
  return acceptance;
}

// ------------ method called once each stream before processing any runs, lumis or events  ------------
void randomSelectionFilter::beginStream(edm::StreamID) {
  // please remove this method if not needed
}

// ------------ method called once each stream after processing all runs, lumis and events  ------------
void randomSelectionFilter::endStream() {
  // please remove this method if not needed
}

// ------------ method called when starting to processes a run  ------------
/*
void
randomSelectionFilter::beginRun(edm::Run const&, edm::EventSetup const&)
{ 
}
*/

// ------------ method called when ending the processing of a run  ------------
/*
void
randomSelectionFilter::endRun(edm::Run const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when starting to processes a luminosity block  ------------
/*
void
randomSelectionFilter::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a luminosity block  ------------
/*
void
randomSelectionFilter::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void randomSelectionFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}
//define this as a plug-in
DEFINE_FWK_MODULE(randomSelectionFilter);
