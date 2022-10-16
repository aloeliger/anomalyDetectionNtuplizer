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

class evenEventNumFilter : public edm::stream::EDFilter<> {
public:
  explicit evenEventNumFilter(const edm::ParameterSet&);
  ~evenEventNumFilter() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginStream(edm::StreamID) override;
  bool filter(edm::Event&, const edm::EventSetup&) override;
  void endStream() override;

};

evenEventNumFilter::evenEventNumFilter(const edm::ParameterSet& iConfig)
{

}

evenEventNumFilter::~evenEventNumFilter()
{

}

bool evenEventNumFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup){
  using namespace edm;

  unsigned int evt = iEvent.id().event();

  return evt % 2 == 0;
}

void evenEventNumFilter::beginStream(edm::StreamID) {}

void evenEventNumFilter::endStream() {}

void evenEventNumFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(evenEventNumFilter);
