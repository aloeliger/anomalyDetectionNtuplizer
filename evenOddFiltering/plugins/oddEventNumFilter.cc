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

class oddEventNumFilter : public edm::stream::EDFilter<> {
public:
  explicit oddEventNumFilter(const edm::ParameterSet&);
  ~oddEventNumFilter() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginStream(edm::StreamID) override;
  bool filter(edm::Event&, const edm::EventSetup&) override;
  void endStream() override;

};

oddEventNumFilter::oddEventNumFilter(const edm::ParameterSet& iConfig)
{

}

oddEventNumFilter::~oddEventNumFilter()
{

}

bool oddEventNumFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup){
  using namespace edm;

  unsigned int evt = iEvent.id().event();

  return evt % 2 == 1;
}

void oddEventNumFilter::beginStream(edm::StreamID) {}

void oddEventNumFilter::endStream() {}

void oddEventNumFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(oddEventNumFilter);
