from anomalyDetectionNtuplizer.samples.condorSubmission import condorSubmission

from anomalyDetectionNtuplizer.samples.condorSamples.zeroBias2018 import ZeroBias2018_Even, ZeroBias2018_Odd
from anomalyDetectionNtuplizer.samples.condorSamples.VBFHToTauTau2018 import VBFHToTauTau2018
from anomalyDetectionNtuplizer.samples.condorSamples.ttHTocc2018 import ttHTocc2018
from anomalyDetectionNtuplizer.samples.condorSamples.SUSYGluGluToBBHToBB2018 import SUSYGluGluToBBHToBB2018
from anomalyDetectionNtuplizer.samples.condorSamples.DYJetsToLL2018 import DYJetsToLL2018

submission_v1 = condorSubmission(
    name='submission_v1',
    outputFileName='submission_v1.pkl',
    listOfCondorSamples=[
        ZeroBias2018_Even,
        ZeroBias2018_Odd,
        VBFHToTauTau2018,
        ttHTocc2018,
        SUSYGluGluToBBHToBB2018,
        DYJetsToLL2018,
    ]
)