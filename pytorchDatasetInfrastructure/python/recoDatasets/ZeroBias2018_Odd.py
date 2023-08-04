from anomalyDetectionNtuplizer.pytorchDatasetInfrastructure.recoDataset import recoDataset_inMemory

ZeroBias2018_Odd = recoDataset_inMemory(
    root='/hdfs/store/user/aloeliger/01Aug2023_0715_AnomalyBumpHuntNtuples/',
    name='ZeroBias2018_Odd',
)