from anomalyDetectionNtuplizer.pytorchDatasetInfrastructure.testDataset import testDataset
import time
import random

dataset = testDataset(rawFileLocation='/nfs_scratch/aloeliger/pfAnomalyTest/raw/',
                      processedFileLocation='/nfs_scratch/aloeliger/pfAnomalyTest/processed/')

length = len(dataset)
print(length)
print(dataset.num_node_features)
start_time = time.perf_counter()
print(dataset[0])
end_time = time.perf_counter()
print(f'load of element zero took {end_time-start_time} seconds')

randomElement = random.randint(0,length-1)
start_time = time.perf_counter()
print(dataset[randomElement])
end_time = time.perf_counter()
print(f'load of random element took {end_time-start_time} seconds')
print("shuffling...")
dataset.shuffle()
print("shuffle done!")
