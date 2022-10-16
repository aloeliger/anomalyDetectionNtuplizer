import torch
from tqdm import tqdm,trange

from anomalyDetectionNtuplizer.model.pfAnomalyGAE import pfAnomaly
from anomalyDetectionNtuplizer.pytorchDatasetInfrastructure.testDataset import testDataset

from torch_geometric.loader import DataLoader
from torch_geometric.nn import GAE

import torch_geometric

import random

import datetime

import numpy as np

def encodeGraphs(dataBatch, model, device):
    x = dataBatch.x.float().to(device)
    edge_index = dataBatch.edge_index.to(device)
    edge_attr  = dataBatch.edge_attr.to(device)

    z = model.encode(x, edge_index, edge_attr)
    loss = model.recon_loss(z, edge_index)

    return x, edge_index, edge_attr, z, loss

def train(dataBatch, model, optimizer, device):
    model.train()
    optimizer.zero_grad()

    x, edge_index, edge_attr, z, loss = encodeGraphs(dataBatch, model, device)

    loss.backward()
    optimizer.step()
    return float(loss)

def main():
    #date time to save the model under
    timeTag = datetime.datetime.now().strftime('%d%b%Y_%H%M')
    
    #set a common seed
    random.seed(1234)
    torch.manual_seed(1234)
    np.random.seed(1234)
    
    dataset = testDataset(rawFileLocation='/hdfs/store/user/aloeliger/pfAnomalyTest/raw/',
                          processedFileLocation='/hdfs/store/user/aloeliger/pfAnomalyTest/processed/')
    #dataset.shuffle()

    datasetLength = len(dataset)
    splitIndex = int(0.7*datasetLength)
    train_dataset = dataset[:splitIndex]
    val_dataset = dataset[splitIndex:]

    print("Dataset info:")
    print(f'Total length: {datasetLength}, train: {splitIndex}, val: {datasetLength - splitIndex}')

    loader = DataLoader(train_dataset,
                        batch_size=32,
                        shuffle=True)
    
    in_channels = dataset.num_features
    edge_channels = 1 #So far only the one edge feature
    hidden_channels = 6 #
    out_channels = 64

    print('GAE feature info:')
    print(f'Input features: {in_channels}, Output features: {out_channels}')
    print(f'PDNConv extra features: Edge features: {edge_channels}, Hidden channels: {hidden_channels}')
    
    epochs = 1
    model = GAE(
        pfAnomaly(
            in_channels = in_channels,
            out_channels = out_channels,
            edge_channels = edge_channels,
            hidden_channels = hidden_channels
        )
    )
    
    if (torch.cuda.is_available()):
        device = torch.device('cuda')
        print('Using cuda...')
    else:
        device = torch.device('cpu')
        print('Defaulting to cpu...')
    
    print('Converting model to device...')
    model = model.to(device)
    print('Starting the training...')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    loss = 0.0
    val_loss = 0.0
    epoch_tqdm = tqdm(range(1, epochs+1), leave=True)

    for epoch in epoch_tqdm:
        batch_tqdm = tqdm(loader, leave=True)
        for dataBatch in batch_tqdm:
            loss = train(dataBatch, model, optimizer, device)

            val_batch = torch_geometric.data.Batch()
            sampledGraphs = random.sample([x for x in range(len(val_dataset))], 32)
            val_data_list = [val_dataset[graphNum] for graphNum in sampledGraphs]
            val_batch = val_batch.from_data_list(val_data_list)

            val_x, val_edge_index, val_edge_attr, val_z, val_loss = encodeGraphs(val_batch, model, device)

            #scheduler.step(val_loss)
            val_loss = float(val_loss)

            lossStatement = f'loss: {loss} val_loss: {val_loss}'
            
            batch_tqdm.set_description(lossStatement)
        epoch_tqdm.set_description(lossStatement)

    print('Saving model...')
    torch.save(model, f'pfAnomaly_{timeTag}.pt')
    print('Done!')
    
if __name__ == '__main__':
    main()
