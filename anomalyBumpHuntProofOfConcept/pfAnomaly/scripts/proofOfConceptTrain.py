import torch
from tqdm import tqdm
from tqdm import trange

from anomalyBumpHunt.pfAnomaly.testZBdatasetOutOfMemory import testZBdatasetOutOfMemory

from anomalyBumpHunt.pfAnomaly.basicGAEmodel import PFAE
from anomalyBumpHunt.pfAnomaly.basicVGAEmodel import vPFAE
from anomalyBumpHunt.pfAnomaly.VGAEwithGATv2model import vPFAE_GATv2
from anomalyBumpHunt.pfAnomaly.VGAEwithGINEmodel import vPFAE_GINE
from anomalyBumpHunt.pfAnomaly.VGAEwithECmodel import vPFAE_EC
from anomalyBumpHunt.pfAnomaly.VGAEwithTranformerModel import vPFAE_Transformer
from anomalyBumpHunt.pfAnomaly.VGAEwithCGmodel import vPFAE_CG
from anomalyBumpHunt.pfAnomaly.VGAEwithGENmodel import vPFAE_GEN
from anomalyBumpHunt.pfAnomaly.VGAEwithPDNmodel import vPFAE_PDN
from anomalyBumpHunt.pfAnomaly.VGAEwithGeneralModel import vPFAE_General
from anomalyBumpHunt.pfAnomaly.VGAEwithSAGEmodel import vPFAE_SAGE
from anomalyBumpHunt.pfAnomaly.VGAEwithGraphConvModel import vPFAE_Graph
from anomalyBumpHunt.pfAnomaly.VGAEwithTAGmodel import vPFAE_TAG
from anomalyBumpHunt.pfAnomaly.VGAEwithEdgeConvModel import vPFAE_Edge
from anomalyBumpHunt.pfAnomaly.VGAEwithDynamicEdgeConvModel import vPFAE_DynamicEdge
from anomalyBumpHunt.pfAnomaly.VGAEwithClusterGCNmodel import vPFAE_ClusterGCN

from anomalyBumpHunt.pfAnomaly.GAEwithPDNmodel import PFAE_PDN

from torch_geometric.loader import DataLoader
from torch_geometric.nn import GAE
from torch_geometric.nn import VGAE

import torch_geometric

import random

dataset= testZBdatasetOutOfMemory()
dataset.shuffle()

datasetLength = len(dataset)
train_dataset = dataset[:int(0.7*datasetLength)]
val_dataset = dataset[int(0.7*datasetLength):]

loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#loader = DataLoader(train_dataset[:3200], batch_size=32, shuffle=True)
#loader = DataLoader(train_dataset[:160], batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=True)

out_channels=50
num_features = dataset.num_node_features
epochs = 1

#model = GAE(PFAE(num_features, out_channels)).float()
#model = VGAE(vPFAE(num_features, out_channels)).float()
#model = VGAE(vPFAE_GATv2(num_features, out_channels, edge_features=1)).float()
#model = VGAE(vPFAE_GINE(num_features, out_channels, edge_features=1)).float()
#model = VGAE(vPFAE_Transformer(num_features, out_channels, edge_features=1)).float()
#model = VGAE(vPFAE_EC(num_features, out_channels, edge_features=1)).float()
#model = VGAE(vPFAE_CG(num_features, out_channels, edge_features=1)).float()
#model = VGAE(vPFAE_GEN(num_features, out_channels)).float()
#model = VGAE(vPFAE_PDN(num_features, out_channels, edge_features=1, hidden_channels=6)).float()
#model = VGAE(vPFAE_General(num_features, out_channels, edge_features=1)).float()
#model = VGAE(vPFAE_SAGE(num_features, out_channels))
#model = VGAE(vPFAE_Graph(num_features, out_channels))
#model = VGAE(vPFAE_TAG(num_features, out_channels, k=5))
#model = VGAE(vPFAE_Edge(num_features, out_channels))
#model = VGAE(vPFAE_DynamicEdge(num_features, out_channels, k=2))
#model = VGAE(vPFAE_ClusterGCN(num_features, out_channels))
model = GAE(PFAE_PDN(num_features, out_channels, edge_features=1, hidden_channels=6)).float()

if(torch.cuda.is_available()):
    print("Using cuda...")
else:
    print('defaulting to cpu...')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=200)

def train(dataBatch):
    model.train()
    optimizer.zero_grad()
    x = dataBatch.x.float().to(device)
    edge_index = dataBatch.edge_index.to(device)
    edge_attr = dataBatch.edge_attr.to(device)
    #z = model.encode(x, edge_index)
    z = model.encode(x, edge_index, edge_attr)
    loss = model.recon_loss(z, edge_index)
    #if the model is variational
    #loss = loss+(1 / dataBatch.num_nodes)*model.kl_loss()
    loss.backward() 
    optimizer.step()
    return float(loss)

loss = 0.0
val_loss = 0.0
epoch_tqdm = tqdm(range(1, epochs+1), leave=True)
for epoch in epoch_tqdm:
    batch_tqdm = tqdm(loader, leave=True)
    for dataBatch in batch_tqdm:
        loss = train(dataBatch)
        
        val_batch = torch_geometric.data.Batch()
        sampledGraphs = random.sample([x for x in range(len(val_dataset))] , 32)
        val_data_list = [val_dataset[graphNum] for graphNum in sampledGraphs]
        val_batch = val_batch.from_data_list(val_data_list)
        #val_batch = random.choice(val_dataset)
        val_x = val_batch.x.float().to(device)
        val_edge_index = val_batch.edge_index.to(device)
        val_edge_attr = val_batch.edge_attr.to(device)
        #val_z = model.encode(val_x, val_edge_index)
        val_z = model.encode(val_x, val_edge_index, val_edge_attr)
        val_loss = model.recon_loss(val_z, val_edge_index)
        #val_loss = loss+(1 / val_batch.num_nodes)*model.kl_loss()

        scheduler.step(val_loss)
        val_loss = float(val_loss)

        lossStatement = "loss: %f val_loss: %f" % (loss,val_loss)
        
        batch_tqdm.set_description(lossStatement)
    epoch_tqdm.set_description(lossStatement)

#let's save the model we have now.
print("Saving model...")
torch.save(model, "PDNVGAE_withEdges_andNorm.pt")
print("Done!")

#Let's just take a look at the model's action on a validation graph
#random.seed(1234)
#val_graph = random.choice(val_dataset)
for graph in val_dataset:
    if graph.num_nodes <= 50:
        val_graph = graph
        break
val_x = val_graph.x.float().to(device)
val_edge_index = val_graph.edge_index.to(device)
val_edge_attr = val_graph.edge_attr.to(device)
#val_z = model.encode(val_x, val_edge_index)
val_z = model.encode(val_x, val_edge_index, val_edge_attr)
val_loss = model.recon_loss(val_z, val_edge_index)
#val_loss = val_loss+(1 / val_graph.num_nodes)*model.kl_loss()
val_loss = float(val_loss)

print("original graph")
print(val_graph)
print("val_loss: ",val_loss)

print("original graph edge index")
print(val_graph.edge_index)
print(val_graph.edge_index.shape)

print("encoding:")
print(val_z)
print(val_z.shape)
#let's do a test on all possible edge pairs
#just to see what the decoder makes of it?
#Will it be able to handle that and give us a 
#prob for each, or will it break down?

val_reconstructed = model.decode(val_z, val_edge_index)

print("reconstruction edge indices (using only original as possible!)")
print(val_reconstructed)
print(val_reconstructed.shape)


edgePairs = []
for i in range(val_graph.edge_index.shape[1]):
    pair = []
    for j in range(val_graph.edge_index.shape[0]):
        pair.append(val_graph.edge_index[j][i])
    edgePairs.append(pair)

indicesToEject = []
for i in range(val_reconstructed.shape[0]):
    if float(val_reconstructed[i]) <= 0.5:
        indicesToEject.append(i)

print("indices remaining after unlikely ones removed: ", len(edgePairs)-len(indicesToEject))
for i in sorted(indicesToEject, reverse=True):
    del edgePairs[i]

allEdgeIndex_limited = torch.tensor(edgePairs).t().contiguous()

recon_graph_limited = val_graph.clone()
recon_graph_limited.edge_index = allEdgeIndex_limited
recon_graph_limited.edge_attr = None

print("reconstructed graph using only original indices")
print(recon_graph_limited)

print("reconstruction edge indices (using all as possible!)")

numNodes = val_graph.num_nodes
edgePairs = []
for i in range(numNodes):
    for j in range(i+1,numNodes):
        edgePairs.append([i,j])
        edgePairs.append([j,i])
allEdgeIndex = torch.tensor(edgePairs).t().contiguous()

val_reconstructed = model.decode(val_z, allEdgeIndex)

print(val_reconstructed)
print(val_reconstructed.shape)
print(val_reconstructed.shape[0])
print(val_reconstructed[0])
print(float(val_reconstructed[0]))

indicesToEject = []
for i in range(len(edgePairs)):
    if float(val_reconstructed[i]) <= 0.5:
        indicesToEject.append(i)

print("indices remaining after unlikely ones removed: ", len(edgePairs)-len(indicesToEject))

for i in sorted(indicesToEject, reverse=True):
    del(edgePairs[i])

recon_graph = val_graph.clone()
recon_graph.edge_index = torch.tensor(edgePairs).t().contiguous()
recon_graph.edge_attr=None

print("new graph")
print(recon_graph)

import networkx as nx
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt

drawOptions={'node_size': 25}

print("drawing original...")
nx_val_graph = to_networkx(val_graph, to_undirected=True)
nx.draw(nx_val_graph, **drawOptions)
plt.savefig("original.png")

plt.clf()

print("drawing reconstruction...")
nx_recon_graph = to_networkx(recon_graph, to_undirected=True)
nx.draw(nx_recon_graph, **drawOptions)
plt.savefig("reconstructed.png")

plt.clf()

print("drawing limited reconstruction...")
nx_recon_graph_limited = to_networkx(recon_graph_limited, to_undirected=True)
nx.draw(nx_recon_graph_limited, **drawOptions)
plt.savefig("reconstructed_limited.png")
