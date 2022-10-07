import uproot
import numpy as np
import awkward as ak
import ROOT
from torch_geometric.data import Data
import torch
from multiprocessing import Queue, Pool, Lock
import time
from torch.utils.data import random_split
from tqdm import tqdm

def createNodeList(nodeType, branches, event):
    listOfInputBranches = []
    for branch in branches:
        listOfInputBranches.append(np.expand_dims(nodeType[branch][event].to_numpy(), axis=1))
    tupleInput = tuple(listOfInputBranches)

    nodeList = np.concatenate(tupleInput, axis=1)
    
    return nodeList
    
dataList = []
with uproot.open('../testPFcands.root') as theFile:
    basicInfoTree = theFile['basicEventInfo']['basicInfo']

    basicInfo = basicInfoTree.arrays(['run','lumi','evt'], library='np')
    
    chargedHadronTree = theFile['chargedHadronPFcandidateAnalyzer']['chargedHadronPFcands']
    neutralHadronTree = theFile['neutralHadronPFcandidateAnalyzer']['neutralHadronPFcands']
    electronTree = theFile['electronPFcandidateAnalyzer']['electronPFcands']
    gammaTree = theFile ['gammaPFcandidateAnalyzer']['gammaPFcands']
    muonTree = theFile ['muonPFcandidateAnalyzer']['muonPFcands']

    fourVectorPlusChargeBranches = ['ptVector','etaVector','phiVector','massVector','chargeVector']
    
    print("loading complete PF cand info...")
    chargedHadronsArray = chargedHadronTree.arrays(fourVectorPlusChargeBranches)
    neutralHadronsArray = neutralHadronTree.arrays(fourVectorPlusChargeBranches)
    electronsArray = electronTree.arrays(fourVectorPlusChargeBranches)
    gammasArray = electronTree.arrays(fourVectorPlusChargeBranches)
    muonsArray = electronTree.arrays(fourVectorPlusChargeBranches)
    print("done!")
    #for node_i in tqdm(range(len(basicInfo['evt']))):

    #translate the awkward arrays read from uproot for the event into numpy
    #nodeLock = Lock()
    def createNodes(index):
        #nodeLock.acquire()
        chargedHadrons = createNodeList(chargedHadronsArray, fourVectorPlusChargeBranches, index)
        neutralHadrons = createNodeList(neutralHadronsArray, fourVectorPlusChargeBranches, index)
        electrons = createNodeList(electronsArray, fourVectorPlusChargeBranches, index)
        gammas = createNodeList(gammasArray, fourVectorPlusChargeBranches, index)
        muons = createNodeList(muonsArray, fourVectorPlusChargeBranches, index)
        #nodeLock.release()

        #one hot encode pfcand type into the node features.
        chargedHadrons = np.concatenate((chargedHadrons,
                                         np.zeros(chargedHadrons.shape) + np.array([1,0,0,0,0])),
                                        axis=1)
        neutralHadrons = np.concatenate((neutralHadrons,
                                         np.zeros(neutralHadrons.shape) + np.array([0,1,0,0,0])),
                                        axis=1)
        electrons = np.concatenate((electrons,
                                    np.zeros(electrons.shape) + np.array([0,0,1,0,0])),
                                   axis=1)
        gammas = np.concatenate((gammas,
                                 np.zeros(gammas.shape) + np.array([0,0,0,1,0])),
                                axis=1)
        muons = np.concatenate((muons,
                                np.zeros(muons.shape) + np.array([0,0,0,0,1])),
                               axis=1)

        #let's concatenate this into a final list of nodes
        nodes = np.concatenate((chargedHadrons,
                                neutralHadrons,
                                electrons,
                                gammas,
                                muons),
                               axis=0)
        return nodes
        #print(nodes.shape)
        #create a list of four vectors for each node in the list
    def createEdges(nodes):
        listOfFourVectors = []
        for i in range(len(nodes)):
            newVector = ROOT.TLorentzVector()
            newVector.SetPtEtaPhiM(nodes[i][0],
                                   nodes[i][1],
                                   nodes[i][2],
                                   nodes[i][3])
            listOfFourVectors.append(newVector)
        #Now we need to go through and extract a series of edge connections from this.
        #this is going to depend on what the closest neighbors are, and how many of them
        #we are expecting.
        #how do we do this?
        #We start with a 4 vector picked from our list.
        #from there, we scroll through the list of all four vectors
        # (ignoring the current one)
        #and with each one,
        #if we have less than five distances, we insert a pair of this distance,
        #and corresponding edge pairs to a list, at an appropriate place
        kNearestNeighbors = 5

        edgePairs = []
        edgeFeatures = []
        for  i in range(len(listOfFourVectors)):
            firstVector = listOfFourVectors[i]
            listOfClosestVectors = []
            for j in range(len(listOfFourVectors)):
                if i==j:
                    continue
                secondVector = listOfFourVectors[j]
                deltaR = firstVector.DeltaR(secondVector)
                deltaRTuple = (deltaR, j)
                if len(listOfClosestVectors) == 0:
                    listOfClosestVectors.append(deltaRTuple)
                else: #find an appropriate index
                    bestIndex = 0
                    for k in range(len(listOfClosestVectors)):
                        if deltaRTuple[0] < listOfClosestVectors[k][0]:
                            break
                        bestIndex+=1
                    listOfClosestVectors.insert(bestIndex, deltaRTuple)
                    if len(listOfClosestVectors) > kNearestNeighbors:
                        listOfClosestVectors.pop()
            #Okay, we should now have a list of indices for this vector
            #let's construct the edge pairs and edge features
            for j in range(len(listOfClosestVectors)):
                firstEdgePair = [i, listOfClosestVectors[j][1]]
                secondEdgePair = [listOfClosestVectors[j][1], i]
                #let's figure out if our edge pairs are already there
                #added from a previous node
                if firstEdgePair not in edgePairs and secondEdgePair not in edgePairs:
                    edgePairs.append(firstEdgePair)
                    edgePairs.append(secondEdgePair)
                    edgeFeatures.append([listOfClosestVectors[j][0]]) #add an entry for the edge feature
                    edgeFeatures.append([listOfClosestVectors[j][0]])
                elif firstEdgePair in edgePairs and secondEdgePair in edgePairs:
                    continue
                else:
                    raise RuntimeError("Directed edge found in the graph!")
        return edgePairs, edgeFeatures

    dataQueue = Queue()

    def createGraph(index):
        nodes = createNodes(index)
        edgePairs, edgeFeatures = createEdges(nodes)

        #now we should scale the physics features of 
        #Okay At this stage, for this event, we have a series of nodes
        #and then the edges between them
        #with their features.
        #Can this be added to a graph reasonably?        
        #compress down to a data/graph, and see if we get any issues.
        nodes = torch.from_numpy(nodes)
        edgePairs = torch.tensor(edgePairs).t().contiguous()
        edgeFeatures = torch.tensor(edgeFeatures)

        data = Data(x=nodes, edge_index=edgePairs, edge_attr=edgeFeatures)
        dataQueue.put(data)
        return

    print("Processing data to list...")
    workList = []
    for node_i in range(300):
        #createGraph(node_i)
        workList.append(node_i)
    workList = tuple(workList)
    workPool = Pool(10)
    startTime = time.perf_counter()
    workPool.map(createGraph, workList)
    endTime = time.perf_counter()
    print(f"elapsed pool time {endTime - startTime} seconds")

    while not dataQueue.empty():
        dataList.append(dataQueue.get())
    #print("Final graphs: ",len(dataList))
    workPool.close()

from torch_geometric.loader import DataLoader
print("making data loader")
trainList, testList = random_split(dataList, [int(0.3*len(dataList)),int(0.7*len(dataList))], generator=torch.Generator().manual_seed(42))
print(len(trainList))
print(len(testList))
loader = DataLoader(trainList, batch_size=32, shuffle=True)

#can we train a GAE on this?
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GAE

class PFAE(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PFAE, self).__init__()
        self.conv1 = GCNConv(in_channels, 2*out_channels)
        self.conv2 = GCNConv(2*out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

out_channels=4
num_features = dataList[0].num_features
epochs = 100

model = GAE(PFAE(num_features, out_channels)).float()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train(dataBatch):
    model.train()
    optimizer.zero_grad()
    #avgLoss=0.0
    #for data in dataBatch:
        #print(data)
        #x = data.x.to(device)
        #x = data['x'].to(device)
        #train_pos_edge_index = data.train_pos_edge_index.to(device)
        
        #z = model.encode(x, train_pos_edge_index)
        #loss =  model.recon_loss(z, train_pos_edge_index)
        #loss.backward()
        #optimizer.step()
        #avgLoss += float(loss)
    x = dataBatch.x.float().to(device)
    #print(x)
    #print(len(x))
    edge_index = dataBatch.edge_index.to(device)
    #print(edge_index)
    #print(len(edge_index))
    z = model.encode(x, edge_index)
    loss = model.recon_loss(z, edge_index)
    loss.backward()
    optimizer.step()
    return float(loss)
    #print(dataBatch)
    #train_pos_edge_index 
    #avgLoss = avgLoss / len(dataBatch)
    #return avgLoss

loss = 0.0
for epoch in tqdm(range(1, epochs+1), leave=True, postfix = "loss: %f" % loss):
    for dataBatch in tqdm(loader, leave=False, postfix = "loss: %f" % loss):
        loss = train(dataBatch)
