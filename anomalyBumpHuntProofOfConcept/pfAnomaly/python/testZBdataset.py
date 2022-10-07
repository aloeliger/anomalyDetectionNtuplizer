import torch
from torch_geometric.data import InMemoryDataset
import os
import uproot
import numpy as np
import awkward as ak
import ROOT
from torch_geometric.data import Data
import torch
from multiprocessing import Queue, Pool
import time


dataQueue = Queue()

class testZBdataset(InMemoryDataset):
    def __init__(self, root='', transform=None, pre_transform=None, pre_filter=None):
        self.NFSpath = '/nfs_scratch/aloeliger/'
        self.fourVectorPlusChargeBranches = ['ptVector','etaVector','phiVector','massVector','chargeVector']
        self.theFile = uproot.open(self.raw_file_names[0])
        self.numberOfGraphs = len(uproot.lazy([self.raw_file_names[0]+':chargedHadronPFcandidateAnalyzer/chargedHadronPFcands'])['ptVector'])
        self.graphsPerFile = 100
        self.numberOfProcessedFiles = int(self.numberOfGraphs / self.graphsPerFile) if (self.numberOfGraphs % self.graphsPerFile == 0) else int(self.numberOfGraphs / self.graphsPerFile) + 1


        super().__init__(root, transform, pre_transform, pre_filter)
        
        self.data = []
        self.slices = {}
        for i in range(self.numberOfProcessedFiles):
            fileData, fileSlices = torch.load(self.processed_paths[i])
            data.append(fileData)
            slides.update(fileSlices)

        #self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        rawPath = self.NFSpath+'pfAnomalyData/testZBRoot/testPFcands.root'
        return [rawPath]
        
    @property
    def processed_file_names(self):
        processedList = []
        for i in range(self.numberOfProcessedFiles):
            processedPath = self.NFSpath+'pfAnomalyData/testZBtorch/testPFcands_%i.pt' % i
            processedList.append(processedPath)
        return processedList

    def createNodeList(self, nodeType, branches, event):
        listOfInputBranches = []
        for branch in branches:
            listOfInputBranches.append(np.expand_dims(nodeType[branch][event].to_numpy(), axis=1))
            tupleInput = tuple(listOfInputBranches)

            nodeList = np.concatenate(tupleInput, axis=1)
    
            return nodeList        
            
    def createNodes(self, index):
        chargedHadrons = self.createNodeList(self.chargedHadronsArray, self.fourVectorPlusChargeBranches, index)
        neutralHadrons = self.createNodeList(self.neutralHadronsArray, self.fourVectorPlusChargeBranches, index)
        electrons = self.createNodeList(self.electronsArray, self.fourVectorPlusChargeBranches, index)
        gammas = self.createNodeList(self.gammasArray, self.fourVectorPlusChargeBranches, index)
        muons = self.createNodeList(self.muonsArray, self.fourVectorPlusChargeBranches, index)

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

    def createEdges(self, nodes):
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

    def createGraph(self, index):
        nodes = self.createNodes(index)
        edgePairs, edgeFeatures = self.createEdges(nodes)

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
    
    def process(self):
        #okay. Here's where it gets a bit gross
        #theRawFile = self.raw_file_names[0]

        chargedHadronTree = self.theFile['chargedHadronPFcandidateAnalyzer']['chargedHadronPFcands']
        neutralHadronTree = self.theFile['neutralHadronPFcandidateAnalyzer']['neutralHadronPFcands']
        electronTree = self.theFile['electronPFcandidateAnalyzer']['electronPFcands']
        gammaTree = self.theFile ['gammaPFcandidateAnalyzer']['gammaPFcands']
        muonTree = self.theFile ['muonPFcandidateAnalyzer']['muonPFcands']



        print("loading complete PF cand info...")
        self.chargedHadronsArray = chargedHadronTree.arrays(self.fourVectorPlusChargeBranches)
        self.neutralHadronsArray = neutralHadronTree.arrays(self.fourVectorPlusChargeBranches)
        self.electronsArray = electronTree.arrays(self.fourVectorPlusChargeBranches)
        self.gammasArray = electronTree.arrays(self.fourVectorPlusChargeBranches)
        self.muonsArray = electronTree.arrays(self.fourVectorPlusChargeBranches)
        print("done!")
        print("Processing data to list...")
        print("files to be processed: ",self.numberOfProcessedFiles)
        for i in range(self.numberOfProcessedFiles):
            dataList = []
            workList = []
            workRangeStart = i*self.graphsPerFile
            workRangeEnd = min(self.numberOfGraphs,(i+1)*self.graphsPerFile)
            for node_i in range(workRangeStart, workRangeEnd):
                #createGraph(node_i)
                workList.append(node_i)
            workList = tuple(workList)
            workPool = Pool(10)
            startTime = time.perf_counter()
            workPool.map(self.createGraph, workList)
            endTime = time.perf_counter()
            print(f"elapsed pool time {endTime - startTime} seconds")

            while not dataQueue.empty():
                dataList.append(dataQueue.get())
            workPool.close()

            if self.pre_filter is not None:
                dataList = [data for data in dataList if self.pre_filter(data)]

            if self.pre_transform is not None:
                dataList = [self.pre_transform(data) for data in dataList]

            data, slices = self.collate(dataList)
            torch.save((data, slices), self.processed_paths[i])
            del data
            del slices
            del dataList

    def download(self):
        return

        
