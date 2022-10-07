import torch
from torch_geometric.data import Dataset
import os
import uproot
import numpy as np
import ROOT
import math
from torch_geometric.data import Data
from tqdm import tqdm
from multiprocessing import Queue, Pool

dataQueue = Queue()

class testZBdatasetOutOfMemory(Dataset):
    def __init__(self, transform=None, pre_transform=None, pre_filter=None):
        self.NFSpath = '/nfs_scratch/aloeliger'

        self.chargedHadronsArray = uproot.lazy(list(name +':chargedHadronPFcandidateAnalyzer/chargedHadronPFcands' for name in self.raw_file_names))
        self.neutralHadronsArray = uproot.lazy(list(name+':neutralHadronPFcandidateAnalyzer/neutralHadronPFcands' for name in self.raw_file_names))
        self.muonsArray = uproot.lazy(list(name+':muonPFcandidateAnalyzer/muonPFcands' for name in self.raw_file_names))
        self.electronsArray = uproot.lazy(list(name+':electronPFcandidateAnalyzer/electronPFcands' for name in self.raw_file_names))
        self.gammasArray = uproot.lazy(list(name+':gammaPFcandidateAnalyzer/gammaPFcands' for name in self.raw_file_names))
        
        self.fourVectorPlusChargeBranches = ['ptVector','etaVector','phiVector','massVector','chargeVector']

        self.numberOfGraphs = len(uproot.lazy([self.raw_file_names[0]+':chargedHadronPFcandidateAnalyzer/chargedHadronPFcands'])['ptVector'])

        #self.graphsPerFile = 200
        #self.numberOfProcessedFiles = int(self.numberOfGraphs / self.graphsPerFile) if (self.numberOfGraphs % self.graphsPerFile == 0) else int(self.numberOfGraphs / self.graphsPerFile) + 1
        self.numberOfProcessedFiles = self.numberOfGraphs
        
        super().__init__('',transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        rawPath = self.NFSpath+'/pfAnomalyData/testZBRoot/testPFcands.root'
        return [rawPath]

    @property
    def processed_file_names(self):
        processedList = []
        for i in range(self.numberOfProcessedFiles):
            processedPath = self.NFSpath+'/pfAnomalyData/testZBtorchOutOfMemory/testPFcands_%i.pt' % i
            processedList.append(processedPath)
        return processedList
    #Create appropriate numpy node lists of features
    #TODO: we need to scale the features before they are used.
    #so, the nodes have slightly different scalings
    #maybe we should pass a flag along,
    #so that we know what values to use in the scaling

    #np.apply_along_axis(standardScaler(200, 50), 0, theArray)
    def standardScaler(self, mean, stddev):
        return lambda x: (x-mean)/stddev
    def linearScaler(self, endPoint1, endPoint2):
        length = (endPoint2-endPoint1)
        midPoint = (endPoint1+endPoint2)/2.0
        return lambda x: (x-midPoint)/length

    def createNodeList(self, nodeType, branches, event, particleType):
        if particleType == 'chargedHadron':
            etaTransformer = self.standardScaler(0.0, 1.5)
            massTransformer = self.standardScaler(0.1394, 0.003)
        elif particleType == 'neutralHadron':
            etaTransformer = self.standardScaler(0.0, 1.9)
            massTransformer = self.standardScaler(0.0, 1.1e-7)
        elif particleType == 'electron':
            etaTransformer = self.standardScaler(0.0, 1.43)
            massTransformer = self.standardScaler(1.491e-5, 0.0029)
        elif particleType == 'gamma':
            etaTransformer = self.standardScaler(0.0, 1.179)
            massTransformer = None
        elif particleType == 'muon':
            etaTransformer = self.standardScaler(0.0, 1.98)
            massTransformer = None
        
        phiTransformer = self.linearScaler(-1.0 * math.pi, math.pi)
        ptTransformer = self.linearScaler(0, 100)

        listOfInputBranches = []
        for branch in branches:
            theArray = nodeType[branch][event].to_numpy()
            if branch == 'ptVector':
                theArray = np.apply_along_axis(ptTransformer, 0, theArray)
            elif branch == 'etaVector':
                theArray = np.apply_along_axis(etaTransformer, 0, theArray)
            elif branch == 'phiVector':
                theArray = np.apply_along_axis(phiTransformer, 0, theArray)
            elif branch == 'massVector':
                if massTransformer != None:
                    theArray = np.apply_along_axis(massTransformer, 0, theArray)
                    
            listOfInputBranches.append(np.expand_dims(theArray, axis=1))
        tupleInput = tuple(listOfInputBranches)
        
        nodeList = np.concatenate(tupleInput, axis=1)
    
        return nodeList        
            
    def createNodes(self, index):
        chargedHadrons = self.createNodeList(self.chargedHadronsArray, self.fourVectorPlusChargeBranches, index, 'chargedHadron')
        neutralHadrons = self.createNodeList(self.neutralHadronsArray, self.fourVectorPlusChargeBranches, index, 'neutralHadron')
        electrons = self.createNodeList(self.electronsArray, self.fourVectorPlusChargeBranches, index, 'electron')
        gammas = self.createNodeList(self.gammasArray, self.fourVectorPlusChargeBranches, index, 'gamma')
        muons = self.createNodeList(self.muonsArray, self.fourVectorPlusChargeBranches, index, 'muon')

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
        
        #let's write this out to it's own file
        torch.save(data, self.processed_file_names[index])

        #dataQueue.put(data)
        return
    
    def process(self):
        #okay. Here's where it gets a bit gross
        #theRawFile = self.raw_file_names[0]

        print("Processing data to list...")
        print("files to be processed: ",self.numberOfProcessedFiles)
        
        workList = [i for i in range(self.numberOfProcessedFiles)]
        workPool = Pool(20)
        workPool.map(self.createGraph, workList)

    def download(self):
        return

    def len(self):
        return len(self.chargedHadronsArray)

    def get(self, idx):
        #We need to get the idx-th graph from it's file, and load
        #it into memory
        
        #first thing's first, what file number does this belong to?
        #fileNum = idx / self.graphsPerFile
        #we also need to know what entry it is in that file
        #entryNum = idx % self.graphsPerFile
        
        #let's load the file objects
        #data, slices = torch.load(self.processed_paths[fileNum])
        data = torch.load(self.processed_file_names[idx])
        
        #dataEntry = data[entryNum]
        return data
