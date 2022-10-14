import torch
from torch_geometric.data import Dataset
import numpy as np
import uproot
import math
from torch_geometric.data import Data
from numba import njit
from multiprocessing import Pool

import time

class testDataset(Dataset):
    
    def __init__(self, transform=None, 
                 pre_transform=None, 
                 pre_filter=None, 
                 rawFileLocation='./', 
                 processedFileLocation='./',
                 workspaceLocation='./'):
        self.rawFileLocation = rawFileLocation
        self.processedFileLocation = processedFileLocation

        self.kNearestNeighbors = 5
        self.numberOfGraphs = len(uproot.lazy([self.raw_file_names[0]+':chargedHadronPFcandidateAnalyzer/chargedHadronPFcands'])['ptVector']) #not crazy about this little hack...
    
        self.nodeTypes = ['chargedHadrons',
                          'neutralHadrons',
                          'muons',
                          'electrons',
                          'gammas']
        self.oneHotCodes = [np.array([[1,0,0,0,0]]),
                            np.array([[0,1,0,0,0]]),
                            np.array([[0,0,1,0,0]]),
                            np.array([[0,0,0,1,0]]),
                            np.array([[0,0,0,0,1]])]
        self.branches = [
            'ptVector',
            'etaVector',
            'phiVector',
            'etVector',
            'chargeVector',
            'mtVector',
            'vxVector',
            'vyVector',
            'vzVector',
            'dxyVector',
            'dzVector'
        ]
    
        super().__init__('', transform, pre_transform, pre_filter)
    
    @property
    def raw_file_names(self):
        rawPath = self.rawFileLocation+'testPFcandsNew.root'
        return [rawPath]

    @property
    def processed_file_names(self):
        processedList = []
        for i in range(self.numberOfGraphs):
            processedPath = self.processedFileLocation+f'testGraph_{i}.pt'
            processedList.append(processedPath)
        return processedList


    @staticmethod
    def createNodeList(lazyNodeList, branches, event):
        #start_time = time.perf_counter()
        listOfInputBranches = []
        for branch in branches:
            #start_branch_numpy_time = time.perf_counter()
            theArray = lazyNodeList[branch][event].to_numpy()
            #theArray = self.fastNumpyConversion(lazyNodeList[branch][event])
            #end_branch_numpy_time = time.perf_counter()
            #print(f'branch numpification time: {end_branch_numpy_time - start_branch_numpy_time} seconds')
            listOfInputBranches.append(np.expand_dims(theArray, axis=1))
        tupleInput = tuple(listOfInputBranches)
        nodeList = np.concatenate(tupleInput, axis=1)
        #end_time = time.perf_counter()
        #print(f'individual node list creation time: {end_time - start_time} seconds')
        return nodeList

    def createNodes(self, index):
        completedNodes = []
        #start_time = time.perf_counter()
        for nodeType, code in zip(self.nodeTypes, self.oneHotCodes):
            filePathName = nodeType[:-1] #name in the various paths. Generally just the same thing without an "s". this is a little sketchy
            theLazyList = uproot.lazy(list(name+':'+filePathName+'PFcandidateAnalyzer/'+filePathName+'PFcands' for name in self.raw_file_names))
            theNumpyList = self.createNodeList(theLazyList, self.branches, index)
            theNumpyList = np.concatenate(
                (theNumpyList, np.zeros((theNumpyList.shape[0],5)) + code),
                axis=1
            )
            completedNodes.append(theNumpyList)
        #end_time = time.perf_counter()
        #print(f'node list creation time: {end_time - start_time} seconds')
        completedNodes = tuple(completedNodes)
        nodes = np.concatenate(
            completedNodes,
            axis=0
        )
        return nodes
        
    @staticmethod
    @njit
    def createEdges(listOfFourVectors, kNearestNeighbors):
        edgePairs = []
        edgeFeatures = []
        for  i in range(len(listOfFourVectors)):
            firstVector = listOfFourVectors[i]
            listOfClosestVectors = []
            for j in range(len(listOfFourVectors)):
                if i==j:
                    continue
                secondVector = listOfFourVectors[j]
                #deltaR = firstVector.DeltaR(secondVector)
                #I should check my math on this...
                deltaEta = firstVector[1]-secondVector[1]
                deltaPhi = abs(firstVector[2]-secondVector[2])
                if deltaPhi > 3.14:
                    deltaPhi -= 3.14
                deltaR = math.sqrt(deltaEta**2+deltaPhi**2)
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
        #edgePairs, edgeFeatures = self.createEdges(nodes)
        edgePairs, edgeFeatures = self.createEdges(nodes, self.kNearestNeighbors)

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
        #return data
        
    def process(self):
        print("Creating processed dataset")
        print(f"Files to be processed: {self.numberOfGraphs}")

        start_time = time.perf_counter()
        workList = [i for i in range(self.numberOfGraphs)]
        workPool = Pool(20)
        workPool.map(self.createGraph, workList)
        end_time = time.perf_counter()
        print(f'Time spent in pool: f{end_time - start_time} seconds')

        #for i in trange(self.numberOfGraphs):
        #    data = self.createGraph(i)
        #    torch.save(data, self.processed_file_names[i])
        #    end_time = time.perf_counter()

    def download(self):
        return
    
    def len(self):
        return self.numberOfGraphs

    def get(self, idx):
        data = torch.load(self.processed_file_names[idx])
        
        return data
