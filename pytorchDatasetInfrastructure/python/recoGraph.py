import ROOT
from torch_geometric.data import Data
import torch
import math

import networkx
import matplotlib.pyplot as plt

from anomalyDetectionNtuplizer.pytorchDatasetInfrastructure.recoNode import recoNode

class recoGraph():
    def __init__(self, nodeList = None, edgePairs = None):
        self.nodeList = nodeList
        self.edgePairs = edgePairs
    
    def createNodeListFromChainNamePair(self, chain, name):
        return recoNode.createListFromChain(chain, name)

    def createNodeListFromChainNamePairs(self, chainNamePairs):
        theList = []
        for chain, name in chainNamePairs:
            theList+=self.createNodeListFromChainNamePair(chain, name)
        self.nodeList = theList
    
    def naiveKNearestNeighborsEdges(self, listOfNodes, kNearestNeighbors):
        # final output will be a list of edge pairs
        # in start node, end node tuple pairs
        edgePairs = []
        for i in range(len(listOfNodes)):
            # get a vector to compare
            firstNodeVector = listOfNodes[i]
            distances = []
            for j in range(len(listOfNodes)):
                # check all other vectors, except the duplicate
                if i == j:
                    continue
                secondNodeVector = listOfNodes[j]
                # Distance is delta R, we store all these distances,
                # And the node the distance is to
                distance = firstNodeVector.DeltaR(secondNodeVector)
                distances.append((distance, j))
            # Sort the distances node pairs based on the delta R
            distances.sort(key=lambda x: x[0])
            # Get the k smallest distances, or the whole list if that is smaller than k
            actualNeighbors = min(len(distances), kNearestNeighbors)
            distances = distances[:actualNeighbors]
            # now we create the pairs
            for _, j in distances:
                edgePairs.append([i,j])
                edgePairs.append([j,i])
        # Now let's remove duplicates
        finalPairs = []
        for pair in edgePairs:
            if pair not in finalPairs:
                finalPairs.append(pair)
        return finalPairs

    def createKNearestNeighborsEdges(self, kNearestNeighbors):
        self.edgePairs = self.naiveKNearestNeighborsEdges(self.nodeList, kNearestNeighbors)

    def toNXGraph(self):
        colors = ['red','orange','yellow','blue','green','purple','pink']

        nodeList = []
        for nodeNum, node in enumerate(self.nodeList):
            nodeList.append(
                (nodeNum, {'color': f'{colors[node.nodeType]}'})
            )
        theGraph = nx.Graph()
        theGraph.add_nodes_from(nodeList)
        theGraph.add_edges_from(self.edgePairs)
        return theGraph
    
    def drawSelf(self):
        theGraph = self.toNXGraph()
        colors = ['red','orange','yellow','blue','green','purple','pink']
        labels = ['Electron','Jet','Fat Jet','Muon','Photon','Tau','Boosted tau']

        colorList = []
        labelList = {}
        posList = {}
        for nodeIndex, node in enumerate(self.nodeList):
            colorList.append(colors[node.nodeType])
            # labelList.append(labels[node.nodeType])
            labelList[nodeIndex] = labels[node.nodeType]
            posList[nodeIndex] = [
                (node.lorentzVector.Eta()+5.0)/(10.0),
                (node.lorentzVector.Phi()+math.pi)/(2.0*math.pi),
            ]

        nx.draw_networkx(
            theGraph,
            node_color=colorList,
            labels=labelList,
            pos=posList,
            node_size=150,
            font_weight='bold',
            font_size=10,
        )

    def toPytorchGeoData(self):
        nodeFeatures = torch.tensor([x.getFeatureVector() for x in self.nodeList])
        edgePairTensor = torch.tensor(self.edgePairs).t().contiguous()

        theGraph = Data(x=nodeFeatures, edge_index=edgePairTensor)
        return theGraph