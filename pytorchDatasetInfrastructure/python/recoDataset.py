from torch_geometric.data import Data, Dataset
import os
from tqdm import trange
import torch
from anomalyDetectionNtuplizer.pytorchDatasetInfrastructure.recoGraph import recoGraph
from torch_geometric.data import InMemoryDataset
import ROOT

class recoDataset_inMemory(InMemoryDataset):
    def __init__(self, name, root, transform=None, pre_transform=None, pre_filter=None):
        self.name = name
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_dir(self):
        return os.path.join(self.root, self.name, 'raw')
    
    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        allFiles = []
        for dirPath, _, files in os.walk(self.raw_dir):
            for name in files:
                allFiles.append(os.path.join(dirPath,name))
        return allFiles
    
    @property
    def processed_file_names(self):
        return ['data.pt']
    
    def download(self):
        raise RuntimeError("Download called for recoDataset_inMemory. This should never happen. All files should exist.")
    
    def process(self):
        # process everything into a singular file to have in memory
        dataList = []
        print(f'Processing for dataset name: {self.name}...')

        theChain = ROOT.TChain('basicEventInfo/basicInfo')

        electronChain = ROOT.TChain('electronNtuplizer/Electron_info')
        jetChain = ROOT.TChain('jetNtuplizer/Jet_info')
        fatJetChain = ROOT.TChain('fatJetNtuplizer/FatJet_info')
        muonChain = ROOT.TChain('muonNtuplizer/Muon_info')
        photonChain = ROOT.TChain('photonNtuplizer/Photon_info')
        tauChain = ROOT.TChain('tauNtuplizer/Tau_info')
        boostedTauChain = ROOT.TChain('boostedTauNtuplizer/BoostedTau_info')

        print(f'Using {len(self.raw_file_names)} files...')

        nodeTypes = [
            (electronChain, 'Electron'),
            (jetChain, 'Jet'),
            (fatJetChain, 'FatJet'),
            (muonChain, 'Muon'),
            (photonChain, 'Photon'),
            (tauChain, 'Tau'),
            (boostedTauChain, 'BoostedTau')
        ]

        for fileName in self.raw_file_names:
            theChain.Add(fileName)

            electronChain.Add(fileName)
            jetChain.Add(fileName)
            fatJetChain.Add(fileName)
            muonChain.Add(fileName)
            photonChain.Add(fileName)
            tauChain.Add(fileName)
            boostedTauChain.Add(fileName)

        totalEntries = theChain.GetEntries()
        print(f'Total entries in zero bias: {totalEntries}')  
        for i in trange(totalEntries, ascii=True, leave=False, dynamic_ncols=True):
            theChain.GetEntry(i)

            electronChain.GetEntry(i)
            jetChain.GetEntry(i)
            fatJetChain.GetEntry(i)
            muonChain.GetEntry(i)
            photonChain.GetEntry(i)
            tauChain.GetEntry(i)
            boostedTauChain.GetEntry(i)    

            theGraph = recoGraph()
            theGraph.createNodeListFromChainNamePairs(nodeTypes)
            theGraph.createKNearestNeighborsEdges(5)
            
            theData = theGraph.toPytorchGeoData()
            dataList.append(theData)
        
        data, slices = self.collate(dataList)
        torch.save((data,slices), self.processed_paths[0])