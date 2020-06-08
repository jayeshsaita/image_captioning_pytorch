import torch
import json

class ImageCaptioningDataset(torch.utils.data.Dataset):
    def __init__(self, train_json, feature_pkl, annot_json):
        self.images = list(json.load(open(train_json)).keys())
        self.features = torch.load(feature_pkl)
        self.annots = json.load(open(annot_json))
    
    def __getitem__(self, index):
        img_name = self.images[index]
        feature = self.features[img_name]
        ci = torch.randperm(5)[0]
        annot = torch.LongTensor(self.annots[img_name][ci])
        
        return [feature, annot], [annot, torch.LongTensor(self.annots[img_name])]
    
    def __len__(self):
        return len(self.images)