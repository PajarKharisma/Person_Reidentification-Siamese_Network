import torch
import torch.nn.functional as F
import torch.nn as nn

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        # euclidean_distance = 1 - (1/(1 + euclidean_distance))
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) + (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive

class TripletLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        d1 = F.pairwise_distance(anchor, positive, keepdim=True)
        d2 = F.pairwise_distance(anchor, negative, keepdim=True)
        distance = d1 - d2 + self.margin 
        loss = torch.mean(torch.max(distance, torch.zeros_like(distance)))
        
        return loss

class AbsoluteLoss(torch.nn.Module):
    def __init__(self):
        pass

    def forward(self, output1, output2):
        return torch.mean(torch.abs(output1 - output2))