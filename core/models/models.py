import torch
import torch.nn as nn
import numpy as np
       

class ActionProposalGenerator(nn.Module):
    """1D Temporal Action Proposal generator to create proposals
       for Temporal Action Localization
    """
    def __init__(self, spn, soi):
        super(ActionProposalGenerator, self).__init__()
        self.spn = spn
        self.soi = soi

    def train(self, mode=True):
        super(ActionProposalGenerator, self).train()
        self.soi.train()
        self.spn.eval()

    def forward(self, x):
        _, _, proposals, context_proposals, logits, _ = self.spn(x)

        video_scores, contrast_pairs, actioness, cas = self.soi(logits, proposals, context_proposals)
        
        return video_scores, contrast_pairs, actioness, cas