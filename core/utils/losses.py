###################################################################################
### PARTS OF THIS CODE ARE ORIGINALLY PROVIDED BY THE CoLA REPOSITORY ON GITHUB ###
### https://github.com/zhang-can/CoLA                                           ###
###################################################################################

import torch
import torch.nn as nn


class ActionLoss(nn.Module):
    def __init__(self):
        super(ActionLoss, self).__init__()
        self.bce_criterion = nn.BCELoss()

    def forward(self, video_scores, label):
        label = label / torch.sum(label, dim=1, keepdim=True)
        loss = self.bce_criterion(video_scores, label)
        return loss

class SniCoLoss(nn.Module):
    def __init__(self):
        super(SniCoLoss, self).__init__()
        self.ce_criterion = nn.CrossEntropyLoss()

    def NCE(self, q, k, neg, T=0.07):
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        neg = neg.permute(0,2,1)
        neg = nn.functional.normalize(neg, dim=1)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,nck->nk', [q, neg])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= T
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        loss = self.ce_criterion(logits, labels)

        return loss

    def forward(self, contrast_pairs):

        HA_refinement = self.NCE(
            torch.mean(contrast_pairs['HA'], 1), 
            torch.mean(contrast_pairs['EA'], 1), 
            contrast_pairs['EB']
        )

        HB_refinement = self.NCE(
            torch.mean(contrast_pairs['HB'], 1), 
            torch.mean(contrast_pairs['EB'], 1), 
            contrast_pairs['EA']
        )

        loss = HA_refinement + HB_refinement
        return loss
    
class TotalLoss(nn.Module):
    def __init__(self):
        super(TotalLoss, self).__init__()
        self.factor = 0.005
        self.action_criterion = ActionLoss()
        self.snico_criterion = SniCoLoss()

    def balance(self, factor):
        self.factor = factor

    def forward(self, video_scores, label, contrast_pairs):
        loss_cls = self.action_criterion(video_scores, label)
        loss_snico = self.snico_criterion(contrast_pairs)
        loss_total = loss_cls + self.factor * loss_snico

        loss_dict = {
            'Net/Loss/Action': loss_cls,
            'Net/Loss/SniCo': loss_snico,
            'Net/Loss/Total': loss_total
        }

        return loss_total, loss_dict
    
class SPNSniCoLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_criterion = nn.CrossEntropyLoss()

    def NCE(self, q, k, neg, T=0.07):
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        neg = neg.permute(0,2,1)
        neg = nn.functional.normalize(neg, dim=1)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,nck->nk', [q, neg])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= T
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        loss = self.ce_criterion(logits, labels)

        return loss

    def forward(self, contrast_pairs):

        rec_num = len(contrast_pairs['HA'])

        HA_refinement = torch.zeros(rec_num)
        HB_refinement = torch.zeros(rec_num)

        for i in range(rec_num):
            HA_refinement[i] = self.NCE(
                torch.mean(contrast_pairs['HA'][i], 1), 
                torch.mean(contrast_pairs['EA'][i], 1), 
                contrast_pairs['EB'][i]
            )
        for i in range(rec_num):
            HB_refinement[i] = self.NCE(
                torch.mean(contrast_pairs['HB'][i], 1), 
                torch.mean(contrast_pairs['EB'][i], 1), 
                contrast_pairs['EA'][i]
            )

        loss = torch.mean(HA_refinement) + torch.mean(HB_refinement)
        return loss
    
class SPNTotalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.factor = 0.002
        self.action_criterion = ActionLoss()
        self.snico_criterion = SPNSniCoLoss()

    def balance(self, factor):
        self.factor = factor

    def forward(self, video_scores, label, contrast_pairs):
        loss_cls = self.action_criterion(video_scores, label)
        loss_snico = self.snico_criterion(contrast_pairs)
        loss_total = loss_cls + self.factor * loss_snico

        loss_dict = {
            'SPN/Loss/Action': loss_cls,
            'SPN/Loss/SniCo': loss_snico,
            'SPN/Loss/Total': loss_total
        }

        return loss_total, loss_dict