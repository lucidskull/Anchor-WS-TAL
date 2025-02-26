from typing import Dict, List
import math

import torch
import torch.nn as nn
import numpy as np
from scipy import ndimage

from ..config import cfg


class SoIPool(nn.Module):
    """SoI Pooling layer along temporal dimension"""
    def __init__(self, output_size, receptive_fields=cfg.receptive_fields):
        super().__init__()
        self.output_size = output_size
        self.r_fields = receptive_fields

    def forward(self, logit_list):
        # logits: list of [batch size, number of segments, segment length, feature dimension] for every segment length
        num_fields = len(logit_list)
        batch = logit_list[0].shape[0]
        num_seg = logit_list[0].shape[1]
        feature_dim = logit_list[0].shape[3]
        out_size = self.output_size

        logits = torch.empty(
            num_fields,
            batch,
            num_seg,
            out_size
        ).cuda()
        for i, rec_logits in enumerate(logit_list):
            # rec_logits: [batch size, number of segments, segment length, feature dimension]
            rec_logits = rec_logits.flatten(start_dim=2)
            # rec_logits: [batch size, number of segments, representation size]

            # padding
            rep_len = rec_logits.shape[2]
            padding = out_size - (rep_len % out_size)
            rep_len = rep_len + padding
            padding = torch.zeros(batch, num_seg, padding).cuda()
            rec_logits = torch.cat([rec_logits, padding], dim=2)

            # max pool
            rec_logits = rec_logits.reshape(
                batch,
                num_seg,
                out_size,
                rep_len // out_size
            )
            rec_logits = rec_logits.max(dim=3)
            logits[i], _ = rec_logits
        # logits: [number of receptive fields, batch size, number of segments, representation size]

        return logits

class SCNNPredictor(nn.Module):
    def __init__(self, in_channels, num_classes=len(cfg.classes)):
        super().__init__()

        self.class_score = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.Dropout(p=0.2),
            nn.Linear(in_channels, num_classes)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        scores = self.class_score(x)
        scores = self.relu(scores)
        return scores
    

class SCNNHead(nn.Module):
    def __init__(self, in_channels, representation_size):
        super().__init__()
        self.in_channels = in_channels
        self.representation_size = representation_size

        self.conv = nn.Conv1d(in_channels, representation_size, 1, padding='same')
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: [number of receptive fields, batch size, number of segments, representation size]
        rec_num = x.shape[0]
        b_size = x.shape[1]
        num_seg = x.shape[2]
        x = x.flatten(start_dim=0, end_dim=2).unsqueeze(-1)
        x = self.conv(x)
        x = self.relu(x)
        x = x.squeeze()
        x = x.reshape([rec_num, b_size, num_seg, self.representation_size])
        # x: [number of receptive fields, batch size, number of segments, new representation size]
        return x


class TemporalSoINetwork(nn.Module):
    """
    
    """
    def __init__(self, in_channels, soi_len, num_classes=len(cfg.classes), receptive_fields=cfg.receptive_fields):
        super().__init__()
        self.num_classes = num_classes
        self.threshold = 0.5
        self.receptive_fields = torch.Tensor(receptive_fields).to(torch.int32)

        self.M = 6
        self.m = 3

        self.soi_len = soi_len * (in_channels // 4)
        self.in_channels = in_channels
        self.representation_size = in_channels // 4

        self.soi_pool = SoIPool(self.soi_len)
        self.embed = SCNNHead(self.soi_len, self.representation_size)
        self.predictor = SCNNPredictor(self.representation_size)

        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=0.5)

    def pad_logits(self, logits, offset):
        # logits: [number of receptive fields, batch size, video length, feature dimension]
        padding = torch.zeros([
            logits.shape[0],
            logits.shape[1],
            offset,
            logits.shape[3]
        ]).cuda()
        padded_logits = torch.cat([padding, logits, padding], dim=2)
        # padded_logits: [number of receptive fields, batch size, offset+video length+offset, feature dimension]
        
        return padded_logits
    
    def extract_logits(self, logits, anchors, anchor_sizes):
        # Pad logits to ensure indices are within bounds
        padding = torch.max(torch.floor(anchor_sizes / 2).to(torch.int32))
        padded_logits = self.pad_logits(logits, padding)

        batch = logits.shape[1]
        time = logits.shape[2]
        feature_dim = logits.shape[3]

        anchored_data = []

        for i, field_len in enumerate(anchor_sizes):
            start_idx = anchors[i,:,:,0] + padding  # Shape: [batch, time]
            # Create a range for each time step
            # Actually we dont care here about the end timestep, we fully index it all. 
            # we add 0 to max_field_length values to start index. 
            time_offsets = torch.arange(field_len, device=logits.device).view(1, 1, -1)  # Shape: [1, 1, field_length]

            # Compute the actual indices
            temporal_indices = start_idx.unsqueeze(-1) + time_offsets  # Shape: [batch, time, field_length]

            expanded_indices = temporal_indices.unsqueeze(-1).expand(-1, -1, -1, feature_dim)

            # Gather along the time dimension (dim=2)
            anchored_data.insert(i, torch.gather(
                padded_logits[i].unsqueeze(2).expand(-1, -1, field_len, -1),  # Expand data to match the indices shape
                dim=1,  # Gather along time dimension
                index=expanded_indices
            ))  # Output shape: [batch, time, field_length, feature_dim]
        return anchored_data

    def preprocess(self, features, proposals):
        # split features into segments based on proposals
        self.time = features[0].shape[1]
        features = self.extract_logits(features, proposals, self.receptive_fields*2)
        return features
    
    def gen_cas(self, segments, scores):
        # init cas and counter to keep track of sum and contributions
        num_fields = segments.shape[0]
        batch = segments.shape[1]
        num_seg = segments.shape[2]

        cas = torch.zeros(batch, self.time, self.num_classes).cuda()
        counter = torch.zeros(batch, self.time).cuda()

        for i in range(num_fields):
            for j in range(batch):
                for l in range(num_seg):
                    segment = segments[i,j,l]
                    start = segment[0]
                    end = segment[1]
                    cas[j,start:end] = cas[j,start:end].add(scores[i,j,l])
                    counter[j,start:end] = counter[j,start:end].add(1)
        counter[counter == 0] = 1
        cas = cas / counter.unsqueeze(-1)
        actioness = cas.sum(dim=2)

        return cas, actioness

    def video_labels(self, cas, k):
        sorted_scores, _= cas.sort(descending=True, dim=1)
        topk_scores = sorted_scores[:, :k, :]
        video_scores = self.softmax(topk_scores.mean(1))
        return video_scores
    
    def select_topk_embeddings(self, scores, anchors, embeddings, k):
        batch = scores.shape[0]
        time = scores.shape[1]
        feature_dim = embeddings.shape[2]
        _, idx_DESC = scores.sort(descending=True, dim=1)

        # top k indicies for every element in batch
        idx_topk = idx_DESC[:, :k]
        # temporal locations of topk snippets
        idx_topk = torch.arange(time).cuda().expand(batch, time).gather(1, idx_topk)

        # locations of snippets
        loc = (anchors[:,:,1] - anchors[:,:,0]) // 2

        indices = torch.empty(batch, idx_topk.shape[1], dtype=torch.int32).cuda()
        for i in range(batch):
            condition = loc[i].unsqueeze(1) == idx_topk[i]
            idx = condition.nonzero()[:,0]
            # fill randomly if not enough
            if idx.shape[0] < idx_topk.shape[1]:
                idx = torch.cat([idx, torch.randint(0,loc.shape[1],[idx_topk.shape[1] - idx.shape[0]]).cuda()], 0)
            if idx.shape[0] > idx_topk.shape[1]:
                idx = idx.view(-1)[torch.randperm(idx.numel())[:idx_topk.shape[1]]]
            indices[i] = idx
        indices = indices.to(torch.int64)

        selected_embeddings = embeddings.gather(1, indices.unsqueeze(-1).expand(indices.shape[0], indices.shape[1], feature_dim))

        return selected_embeddings

    def easy_snippets_mining(self, actionness, anchors, embeddings, k_easy):
        select_idx = torch.ones_like(actionness).cuda()
        select_idx = self.dropout(select_idx)

        actionness_drop = actionness * select_idx

        actionness_rev = torch.max(actionness, dim=1, keepdim=True)[0] - actionness
        actionness_rev_drop = actionness_rev * select_idx

        easy_act = self.select_topk_embeddings(actionness_drop, anchors, embeddings, k_easy)
        easy_bac = self.select_topk_embeddings(actionness_rev_drop, anchors, embeddings, k_easy)

        return easy_act, easy_bac

    def hard_snippets_mining(self, actionness, anchors, embeddings, k_hard):
        aness_np = actionness.cpu().detach().numpy()
        aness_median = np.median(aness_np, 1, keepdims=True)
        aness_bin = np.where(aness_np > aness_median, 1.0, 0.0)

        # erosion masks to identify hard action regions at boundaries
        erosion_M = ndimage.binary_erosion(aness_bin, structure=np.ones((1,self.M))).astype(aness_np.dtype)
        erosion_m = ndimage.binary_erosion(aness_bin, structure=np.ones((1,self.m))).astype(aness_np.dtype)
        idx_region_inner = actionness.new_tensor(erosion_m - erosion_M)
        aness_region_inner = actionness * idx_region_inner
        hard_act = self.select_topk_embeddings(aness_region_inner, anchors, embeddings, k_hard)

        # erosion masks to identify hard background regions at boundaries
        dilation_m = ndimage.binary_dilation(aness_bin, structure=np.ones((1,self.m))).astype(aness_np.dtype)
        dilation_M = ndimage.binary_dilation(aness_bin, structure=np.ones((1,self.M))).astype(aness_np.dtype)
        idx_region_outer = actionness.new_tensor(dilation_M - dilation_m)
        aness_region_outer = actionness * idx_region_outer
        hard_bac = self.select_topk_embeddings(aness_region_outer, anchors, embeddings, k_hard)

        return hard_act, hard_bac
    
    def snippet_mining(self, actioness, anchors, embeddings, k_easy, k_hard):
        easy_act, easy_bac = self.easy_snippets_mining(actioness, anchors, embeddings, k_easy)
        hard_act, hard_bac = self.hard_snippets_mining(actioness, anchors, embeddings, k_hard)
        
        return easy_act, easy_bac, hard_act, hard_bac

    def forward(self, features, proposals, context_proposals):
        features = self.preprocess(features, context_proposals)
        features = self.soi_pool(features)
        # features: [number of receptive fields, batch size, number of segments, representation size]
        logits = self.embed(features)
        # logits: [number of receptive fields, batch size, number of segments, new representation size]
        scores = self.predictor(logits)
        # scores: [number of receptive fields, batch size, number of segments, number of classes]
        cas, actioness = self.gen_cas(proposals, scores)

        # receptive field difference does not matter from this point, therefore treat all the same
        logits = logits.permute(1,0,2,3).flatten(start_dim=1, end_dim=2)
        proposals = proposals.permute(1,0,2,3).flatten(start_dim=1, end_dim=2)
        scores = scores.permute(1,0,2,3).flatten(start_dim=1, end_dim=2)
        # [batch size, all segments, x]
        self.k_easy = logits.shape[1] // 5
        self.k_hard = logits.shape[1] // 10

        video_scores = self.video_labels(cas, self.k_easy)

        if self.training:
            easy_act, easy_bac, hard_act, hard_bac = self.snippet_mining(
                actioness,
                proposals,
                logits,
                self.k_easy,
                self.k_hard
            )
            contrast_pairs = {
                'EA': easy_act,
                'EB': easy_bac,
                'HA': hard_act,
                'HB': hard_bac
            }

        return video_scores, contrast_pairs, actioness, cas