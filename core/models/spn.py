import torch
import torch.nn as nn

from ..config import cfg

class TemporalSegmentProposalNetwork(nn.Module):
    """ Generator network for 1D temporal video segment proposals

        Proposals contain highest and lowest scoring segments to
        represent action and background respectively
    """

    def __init__(
            self,
            feature_dim=cfg.feature_dim,
            receptive_fields=cfg.receptive_fields,
            num_classes = len(cfg.classes)):
        
        super(TemporalSegmentProposalNetwork, self).__init__()
        self.receptive_fields = torch.Tensor(receptive_fields).to(torch.int32)
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        self.temp_conv_nets = nn.ModuleList([
            self.gen_conv_nets(size, feature_dim) for size in receptive_fields
        ])
        self.classifiers = nn.ModuleList([
            self.gen_classifiers(size*feature_dim*2) for size in receptive_fields
        ])

    def gen_conv_nets(self, size, feature_dim):
        net = nn.Sequential(
            nn.MaxPool1d((size//6)*2+1, stride=1, padding=(size//6)),
            nn.Conv1d(
                feature_dim,
                feature_dim,
                kernel_size=3,
                dilation=(size//6)*2+1,
                padding='same'
            ),
            nn.Conv1d(
                feature_dim,
                feature_dim,
                kernel_size=3,
                dilation=(size//6)*2*2+1,
                padding='same'
            )
        )
        return net
    
    def gen_classifiers(self, in_size, intermediate_size=None):
        if not intermediate_size:
            intermediate_size = in_size
        net = nn.Sequential(
            nn.Conv1d(in_size, self.num_classes, 1),
            nn.ReLU()
        )
        return net
    
    def gen_anchors(self, x, anchor_sizes):
        # x: [batch, time, feature dimension]
        num_fields = len(anchor_sizes)
        batch = x.shape[0]
        time = x.shape[1]

        start_offset = (torch.ceil(anchor_sizes / 2) - 1).to(torch.int32)
        end_offset = torch.floor(anchor_sizes / 2).to(torch.int32)

        anchors = torch.empty([
            num_fields,
            time,
            2
            ],
            dtype=torch.int32
        )

        # generate anchors considering offset
        for i in range(num_fields):
            field_anchors = torch.arange(0,time)
            field_anchors = field_anchors.unsqueeze(-1).expand([time, 2])
            field_anchors = torch.cat([
                (field_anchors[:,0] - start_offset[i]).unsqueeze(-1),
                (field_anchors[:,1] + end_offset[i]).unsqueeze(-1)
                ],
                dim=1
            )
            anchors[i] = field_anchors

        # expand for entire batch
        anchors = anchors.cuda()
        anchors = anchors.expand([
            batch,
            num_fields,
            time,
            2
        ])

        anchors = anchors.permute(1,0,2,3)

        return anchors
    
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
            time_offsets = torch.arange(field_len, device=logits.device).view(1, 1, -1)  # Shape: [1, 1, field length]

            # Compute the actual indices
            temporal_indices = start_idx.unsqueeze(-1) + time_offsets  # Shape: [batch, time, field length]

            expanded_indices = temporal_indices.unsqueeze(-1).expand(-1, -1, -1, feature_dim)

            # Gather along the time dimension
            anchored_data.insert(i, torch.gather(
                padded_logits[i].unsqueeze(2).expand(-1, -1, field_len, -1),  # Expand data to match the indices shape
                dim=1,  # Gather along time dimension
                index=expanded_indices
            ))  # anchored data: list of [batch, time, field length, feature dimension] for each receptive field

        return anchored_data


    def filter_proposals(self, anchors, context_anchors, scores):
        """ keep k proposals with highest action confidence (action)
            and k proposals with lowest action confidence (background)
        """
        # anchors: [receptive fields, batch, time, 2]
        # scores: [receptive fields, batch, time, classes]
        actioness = torch.sum(scores, dim=3)
        # actioness: [receptive fields, batch, time]
        
        # sort by actioness
        idx = torch.argsort(actioness, dim=2, descending=True)
        anchors = torch.gather(anchors, 2, idx.unsqueeze(-1).expand(anchors.shape))
        context_anchors = torch.gather(context_anchors, 2, idx.unsqueeze(-1).expand(context_anchors.shape))
        scores = torch.gather(scores, 2, idx.unsqueeze(-1).expand(scores.shape))
        
        # filter
        size = anchors.shape
        filtered_anchors = torch.cat(
            [
                anchors[:,:,size[2]-self.k:size[2]],
                anchors[:,:,0:self.k]
            ],
            dim=2
        )
        size = context_anchors.shape
        filtered_context_anchors = torch.cat(
            [
                context_anchors[:,:,size[2]-self.k:size[2]],
                context_anchors[:,:,0:self.k]
            ],
            dim=2
        )
        size = scores.shape
        filtered_scores = torch.cat(
            [
                scores[:,:,size[2]-self.k:size[2]],
                scores[:,:,0:self.k]
            ],
            dim=2
        )

        return filtered_anchors, filtered_context_anchors, filtered_scores
    
    def video_labels(self, scores):
       # scores: [receptive fields, batch, 2*self.k, classes]
        label = scores.permute(1,0,2,3)
        label = label.flatten(start_dim=1, end_dim=2)
        actioness = label.sum(dim=2)

        # sort by actioness score
        idx = torch.argsort(actioness, dim=1, descending=True)
        label = torch.gather(label, 1, idx.unsqueeze(-1).expand(label.shape))

        # keep topk per receptive field
        topk = label.shape[1] // 2
        label = label[:,0:topk]

        label = label.sum(dim=1)

        label = nn.functional.softmax(label, dim=0)
        return label

    
    def segment_mining(self, logits, context_anchors, scores, k_easy, k_hard):
        """ k easy and hard segments for action and background respectively

            segments are selected by confidence score
        """
        # logits: [receptive fields, batch, time, feature dimension]
        # anchors: [receptive fields, batch, 2*self.k, 2]
        # scores: [receptive fields, batch, 2*self.k, classes]
        actioness = torch.sum(scores, dim=3)
        # actioness: [receptive fields, batch, 2*self.k]

        # sort based on actioness
        idx = torch.argsort(actioness, dim=2, descending=True)
        context_anchors = torch.gather(context_anchors, 2, idx.unsqueeze(-1).expand(context_anchors.shape))
        scores = torch.gather(scores, 2, idx.unsqueeze(-1).expand(scores.shape))

        # select easy action and background segments
        shape = context_anchors.shape
        easy_act_anchors = context_anchors[:,:,0:k_easy,:]
        easy_bac_anchors = context_anchors[:,:,shape[2]-k_easy:shape[2],:]

        # remove selected from anchors
        hard_context_anchors = context_anchors[:,:,k_easy+1:shape[2]-k_easy-1,:]

        # select hard action and background segments
        shape = hard_context_anchors.shape
        hard_act_anchors = hard_context_anchors[:,:,0:k_hard,:]
        hard_bac_anchors = hard_context_anchors[:,:,shape[2]-k_hard:shape[2],:]

        # select respective logits
        easy_act = self.extract_logits(logits, easy_act_anchors, self.receptive_fields*2)
        easy_bac = self.extract_logits(logits, easy_bac_anchors, self.receptive_fields*2)
        hard_act = self.extract_logits(logits, hard_act_anchors, self.receptive_fields*2)
        hard_bac = self.extract_logits(logits, hard_bac_anchors, self.receptive_fields*2)

        for i in range(len(easy_act)):
            easy_act[i] = easy_act[i].flatten(start_dim=2)
            easy_bac[i] = easy_bac[i].flatten(start_dim=2)
            hard_act[i] = hard_act[i].flatten(start_dim=2)
            hard_bac[i] = hard_bac[i].flatten(start_dim=2)

        return easy_act, easy_bac, hard_act, hard_bac
    

    def forward(self, x):
        # x: [batch, time, feature dimension]
        anchors = self.gen_anchors(x, self.receptive_fields)
        context_anchors = self.gen_anchors(x, self.receptive_fields * 2)
        # anchors: [receptive fields, batch, time, 2]
        # context anchors include temporal context

        self.k = anchors.size(dim=2) // 4

        x = x.permute([0,2,1])
        # x: [batch, feature dimension, time]
        
        # sizes
        num_fields = len(self.receptive_fields)
        shape = x.shape
        batch = shape[0]
        feature_dim = shape[1]
        time = shape[2]

        # apply convnets
        logits = torch.empty([num_fields, batch, feature_dim, time]).cuda()
        for i in range(num_fields):
            logits_over_receptive_field = self.temp_conv_nets[i](x)
            logits[i] = logits_over_receptive_field
        # logits: [receptive fields, batch, feature dimension, time]

        logits = logits.permute([0,1,3,2])
        # logits: [receptive fields, batch, time, feature dimension]

        # scores
        scores = torch.empty([
            num_fields,
            batch,
            time,
            self.num_classes
        ]).cuda()
        # scores: [receptive fields, batch, time, classes]
        
        seg_list = self.extract_logits(logits, context_anchors, self.receptive_fields*2)
        # seg_list: list of [batch, time, context length, feature dimension] for each receptive field

        for i, rec_field_logits in enumerate(seg_list):
            rec_field_logits = rec_field_logits.flatten(start_dim=2).permute(0,2,1)
            rec_field_scores = self.classifiers[i](rec_field_logits).permute(0,2,1)
            scores[i] = rec_field_scores

        # select best proposals to keep
        anchors, context_anchors, scores = self.filter_proposals(anchors, context_anchors, scores)
        
        labels = None
        contrast_pairs = None
        if self.training:
            labels = self.video_labels(scores)

            k_easy = self.k // 8
            k_hard = self.k // 8

            easy_act, easy_bac, hard_act, hard_bac = self.segment_mining(
                logits,
                context_anchors,
                scores,
                k_easy,
                k_hard
            )

            contrast_pairs = {
                'EA': easy_act,
                'EB': easy_bac,
                'HA': hard_act,
                'HB': hard_bac
            }

        return scores, labels, anchors, context_anchors, logits, contrast_pairs
