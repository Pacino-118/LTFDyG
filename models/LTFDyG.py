import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
from models.modules import LTF
from utils.utils import NeighborSampler

class LTFDyG(nn.Module):

    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler: NeighborSampler,
                 time_feat_dim: int, channel_embedding_dim: int, patch_size: int = 1, num_layers: int = 2, num_heads: int = 2,
                 dropout: float = 0.1, max_input_sequence_length: int = 512, device: str = 'cpu',
                 lete_p: float = 0.5, lete_layer_norm: bool = True, lete_scale: bool = True, lete_param_grad: bool = True):
        super(LTFDyG, self).__init__()

        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32)).to(device)

        self.neighbor_sampler = neighbor_sampler
        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = self.edge_raw_features.shape[1]
        self.time_feat_dim = time_feat_dim
        self.channel_embedding_dim = channel_embedding_dim
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.max_input_sequence_length = max_input_sequence_length
        self.device = device

        self.time_encoder = LTF(
            dim=time_feat_dim,
            p=lete_p,
            layer_norm=lete_layer_norm,
            scale=lete_scale,
            parameter_requires_grad=lete_param_grad
        )

        self.neighbor_interaction_feat_dim = self.channel_embedding_dim
        self.Temporal_aware_Neighbor_InteractionEncoder = Temporal_aware_Neighbor_InteractionEncoder(
            neighbor_interaction_feat_dim=self.neighbor_interaction_feat_dim,
            device=self.device
        )

        self.projection_layer = nn.ModuleDict({
            'node': nn.Linear(in_features=self.patch_size * self.node_feat_dim, out_features=self.channel_embedding_dim, bias=True),
            'edge': nn.Linear(in_features=self.patch_size * self.edge_feat_dim, out_features=self.channel_embedding_dim, bias=True),
            'time': nn.Linear(in_features=self.patch_size * self.time_feat_dim, out_features=self.channel_embedding_dim, bias=True),
            'Temporal_aware_Neighbor_Interaction': nn.Linear(in_features=self.patch_size * self.neighbor_interaction_feat_dim, out_features=self.channel_embedding_dim, bias=True)
        })

        self.num_channels = 4
        self.transformers = nn.ModuleList([
            Two_stream_TransformerEncoder(attention_dim=self.num_channels * self.channel_embedding_dim, num_heads=self.num_heads,
                               dropout=self.dropout)
            for _ in range(self.num_layers)
        ])

        self.output_layer = nn.Linear(in_features=self.num_channels * self.channel_embedding_dim,
                                      out_features=self.node_feat_dim, bias=True)

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray):

        src_nodes_neighbor_ids_list, src_nodes_edge_ids_list, src_nodes_neighbor_times_list = \
            self.neighbor_sampler.get_all_first_hop_neighbors(node_ids=src_node_ids, node_interact_times=node_interact_times)

        dst_nodes_neighbor_ids_list, dst_nodes_edge_ids_list, dst_nodes_neighbor_times_list = \
            self.neighbor_sampler.get_all_first_hop_neighbors(node_ids=dst_node_ids, node_interact_times=node_interact_times)

        src_padded_nodes_neighbor_ids, src_padded_nodes_edge_ids, src_padded_nodes_neighbor_times = \
            self.pad_sequences(src_node_ids, node_interact_times, src_nodes_neighbor_ids_list,
                               src_nodes_edge_ids_list, src_nodes_neighbor_times_list,
                               patch_size=self.patch_size, max_input_sequence_length=self.max_input_sequence_length)

        dst_padded_nodes_neighbor_ids, dst_padded_nodes_edge_ids, dst_padded_nodes_neighbor_times = \
            self.pad_sequences(dst_node_ids, node_interact_times, dst_nodes_neighbor_ids_list,
                               dst_nodes_edge_ids_list, dst_nodes_neighbor_times_list,
                               patch_size=self.patch_size, max_input_sequence_length=self.max_input_sequence_length)

        src_padded_nodes_Temporal_aware_Neighbor_InteractionEncoder_features, dst_padded_nodes_Temporal_aware_Neighbor_InteractionEncoder_features = \
            self.Temporal_aware_Neighbor_InteractionEncoder(
                src_padded_nodes_neighbor_ids=src_padded_nodes_neighbor_ids,
                dst_padded_nodes_neighbor_ids=dst_padded_nodes_neighbor_ids,
                src_padded_nodes_neighbor_times=src_padded_nodes_neighbor_times,
                dst_padded_nodes_neighbor_times=dst_padded_nodes_neighbor_times,
                node_interact_times=node_interact_times
            )

        src_padded_nodes_neighbor_node_raw_features, src_padded_nodes_edge_raw_features, src_padded_nodes_neighbor_time_features = \
            self.get_features(node_interact_times, src_padded_nodes_neighbor_ids, src_padded_nodes_edge_ids, src_padded_nodes_neighbor_times, self.time_encoder)

        dst_padded_nodes_neighbor_node_raw_features, dst_padded_nodes_edge_raw_features, dst_padded_nodes_neighbor_time_features = \
            self.get_features(node_interact_times, dst_padded_nodes_neighbor_ids, dst_padded_nodes_edge_ids, dst_padded_nodes_neighbor_times, self.time_encoder)

        src_patches_nodes_neighbor_node_raw_features, src_patches_nodes_edge_raw_features, \
        src_patches_nodes_neighbor_time_features, src_patches_nodes_neighbor_co_occurrence_features = \
            self.get_patches(src_padded_nodes_neighbor_node_raw_features,
                             src_padded_nodes_edge_raw_features,
                             src_padded_nodes_neighbor_time_features,
                             src_padded_nodes_neighbor_co_occurrence_features,
                             patch_size=self.patch_size)

        dst_patches_nodes_neighbor_node_raw_features, dst_patches_nodes_edge_raw_features, \
        dst_patches_nodes_neighbor_time_features, dst_patches_nodes_neighbor_co_occurrence_features = \
            self.get_patches(dst_padded_nodes_neighbor_node_raw_features,
                             dst_padded_nodes_edge_raw_features,
                             dst_padded_nodes_neighbor_time_features,
                             dst_padded_nodes_neighbor_co_occurrence_features,
                             patch_size=self.patch_size)

        src_patches_nodes_neighbor_node_raw_features = self.projection_layer['node'](src_patches_nodes_neighbor_node_raw_features)
        src_patches_nodes_edge_raw_features = self.projection_layer['edge'](src_patches_nodes_edge_raw_features)
        src_patches_nodes_neighbor_time_features = self.projection_layer['time'](src_patches_nodes_neighbor_time_features)
        src_patches_nodes_neighbor_co_occurrence_features = self.projection_layer['Temporal_aware_Neighbor_Interaction'](src_patches_nodes_neighbor_co_occurrence_features)

        dst_patches_nodes_neighbor_node_raw_features = self.projection_layer['node'](dst_patches_nodes_neighbor_node_raw_features)
        dst_patches_nodes_edge_raw_features = self.projection_layer['edge'](dst_patches_nodes_edge_raw_features)
        dst_patches_nodes_neighbor_time_features = self.projection_layer['time'](dst_patches_nodes_neighbor_time_features)
        dst_patches_nodes_neighbor_co_occurrence_features = self.projection_layer['Temporal_aware_Neighbor_Interaction'](dst_patches_nodes_neighbor_co_occurrence_features)

        batch_size = len(src_patches_nodes_neighbor_node_raw_features)
        src_num_patches = src_patches_nodes_neighbor_node_raw_features.shape[1]
        dst_num_patches = dst_patches_nodes_neighbor_node_raw_features.shape[1]

        src_masks = (src_padded_nodes_neighbor_ids != 0).astype(np.float32)
        dst_masks = (dst_padded_nodes_neighbor_ids != 0).astype(np.float32)

        src_patches_data = torch.stack([
            src_patches_nodes_neighbor_node_raw_features,
            src_patches_nodes_edge_raw_features,
            src_patches_nodes_neighbor_time_features,
            src_patches_nodes_neighbor_co_occurrence_features
        ], dim=2).reshape(batch_size, src_num_patches, self.num_channels * self.channel_embedding_dim)

        dst_patches_data = torch.stack([
            dst_patches_nodes_neighbor_node_raw_features,
            dst_patches_nodes_edge_raw_features,
            dst_patches_nodes_neighbor_time_features,
            dst_patches_nodes_neighbor_co_occurrence_features
        ], dim=2).reshape(batch_size, dst_num_patches, self.num_channels * self.channel_embedding_dim)

        for transformer in self.transformers:
            src_self_attention = transformer(src_patches_data, src_patches_data, src_patches_data, src_masks)
            dst_self_attention = transformer(dst_patches_data, dst_patches_data, dst_patches_data, dst_masks)
            src_cross_attention = transformer(src_self_attention, dst_self_attention, dst_self_attention, dst_masks)
            dst_cross_attention = transformer(dst_self_attention, src_self_attention, src_self_attention, src_masks)

        src_patches_data, dst_patches_data = src_cross_attention, dst_cross_attention
        src_patches_data = torch.mean(src_patches_data, dim=1)
        dst_patches_data = torch.mean(dst_patches_data, dim=1)

        src_node_embeddings = self.output_layer(src_patches_data)
        dst_node_embeddings = self.output_layer(dst_patches_data)

        return src_node_embeddings, dst_node_embeddings

    def pad_sequences(self, node_ids: np.ndarray, node_interact_times: np.ndarray, nodes_neighbor_ids_list: list, nodes_edge_ids_list: list,
                      nodes_neighbor_times_list: list, patch_size: int = 1, max_input_sequence_length: int = 256):
        """
        pad the sequences for nodes in node_ids
        :param node_ids: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, )
        :param nodes_neighbor_ids_list: list of ndarrays, each ndarray contains neighbor ids for nodes in node_ids
        :param nodes_edge_ids_list: list of ndarrays, each ndarray contains edge ids for nodes in node_ids
        :param nodes_neighbor_times_list: list of ndarrays, each ndarray contains neighbor interaction timestamp for nodes in node_ids
        :param patch_size: int, patch size
        :param max_input_sequence_length: int, maximal number of neighbors for each node
        :return:
        """
        assert max_input_sequence_length - 1 > 0, 'Maximal number of neighbors for each node should be greater than 1!'
        max_seq_length = 0
        # first cut the sequence of nodes whose number of neighbors is more than max_input_sequence_length - 1 (we need to include the target node in the sequence)
        for idx in range(len(nodes_neighbor_ids_list)):
            assert len(nodes_neighbor_ids_list[idx]) == len(nodes_edge_ids_list[idx]) == len(nodes_neighbor_times_list[idx])
            if len(nodes_neighbor_ids_list[idx]) > max_input_sequence_length - 1:
                # cut the sequence by taking the most recent max_input_sequence_length interactions
                nodes_neighbor_ids_list[idx] = nodes_neighbor_ids_list[idx][-(max_input_sequence_length - 1):]
                nodes_edge_ids_list[idx] = nodes_edge_ids_list[idx][-(max_input_sequence_length - 1):]
                nodes_neighbor_times_list[idx] = nodes_neighbor_times_list[idx][-(max_input_sequence_length - 1):]
            if len(nodes_neighbor_ids_list[idx]) > max_seq_length:
                max_seq_length = len(nodes_neighbor_ids_list[idx])

        # include the target node itself
        max_seq_length += 1
        if max_seq_length % patch_size != 0:
            max_seq_length += (patch_size - max_seq_length % patch_size)
        assert max_seq_length % patch_size == 0

        # pad the sequences
        # three ndarrays with shape (batch_size, max_seq_length)
        padded_nodes_neighbor_ids = np.zeros((len(node_ids), max_seq_length)).astype(np.longlong)
        padded_nodes_edge_ids = np.zeros((len(node_ids), max_seq_length)).astype(np.longlong)
        padded_nodes_neighbor_times = np.zeros((len(node_ids), max_seq_length)).astype(np.float32)

        for idx in range(len(node_ids)):
            padded_nodes_neighbor_ids[idx, 0] = node_ids[idx]
            padded_nodes_edge_ids[idx, 0] = 0
            padded_nodes_neighbor_times[idx, 0] = node_interact_times[idx]

            if len(nodes_neighbor_ids_list[idx]) > 0:
                padded_nodes_neighbor_ids[idx, 1: len(nodes_neighbor_ids_list[idx]) + 1] = nodes_neighbor_ids_list[idx]
                padded_nodes_edge_ids[idx, 1: len(nodes_edge_ids_list[idx]) + 1] = nodes_edge_ids_list[idx]
                padded_nodes_neighbor_times[idx, 1: len(nodes_neighbor_times_list[idx]) + 1] = nodes_neighbor_times_list[idx]

        # three ndarrays with shape (batch_size, max_seq_length)
        return padded_nodes_neighbor_ids, padded_nodes_edge_ids, padded_nodes_neighbor_times

    def get_features(self, node_interact_times: np.ndarray, padded_nodes_neighbor_ids: np.ndarray, padded_nodes_edge_ids: np.ndarray,
                     padded_nodes_neighbor_times: np.ndarray, time_encoder: LTF):
        """
        get node, edge and time features
        :param node_interact_times: ndarray, shape (batch_size, )
        :param padded_nodes_neighbor_ids: ndarray, shape (batch_size, max_seq_length)
        :param padded_nodes_edge_ids: ndarray, shape (batch_size, max_seq_length)
        :param padded_nodes_neighbor_times: ndarray, shape (batch_size, max_seq_length)
        :param time_encoder: TimeEncoder, time encoder
        :return:
        """
        # Tensor, shape (batch_size, max_seq_length, node_feat_dim)
        padded_nodes_neighbor_node_raw_features = self.node_raw_features[torch.from_numpy(padded_nodes_neighbor_ids)]
        # Tensor, shape (batch_size, max_seq_length, edge_feat_dim)
        padded_nodes_edge_raw_features = self.edge_raw_features[torch.from_numpy(padded_nodes_edge_ids)]
        # Tensor, shape (batch_size, max_seq_length, time_feat_dim)
       # padded_nodes_neighbor_time_features = time_encoder(timestamps=torch.from_numpy(node_interact_times[:, np.newaxis] - padded_nodes_neighbor_times).float().to(self.device))
        time_diff = torch.from_numpy(node_interact_times[:, np.newaxis] - padded_nodes_neighbor_times).float().to(
            self.device)

        # ndarray, set the time features to all zeros for the padded timestamp
        padded_nodes_neighbor_time_features[torch.from_numpy(padded_nodes_neighbor_ids == 0)] = 0.0

        return padded_nodes_neighbor_node_raw_features, padded_nodes_edge_raw_features, padded_nodes_neighbor_time_features

    def get_patches(self, padded_nodes_neighbor_node_raw_features: torch.Tensor, padded_nodes_edge_raw_features: torch.Tensor,
                    padded_nodes_neighbor_time_features: torch.Tensor, padded_nodes_neighbor_co_occurrence_features: torch.Tensor = None, patch_size: int = 1):
        """
        get the sequence of patches for nodes
        :param padded_nodes_neighbor_node_raw_features: Tensor, shape (batch_size, max_seq_length, node_feat_dim)
        :param padded_nodes_edge_raw_features: Tensor, shape (batch_size, max_seq_length, edge_feat_dim)
        :param padded_nodes_neighbor_time_features: Tensor, shape (batch_size, max_seq_length, time_feat_dim)
        :param padded_nodes_neighbor_co_occurrence_features: Tensor, shape (batch_size, max_seq_length, neighbor_interaction_feat_dim)
        :param patch_size: int, patch size
        :return:
        """
        assert padded_nodes_neighbor_node_raw_features.shape[1] % patch_size == 0
        num_patches = padded_nodes_neighbor_node_raw_features.shape[1] // patch_size

        # list of Tensors with shape (num_patches, ), each Tensor with shape (batch_size, patch_size, node_feat_dim)
        patches_nodes_neighbor_node_raw_features, patches_nodes_edge_raw_features, \
        patches_nodes_neighbor_time_features, patches_nodes_neighbor_co_occurrence_features = [], [], [], []

        for patch_id in range(num_patches):
            start_idx = patch_id * patch_size
            end_idx = patch_id * patch_size + patch_size
            patches_nodes_neighbor_node_raw_features.append(padded_nodes_neighbor_node_raw_features[:, start_idx: end_idx, :])
            patches_nodes_edge_raw_features.append(padded_nodes_edge_raw_features[:, start_idx: end_idx, :])
            patches_nodes_neighbor_time_features.append(padded_nodes_neighbor_time_features[:, start_idx: end_idx, :])
            patches_nodes_neighbor_co_occurrence_features.append(padded_nodes_neighbor_co_occurrence_features[:, start_idx: end_idx, :])

        batch_size = len(padded_nodes_neighbor_node_raw_features)
        # Tensor, shape (batch_size, num_patches, patch_size * node_feat_dim)
        patches_nodes_neighbor_node_raw_features = torch.stack(patches_nodes_neighbor_node_raw_features, dim=1).reshape(batch_size, num_patches, patch_size * self.node_feat_dim)
        # Tensor, shape (batch_size, num_patches, patch_size * edge_feat_dim)
        patches_nodes_edge_raw_features = torch.stack(patches_nodes_edge_raw_features, dim=1).reshape(batch_size, num_patches, patch_size * self.edge_feat_dim)
        # Tensor, shape (batch_size, num_patches, patch_size * time_feat_dim)
        patches_nodes_neighbor_time_features = torch.stack(patches_nodes_neighbor_time_features, dim=1).reshape(batch_size, num_patches, patch_size * self.time_feat_dim)

        patches_nodes_neighbor_co_occurrence_features = torch.stack(patches_nodes_neighbor_co_occurrence_features, dim=1).reshape(batch_size, num_patches, patch_size * self.neighbor_interaction_feat_dim)

        return patches_nodes_neighbor_node_raw_features, patches_nodes_edge_raw_features, patches_nodes_neighbor_time_features, patches_nodes_neighbor_co_occurrence_features

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        """
        set neighbor sampler to neighbor_sampler and reset the random state (for reproducing the results for uniform and time_interval_aware sampling)
        :param neighbor_sampler: NeighborSampler, neighbor sampler
        :return:
        """
        self.neighbor_sampler = neighbor_sampler
        if self.neighbor_sampler.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
            assert self.neighbor_sampler.seed is not None
            self.neighbor_sampler.reset_random_state()

class Temporal_aware_Neighbor_InteractionEncoder(nn.Module):
    def __init__(self, neighbor_interaction_feat_dim: int, time_encoder: LTF, device: str = 'cpu'):
        super(Temporal_aware_Neighbor_InteractionEncoder, self).__init__()
        self.neighbor_interaction_feat_dim = neighbor_interaction_feat_dim
        self.device = device
        self.time_encoder = time_encoder

        self.time_to_scalar = nn.Linear(self.time_encoder.dim, 1)
        self.temporal_aware_neighbor_interaction_encoder_layer = nn.Sequential(
            nn.Linear(in_features=1, out_features=self.neighbor_interaction_feat_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.neighbor_interaction_feat_dim, out_features=self.neighbor_interaction_feat_dim)
        )

    def forward(self, src_padded_nodes_neighbor_ids, dst_padded_nodes_neighbor_ids,
                src_padded_nodes_neighbor_times, dst_padded_nodes_neighbor_times,
                node_interact_times):
        """
        Parallel GPU version without Python loops.
        Inputs:
          - src_padded_nodes_neighbor_ids, dst_padded_nodes_neighbor_ids : np.ndarray (B, L)
          - src_padded_nodes_neighbor_times, dst_padded_nodes_neighbor_times : np.ndarray (B, L)
          - node_interact_times : np.ndarray (B,)
        Returns:
          - src_features, dst_features : torch.Tensor, shapes (B, L, neighbor_interaction_feat_dim)
        """

        # to device tensors
        src_ids = torch.from_numpy(src_padded_nodes_neighbor_ids).long().to(self.device)  # (B, L)
        dst_ids = torch.from_numpy(dst_padded_nodes_neighbor_ids).long().to(self.device)  # (B, L)

        B, L = src_ids.shape

        # handle degenerate case: no neighbors at all
        if B == 0 or L == 0:
            empty_src = torch.zeros((0, 0, self.neighbor_interaction_feat_dim), device=self.device)
            return empty_src, empty_src

        # Compute max id in this batch (avoid huge global max); at least 0
        max_id_in_batch = int(max(int(src_ids.max().item()), int(dst_ids.max().item()), 0))
        # ensure at least 1 to avoid zero-sized histogram
        if max_id_in_batch < 1:
            # all zeros -> nothing to count, return zero features of correct shape
            zero_src = torch.zeros((B, L, self.neighbor_interaction_feat_dim), device=self.device)
            zero_dst = torch.zeros((B, L, self.neighbor_interaction_feat_dim), device=self.device)
            return zero_src, zero_dst

        # Prepare batch offsets for flatten indexing
        bucket_size = max_id_in_batch + 1  # include id==max_id_in_batch
        batch_offsets = (torch.arange(B, device=self.device, dtype=torch.long) * bucket_size).unsqueeze(1)  # (B,1)

        # Mask padding positions where id == 0 (we don't want to count padding)
        src_mask = (src_ids != 0).long()   # (B,L)
        dst_mask = (dst_ids != 0).long()   # (B,L)

        # Flattened indices and values for scatter_add
        # For src histogram
        flat_idx_src = (src_ids + batch_offsets).view(-1)  # (B*L,)
        vals_src = src_mask.view(-1).float()               # 1 for non-pad, 0 for pad

        # For dst histogram
        flat_idx_dst = (dst_ids + batch_offsets).view(-1)
        vals_dst = dst_mask.view(-1).float()

        total_buckets = B * bucket_size  # length of flattened histogram

        # allocate flattened histograms and scatter_add (fully vectorized)
        hist_src_flat = torch.zeros(total_buckets, dtype=torch.float32, device=self.device)
        hist_dst_flat = torch.zeros(total_buckets, dtype=torch.float32, device=self.device)

        # Use scatter_add_ with index and values
        hist_src_flat = hist_src_flat.scatter_add(0, flat_idx_src, vals_src)
        hist_dst_flat = hist_dst_flat.scatter_add(0, flat_idx_dst, vals_dst)

        # reshape histograms back to (B, bucket_size)
        hist_src = hist_src_flat.view(B, bucket_size)  # counts of each id per row for src rows
        hist_dst = hist_dst_flat.view(B, bucket_size)

        # Gather counts at each neighbor position:
        # note: ids are within [0, max_id_in_batch], so gather is safe
        # counts_src_in_src: for each src position, how many times that id appears in its own src row
        counts_src_in_src = hist_src.gather(1, src_ids)  # (B, L)
        counts_dst_in_dst = hist_dst.gather(1, dst_ids)  # (B, L)

        # cross counts: how many times a src-position id appears in the corresponding dst row
        counts_src_in_dst = hist_dst.gather(1, src_ids)  # (B, L)
        counts_dst_in_src = hist_src.gather(1, dst_ids)  # (B, L)

        # Convert to float
        counts_src_in_src = counts_src_in_src.float()
        counts_dst_in_dst = counts_dst_in_dst.float()
        counts_src_in_dst = counts_src_in_dst.float()
        counts_dst_in_src = counts_dst_in_src.float()

        # Build (B, L, 2) counts arrays
        src_counts = torch.stack([counts_src_in_src, counts_src_in_dst], dim=2)  # (B, L, 2)
        dst_counts = torch.stack([counts_dst_in_src, counts_dst_in_dst], dim=2)  # (B, L, 2)

        # Compute time differences and weights (same as original)
        src_time_diff = torch.from_numpy(node_interact_times[:, None] - src_padded_nodes_neighbor_times).float().to(self.device)  # (B, L)
        dst_time_diff = torch.from_numpy(node_interact_times[:, None] - dst_padded_nodes_neighbor_times).float().to(self.device)

        src_weights = torch.sigmoid(self.time_to_scalar(self.time_encoder(src_time_diff))).squeeze(-1)  # (B, L)
        dst_weights = torch.sigmoid(self.time_to_scalar(self.time_encoder(dst_time_diff))).squeeze(-1)

        # Apply weights to counts
        src_weighted_counts = src_counts * src_weights.unsqueeze(-1)  # (B, L, 2)
        dst_weighted_counts = dst_counts * dst_weights.unsqueeze(-1)

        # Encode and aggregate along neighbor dimension to get per-position encoded vector,
        # then sum across the last-but-one dim as original implementation
        # (original used .unsqueeze(-1) then .sum(dim=2) after a small linear block)
        src_features = self.temporal_aware_neighbor_interaction_encoder_layer(src_weighted_counts.unsqueeze(-1)).sum(dim=2)  # (B, L, feat_dim)
        dst_features = self.temporal_aware_neighbor_interaction_encoder_layer(dst_weighted_counts.unsqueeze(-1)).sum(dim=2)

        return src_features, dst_features


class Two_stream_TransformerEncoder(nn.Module):

    def __init__(self, attention_dim: int, num_heads: int, dropout: float = 0.1):
        """
        Transformer encoder (from TCL).
        :param attention_dim: int, dimension of the attention vector
        :param num_heads: int, number of attention heads
        :param dropout: float, dropout rate
        """
        super(Two_stream_TransformerEncoder, self).__init__()
        # use the MultiheadAttention implemented by PyTorch
        self.multi_head_attention = nn.MultiheadAttention(embed_dim=attention_dim, num_heads=num_heads, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

        self.linear_layers = nn.ModuleList([
            nn.Linear(in_features=attention_dim, out_features=4 * attention_dim),
            nn.Linear(in_features=4 * attention_dim, out_features=attention_dim)
        ])
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(attention_dim),
            nn.LayerNorm(attention_dim)
        ])

    def forward(self, inputs_query: torch.Tensor, inputs_key: torch.Tensor = None, inputs_value: torch.Tensor = None,
                neighbor_masks: np.ndarray = None):
        """
        encode the inputs by Transformer encoder
        :param inputs_query: Tensor, shape (batch_size, target_seq_length, self.attention_dim)
        :param inputs_key: Tensor, shape (batch_size, source_seq_length, self.attention_dim)
        :param inputs_value: Tensor, shape (batch_size, source_seq_length, self.attention_dim)
        :param neighbor_masks: ndarray, shape (batch_size, source_seq_length), used to create mask of neighbors for nodes in the batch
        :return:
        """
        if inputs_key is None or inputs_value is None:
            assert inputs_key is None and inputs_value is None
            inputs_key = inputs_value = inputs_query

        # note that the MultiheadAttention module accept input data with shape (seq_length, batch_size, input_dim), so we need to transpose the input
        # transposed_inputs_query, Tensor, shape (target_seq_length, batch_size, self.attention_dim)
        # transposed_inputs_key, Tensor, shape (source_seq_length, batch_size, self.attention_dim)
        # transposed_inputs_value, Tensor, shape (source_seq_length, batch_size, self.attention_dim)
        transposed_inputs_query, transposed_inputs_key, transposed_inputs_value = inputs_query.transpose(0,1), inputs_key.transpose(0,1), inputs_value.transpose(0,1)

        if neighbor_masks is not None:
            # Ensure numpy array is contiguous and numeric
            neighbor_masks = np.ascontiguousarray(neighbor_masks)

            # Optional sanity checks (can keep for debugging)
            # print("neighbor_masks dtype:", neighbor_masks.dtype, "shape:", neighbor_masks.shape, "flags:", neighbor_masks.flags)

            # Create a tensor on the correct device via torch.tensor (safe copy)
            # then convert to boolean mask expected by MultiheadAttention (True = mask)
            neighbor_masks_tensor = torch.tensor(neighbor_masks, device=inputs_query.device)
            # If neighbor_masks was float where 1.0 = valid, 0.0 = pad, you used ==0 to mark padding:
            neighbor_masks = (neighbor_masks_tensor == 0)
            # ensure dtype is bool (key_padding_mask expects bool or byte)
            neighbor_masks = neighbor_masks.to(torch.bool)

        # Tensor, shape (batch_size, target_seq_length, self.attention_dim)
        hidden_states = self.multi_head_attention(query=transposed_inputs_query, key=transposed_inputs_key,
                                                  value=transposed_inputs_value, key_padding_mask=neighbor_masks)[
            0].transpose(0, 1)
        # Tensor, shape (batch_size, target_seq_length, self.attention_dim)
        outputs = self.norm_layers[0](inputs_query + self.dropout(hidden_states))
        # Tensor, shape (batch_size, target_seq_length, self.attention_dim)
        hidden_states = self.linear_layers[1](self.dropout(F.relu(self.linear_layers[0](outputs))))
        # Tensor, shape (batch_size, target_seq_length, self.attention_dim)
        outputs = self.norm_layers[1](outputs + self.dropout(hidden_states))

        return outputs