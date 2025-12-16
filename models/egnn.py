import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Sequential, Linear, Conv1d, ModuleList
from torch_scatter import scatter_sum, scatter_softmax
from torch_geometric.nn import radius_graph, knn_graph, knn
from models.common import GaussianSmearing, MLP, NONLINEARITIES
import math


class CrossAttentionBlock(Module):
    """
    Cross-Attention module for ligand-protein interaction.
    
    This module enables direct interaction between ligand and protein nodes,
    providing:
    1. Long-range interaction capture without multi-hop message passing
    2. Content-based addressing (not just distance-based)
    3. Robustness during early diffusion stages when ligand positions are noisy
    """
    
    def __init__(self, node_dim, num_heads=4, dropout=0.1, use_pos_encoding=True, cutoff=10.0):
        super().__init__()
        self.node_dim = node_dim
        self.num_heads = num_heads
        self.head_dim = node_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_pos_encoding = use_pos_encoding
        self.cutoff = cutoff
        
        assert node_dim % num_heads == 0, "node_dim must be divisible by num_heads"
        
        # Query, Key, Value projections for ligand (query) attending to protein (key, value)
        self.q_proj = Linear(node_dim, node_dim)
        self.k_proj = Linear(node_dim, node_dim)
        self.v_proj = Linear(node_dim, node_dim)
        self.out_proj = Linear(node_dim, node_dim)
        
        # Optional positional encoding based on 3D distance
        if use_pos_encoding:
            self.dist_embedding = GaussianSmearing(start=0.0, stop=cutoff, num_gaussians=32)
            self.pos_bias_proj = Linear(32, num_heads)
        
        # Layer normalization and dropout
        self.layer_norm_q = nn.LayerNorm(node_dim)
        self.layer_norm_kv = nn.LayerNorm(node_dim)
        self.dropout = nn.Dropout(dropout)
        
        # FFN after attention
        self.ffn = Sequential(
            Linear(node_dim, node_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            Linear(node_dim * 4, node_dim),
            nn.Dropout(dropout)
        )
        self.layer_norm_ffn = nn.LayerNorm(node_dim)
    
    def forward(self, ligand_h, ligand_pos, ligand_batch, 
                protein_h, protein_pos, protein_batch):
        """
        Cross-attention from ligand to protein.
        
        Args:
            ligand_h: (N_lig, node_dim) - Ligand node features
            ligand_pos: (N_lig, 3) - Ligand node positions  
            ligand_batch: (N_lig,) - Batch indices for ligand nodes
            protein_h: (N_prot, node_dim) - Protein node features
            protein_pos: (N_prot, 3) - Protein node positions
            protein_batch: (N_prot,) - Batch indices for protein nodes
            
        Returns:
            updated_ligand_h: (N_lig, node_dim) - Updated ligand features
        """
        N_lig = ligand_h.size(0)
        N_prot = protein_h.size(0)
        device = ligand_h.device
        
        # Pre-norm
        ligand_h_norm = self.layer_norm_q(ligand_h)
        protein_h_norm = self.layer_norm_kv(protein_h)
        
        # Compute Q, K, V
        Q = self.q_proj(ligand_h_norm).view(N_lig, self.num_heads, self.head_dim)
        K = self.k_proj(protein_h_norm).view(N_prot, self.num_heads, self.head_dim)
        V = self.v_proj(protein_h_norm).view(N_prot, self.num_heads, self.head_dim)
        
        # Compute attention scores with batch masking
        # We need to handle variable-size batches efficiently
        num_graphs = ligand_batch.max().item() + 1
        
        # Create batch mask: (N_lig, N_prot) where True means same batch
        batch_mask = ligand_batch.unsqueeze(1) == protein_batch.unsqueeze(0)  # (N_lig, N_prot)
        
        # Compute attention scores: (N_lig, N_prot, num_heads)
        attn_scores = torch.einsum('lhd,phd->lph', Q, K) * self.scale
        
        # Add positional bias if enabled
        if self.use_pos_encoding:
            # Compute pairwise distances: (N_lig, N_prot)
            dist = torch.cdist(ligand_pos, protein_pos, p=2)  # (N_lig, N_prot)
            dist_clamped = dist.clamp(max=self.cutoff)
            
            # Get distance embeddings and project to bias: (N_lig, N_prot, num_heads)
            dist_flat = dist_clamped.view(-1)
            dist_emb = self.dist_embedding(dist_flat)  # (N_lig * N_prot, 32)
            pos_bias = self.pos_bias_proj(dist_emb).view(N_lig, N_prot, self.num_heads)
            
            attn_scores = attn_scores + pos_bias
        
        # Apply batch mask (set scores to -inf for cross-batch pairs)
        attn_mask = ~batch_mask  # True where we should mask
        attn_scores = attn_scores.masked_fill(attn_mask.unsqueeze(-1), float('-inf'))
        
        # Softmax over protein dimension
        attn_weights = F.softmax(attn_scores, dim=1)  # (N_lig, N_prot, num_heads)
        attn_weights = self.dropout(attn_weights)
        
        # Handle NaN from empty protein batches
        attn_weights = attn_weights.nan_to_num(0.0)
        
        # Compute weighted sum of values: (N_lig, num_heads, head_dim)
        out = torch.einsum('lph,phd->lhd', attn_weights, V)
        out = out.reshape(N_lig, self.node_dim)
        out = self.out_proj(out)
        
        # Residual connection
        ligand_h = ligand_h + self.dropout(out)
        
        # FFN with residual
        ligand_h = ligand_h + self.ffn(self.layer_norm_ffn(ligand_h))
        # add debug print
        # print(f"Ligand: {ligand_h.shape}, Protein: {protein_h.shape}, Batch Mask: {batch_mask.shape}")
        return ligand_h


class EfficientCrossAttentionBlock(Module):
    """
    Memory-efficient Cross-Attention with local windowing.
    
    For large protein pockets, full attention can be expensive.
    This version uses distance-based neighbor selection to limit computation.
    """
    
    def __init__(self, node_dim, num_heads=4, dropout=0.1, 
                 cutoff=10.0, max_neighbors=64):
        super().__init__()
        self.node_dim = node_dim
        self.num_heads = num_heads
        self.head_dim = node_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        
        assert node_dim % num_heads == 0
        
        self.q_proj = Linear(node_dim, node_dim)
        self.k_proj = Linear(node_dim, node_dim)
        self.v_proj = Linear(node_dim, node_dim)
        self.out_proj = Linear(node_dim, node_dim)
        
        self.dist_embedding = GaussianSmearing(start=0.0, stop=cutoff, num_gaussians=32)
        self.pos_bias_proj = Linear(32, num_heads)
        
        self.layer_norm_q = nn.LayerNorm(node_dim)
        self.layer_norm_kv = nn.LayerNorm(node_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.ffn = Sequential(
            Linear(node_dim, node_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            Linear(node_dim * 4, node_dim),
            nn.Dropout(dropout)
        )
        self.layer_norm_ffn = nn.LayerNorm(node_dim)
    
    def forward(self, ligand_h, ligand_pos, ligand_batch,
                protein_h, protein_pos, protein_batch):
        """
        Efficient cross-attention using sparse neighbor connections.
        """
        N_lig = ligand_h.size(0)
        device = ligand_h.device
        
        # Pre-norm
        ligand_h_norm = self.layer_norm_q(ligand_h)
        protein_h_norm = self.layer_norm_kv(protein_h)
        
        # Find k-nearest protein neighbors for each ligand atom
        # knn(x, y, ...) returns edge_index where:
        #   edge_index[0] = indices in y (ligand, the query points)
        #   edge_index[1] = indices in x (protein, the neighbor points)
        edge_index = knn(protein_pos, ligand_pos, k=self.max_neighbors,
                         batch_x=protein_batch, batch_y=ligand_batch)
        lig_idx, prot_idx = edge_index  # Fixed: was incorrectly swapped before
        
        # Compute Q, K, V
        Q = self.q_proj(ligand_h_norm).view(N_lig, self.num_heads, self.head_dim)
        K = self.k_proj(protein_h_norm)
        V = self.v_proj(protein_h_norm)
        
        # Gather K, V for each edge
        K_neighbors = K[prot_idx].view(-1, self.num_heads, self.head_dim)  # (E, heads, head_dim)
        V_neighbors = V[prot_idx].view(-1, self.num_heads, self.head_dim)
        Q_expanded = Q[lig_idx]  # (E, heads, head_dim)
        
        # Compute attention scores for edges
        attn_scores = (Q_expanded * K_neighbors).sum(-1) * self.scale  # (E, heads)
        
        # Add positional bias
        dist = torch.norm(ligand_pos[lig_idx] - protein_pos[prot_idx], dim=-1)  # (E,)
        dist_clamped = dist.clamp(max=self.cutoff)
        dist_emb = self.dist_embedding(dist_clamped)  # (E, 32)
        pos_bias = self.pos_bias_proj(dist_emb)  # (E, heads)
        attn_scores = attn_scores + pos_bias
        
        # Softmax over neighbors (grouped by ligand index)
        attn_weights = scatter_softmax(attn_scores, lig_idx, dim=0)  # (E, heads)
        attn_weights = self.dropout(attn_weights)
        
        # Weighted sum of values
        weighted_V = attn_weights.unsqueeze(-1) * V_neighbors  # (E, heads, head_dim)
        out = scatter_sum(weighted_V, lig_idx, dim=0, dim_size=N_lig)  # (N_lig, heads, head_dim)
        out = out.reshape(N_lig, self.node_dim)
        out = self.out_proj(out)
        
        # Residual connection
        ligand_h = ligand_h + self.dropout(out)
        
        # FFN with residual
        ligand_h = ligand_h + self.ffn(self.layer_norm_ffn(ligand_h))
        
        return ligand_h


class NodeBlock(Module):

    def __init__(self, node_dim, edge_dim, hidden_dim, use_gate):
        super().__init__()
        self.use_gate = use_gate
        self.node_dim = node_dim
        
        self.node_net = MLP(node_dim, hidden_dim, hidden_dim)
        self.edge_net = MLP(edge_dim, hidden_dim, hidden_dim)
        self.msg_net = Linear(hidden_dim, hidden_dim)

        if self.use_gate:
            self.gate = MLP(edge_dim+node_dim+1, hidden_dim, hidden_dim) # add 1 for time

        self.centroid_lin = Linear(node_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.act = nn.ReLU()
        self.out_transform = Linear(hidden_dim, node_dim)

    def forward(self, x, edge_index, edge_attr, node_time):
        """
        Args:
            x:  Node features, (N, H).
            edge_index: (2, E).
            edge_attr:  (E, H)
        """
        N = x.size(0)
        row, col = edge_index   # (E,) , (E,)

        node_h = self.node_net(x)  # (N, H)

        # Compose messages
        edge_h = self.edge_net(edge_attr)  # (E, H_per_head)
        msg_j = self.msg_net(edge_h * node_h[col])

        if self.use_gate:
            gate = self.gate(torch.cat([edge_attr, x[col], node_time[col]], dim=-1))
            msg_j = msg_j * torch.sigmoid(gate)

        # Aggregate messages
        aggr_msg = scatter_sum(msg_j, row, dim=0, dim_size=N)
        out = self.centroid_lin(x) + aggr_msg

        out = self.layer_norm(out)
        out = self.out_transform(self.act(out))
        return out


class NodeEncoder(Module):
    
    def __init__(self, node_dim=256, edge_dim=64, key_dim=128, num_heads=4, 
                    num_blocks=6, k=48, cutoff=10.0, use_atten=True, use_gate=True,
                    dist_version='new'):
        super().__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.key_dim = key_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.k = k
        self.cutoff = cutoff
        self.use_atten = use_atten
        self.use_gate = use_gate

        if dist_version == 'new':
            self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=20)
            self.edge_emb = Linear(self.additional_edge_feat+20, edge_dim)
        elif dist_version == 'old':
            self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=edge_dim-self.additional_edge_feat)
            self.edge_emb = Linear(edge_dim, edge_dim)
        else:
            raise NotImplementedError('dist_version notimplemented')
        self.node_blocks = ModuleList()
        for _ in range(num_blocks):
            block = NodeBlock(
                node_dim=node_dim,
                edge_dim=edge_dim,
                key_dim=key_dim,
                num_heads=num_heads,
                use_atten=use_atten,
                use_gate=use_gate,
            )
            self.node_blocks.append(block)

    @property
    def out_channels(self):
        return self.node_dim

    def forward(self, h, pos, edge_index, is_mol):
        #NOTE in the encoder, the edge dose not change since the position of mol and protein is fixed
        edge_attr = self._add_edge_features(pos, edge_index, is_mol)
        for interaction in self.node_blocks:
            h = h + interaction(h, edge_index, edge_attr)
        return h

    @property
    def additional_edge_feat(self,):
        return 2

    def _add_edge_features(self, pos, edge_index, is_mol):
        edge_length = torch.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=1)
        edge_attr = self.distance_expansion(edge_length)
        # 2-vector represent the two node types (atoms of protein or mol)
        edge_src_feat = is_mol[edge_index[0]].float().view(-1, 1)
        edge_dst_feat = is_mol[edge_index[1]].float().view(-1, 1)
        edge_attr = torch.cat([edge_attr, edge_src_feat, edge_dst_feat], dim=1)
        edge_attr = self.edge_emb(edge_attr)
        return edge_attr


class BondFFN(Module):
    def __init__(self, bond_dim, node_dim, inter_dim, use_gate, out_dim=None):
        super().__init__()
        out_dim = bond_dim if out_dim is None else out_dim
        self.use_gate = use_gate
        self.bond_linear = Linear(bond_dim, inter_dim, bias=False)
        self.node_linear = Linear(node_dim, inter_dim, bias=False)
        self.inter_module = MLP(inter_dim, out_dim, inter_dim)
        if self.use_gate:
            self.gate = MLP(bond_dim+node_dim+1, out_dim, 32)  # +1 for time

    def forward(self, bond_feat_input, node_feat_input, time):
        bond_feat = self.bond_linear(bond_feat_input)
        node_feat = self.node_linear(node_feat_input)
        inter_feat = bond_feat * node_feat
        inter_feat = self.inter_module(inter_feat)
        if self.use_gate:
            gate = self.gate(torch.cat([bond_feat_input, node_feat_input, time], dim=-1))
            inter_feat = inter_feat * torch.sigmoid(gate)
        return inter_feat


class QKVLin(Module):
    def __init__(self, h_dim, key_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.q_lin = Linear(h_dim, key_dim)
        self.k_lin = Linear(h_dim, key_dim)
        self.v_lin = Linear(h_dim, h_dim)

    def forward(self, inputs):
        n = inputs.size(0)
        return [
            self.q_lin(inputs).view(n, self.num_heads, -1),
            self.k_lin(inputs).view(n, self.num_heads, -1),
            self.v_lin(inputs).view(n, self.num_heads, -1),
        ]


class BondBlock(Module):
    def __init__(self, bond_dim, node_dim, use_gate=True, use_atten=False, num_heads=2, key_dim=128):
        super().__init__()
        self.use_atten = use_atten
        self.use_gate = use_gate
        inter_dim = bond_dim * 2

        self.bond_ffn_left = BondFFN(bond_dim, node_dim, inter_dim=inter_dim, use_gate=use_gate)
        self.bond_ffn_right = BondFFN(bond_dim, node_dim, inter_dim=inter_dim, use_gate=use_gate)
        if self.use_atten:
            assert bond_dim % num_heads == 0
            assert key_dim % num_heads == 0
            # linear transformation for attention 
            self.qkv_left = QKVLin(bond_dim, key_dim, num_heads)
            self.qkv_right = QKVLin(bond_dim, key_dim, num_heads)
            self.layer_norm_atten1 = nn.LayerNorm(bond_dim)
            self.layer_norm_atten2 = nn.LayerNorm(bond_dim)
        
        self.node_ffn_left = Linear(node_dim, bond_dim)
        self.node_ffn_right = Linear(node_dim, bond_dim)

        self.self_ffn = Linear(bond_dim, bond_dim)
        self.layer_norm = nn.LayerNorm(bond_dim)
        self.out_transform = Linear(bond_dim, bond_dim)
        self.act = nn.ReLU()

    def forward(self, bond_h, bond_index, node_h, atten_index=None):
        """
        bond_h: (b, bond_dim)
        bond_index: (2, b)
        node_h: (n, node_dim)
        node_pos: (n, 3)
        """
        N = node_h.size(0)
        left_node, right_node = bond_index

        # message from neighbor bonds
        msg_bond_left = self.bond_ffn_left(bond_h, node_h[left_node])
        msg_bond_left = scatter_sum(msg_bond_left, right_node, dim=0, dim_size=N)
        msg_bond_left = msg_bond_left[left_node]

        msg_bond_right = self.bond_ffn_right(bond_h, node_h[right_node])
        msg_bond_right = scatter_sum(msg_bond_right, left_node, dim=0, dim_size=N)
        msg_bond_right = msg_bond_right[right_node]
        
        bond_h = (
            msg_bond_left + msg_bond_right
            + self.node_ffn_left(node_h[left_node])
            + self.node_ffn_right(node_h[right_node])
            + self.self_ffn(bond_h)
        )
        bond_h = self.layer_norm(bond_h)

        if self.use_atten:
            index_query_bond_left, index_key_bond_left, index_query_bond_right, index_key_bond_right = atten_index

            # left node
            h_queries, h_keys, h_values = self.qkv_left(bond_h)
            queries_i = h_queries[index_query_bond_left]
            keys_j = h_keys[index_key_bond_left]
            qk_ij = (queries_i * keys_j).sum(-1)
            alpha = scatter_softmax(qk_ij, index_query_bond_left, dim=0)
            values_j = h_values[index_key_bond_left]
            num_attns = len(index_key_bond_left)
            bond_h = scatter_sum((alpha.unsqueeze(-1) * values_j).view(num_attns, -1), 
                                        index_query_bond_left, dim=0, dim_size=bond_h.size(0))
            bond_h = self.layer_norm_atten1(bond_h)

            # right node
            h_queries, h_keys, h_values = self.qkv_right(bond_h)
            queries_i = h_queries[index_query_bond_right]
            keys_j = h_keys[index_key_bond_right]
            qk_ij = (queries_i * keys_j).sum(-1)
            alpha = scatter_softmax(qk_ij, index_query_bond_right, dim=0)
            values_j = h_values[index_key_bond_right]
            num_attns = len(index_key_bond_right)
            bond_h = scatter_sum((alpha.unsqueeze(-1) * values_j).view(num_attns, -1), 
                                        index_query_bond_right, dim=0, dim_size=bond_h.size(0))
            bond_h = self.layer_norm_atten2(bond_h)

        bond_h = self.out_transform(self.act(bond_h))
        return bond_h


class EdgeBlock(Module):
    def __init__(self, edge_dim, node_dim, hidden_dim=None, use_gate=True):
        super().__init__()
        self.use_gate = use_gate
        inter_dim = edge_dim * 2 if hidden_dim is None else hidden_dim

        self.bond_ffn_left = BondFFN(edge_dim, node_dim, inter_dim=inter_dim, use_gate=use_gate)
        self.bond_ffn_right = BondFFN(edge_dim, node_dim, inter_dim=inter_dim, use_gate=use_gate)

        self.node_ffn_left = Linear(node_dim, edge_dim)
        self.node_ffn_right = Linear(node_dim, edge_dim)

        self.self_ffn = Linear(edge_dim, edge_dim)
        self.layer_norm = nn.LayerNorm(edge_dim)
        self.out_transform = Linear(edge_dim, edge_dim)
        self.act = nn.ReLU()

    def forward(self, bond_h, bond_index, node_h, bond_time):
        """
        bond_h: (b, bond_dim)
        bond_index: (2, b)
        node_h: (n, node_dim)
        """
        N = node_h.size(0)
        left_node, right_node = bond_index

        # message from neighbor bonds
        msg_bond_left = self.bond_ffn_left(bond_h, node_h[left_node], bond_time)
        msg_bond_left = scatter_sum(msg_bond_left, right_node, dim=0, dim_size=N)
        msg_bond_left = msg_bond_left[left_node]

        msg_bond_right = self.bond_ffn_right(bond_h, node_h[right_node], bond_time)
        msg_bond_right = scatter_sum(msg_bond_right, left_node, dim=0, dim_size=N)
        msg_bond_right = msg_bond_right[right_node]
        
        bond_h = (
            msg_bond_left + msg_bond_right
            + self.node_ffn_left(node_h[left_node])
            + self.node_ffn_right(node_h[right_node])
            + self.self_ffn(bond_h)
        )
        bond_h = self.layer_norm(bond_h)

        bond_h = self.out_transform(self.act(bond_h))
        return bond_h


class EgnnNet(Module):
    def __init__(self, node_dim, edge_dim, num_blocks, cutoff, use_gate, **kwargs):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.num_blocks = num_blocks
        self.cutoff = cutoff
        self.use_gate = use_gate
        self.kwargs = kwargs

        if 'num_gaussians' not in kwargs:
            num_gaussians = 16
        else:
            num_gaussians = kwargs['num_gaussians']
        if 'start' not in kwargs:
            start = 0
        else:
            start = kwargs['start']
        self.distance_expansion = GaussianSmearing(start=start, stop=cutoff, num_gaussians=num_gaussians)
        if ('update_edge' in kwargs) and (not kwargs['update_edge']):
            self.update_edge = False
            input_edge_dim = num_gaussians
        else:
            self.update_edge = True  # default update edge
            input_edge_dim = edge_dim + num_gaussians
            
        if ('update_pos' in kwargs) and (not kwargs['update_pos']):
            self.update_pos = False
        else:
            self.update_pos = True  # default update pos
        
        # Cross-attention configuration
        self.use_cross_attention = kwargs.get('use_cross_attention', False)
        self.cross_attention_freq = kwargs.get('cross_attention_freq', 2)  # Apply every N blocks
        self.cross_attention_type = kwargs.get('cross_attention_type', 'efficient')  # 'full' or 'efficient'
        cross_attention_heads = kwargs.get('cross_attention_heads', 4)
        cross_attention_dropout = kwargs.get('cross_attention_dropout', 0.1)
        cross_attention_cutoff = kwargs.get('cross_attention_cutoff', cutoff)
        cross_attention_max_neighbors = kwargs.get('cross_attention_max_neighbors', 64)
        
        # node network
        self.node_blocks_with_edge = ModuleList()
        self.edge_embs = ModuleList()
        self.edge_blocks = ModuleList()
        self.pos_blocks = ModuleList()
        self.cross_attention_blocks = ModuleList()
        
        for i in range(num_blocks):
            self.node_blocks_with_edge.append(NodeBlock(
                node_dim=node_dim, edge_dim=edge_dim, hidden_dim=node_dim, use_gate=use_gate,
            ))
            self.edge_embs.append(Linear(input_edge_dim, edge_dim))
            if self.update_edge:
                self.edge_blocks.append(EdgeBlock(
                    edge_dim=edge_dim, node_dim=node_dim, use_gate=use_gate,
                ))
            if self.update_pos:
                self.pos_blocks.append(PosUpdate(
                    node_dim=node_dim, edge_dim=edge_dim, hidden_dim=edge_dim, use_gate=use_gate,
                ))
            
            # Add cross-attention at specified frequency
            if self.use_cross_attention and (i % self.cross_attention_freq == 0):
                if self.cross_attention_type == 'full':
                    self.cross_attention_blocks.append(CrossAttentionBlock(
                        node_dim=node_dim,
                        num_heads=cross_attention_heads,
                        dropout=cross_attention_dropout,
                        use_pos_encoding=True,
                        cutoff=cross_attention_cutoff
                    ))
                else:  # efficient
                    self.cross_attention_blocks.append(EfficientCrossAttentionBlock(
                        node_dim=node_dim,
                        num_heads=cross_attention_heads,
                        dropout=cross_attention_dropout,
                        cutoff=cross_attention_cutoff,
                        max_neighbors=cross_attention_max_neighbors
                    ))
            elif self.use_cross_attention:
                self.cross_attention_blocks.append(None)  # Placeholder

    def forward(self, node_h, node_pos, edge_h, edge_index, node_time, edge_time, ligand_mask, node_batch=None):
        # Get batch information from edge_index if cross-attention is enabled
        if self.use_cross_attention:
            # We need to extract protein and ligand information for cross-attention
            protein_mask = ~ligand_mask
            
            # Derive batch indices for ligand and protein if not provided
            if node_batch is not None:
                ligand_batch = node_batch[ligand_mask]
                protein_batch = node_batch[protein_mask]
            else:
                # Fallback: assume single batch
                ligand_batch = torch.zeros(ligand_mask.sum(), dtype=torch.long, device=node_h.device)
                protein_batch = torch.zeros(protein_mask.sum(), dtype=torch.long, device=node_h.device)
            
        for i in range(self.num_blocks):
            # edge fetures before each block
            if self.update_pos or (i==0):
                edge_h_dist, relative_vec, distance = self._build_edges_dist(node_pos, edge_index)
            if self.update_edge:
                edge_h = torch.cat([edge_h, edge_h_dist], dim=-1)
            else:
                edge_h = edge_h_dist
            edge_h = self.edge_embs[i](edge_h)
                
            # node and edge feature updates
            node_h_with_edge = self.node_blocks_with_edge[i](node_h, edge_index, edge_h, node_time)
            if self.update_edge:
                edge_h = edge_h + self.edge_blocks[i](edge_h, edge_index, node_h, edge_time)
            node_h = node_h + node_h_with_edge
            
            # Cross-attention: ligand attends to protein
            if self.use_cross_attention and self.cross_attention_blocks[i] is not None:
                # Extract ligand and protein nodes
                ligand_h = node_h[ligand_mask]
                ligand_pos_curr = node_pos[ligand_mask]
                protein_h = node_h[protein_mask]
                protein_pos_curr = node_pos[protein_mask]
                
                # Apply cross-attention
                updated_ligand_h = self.cross_attention_blocks[i](
                    ligand_h, ligand_pos_curr, ligand_batch,
                    protein_h, protein_pos_curr, protein_batch
                )
                
                # Update ligand features in full node tensor
                node_h = node_h.clone()
                node_h[ligand_mask] = updated_ligand_h
            
            # pos updates
            if self.update_pos:
                delta_pos = self.pos_blocks[i](node_h, edge_h, edge_index, relative_vec, distance, edge_time)
                node_pos = node_pos + delta_pos * ligand_mask[:, None]
        return node_h, node_pos, edge_h

    def _build_edges_dist(self, pos, edge_index):
        # distance
        relative_vec = pos[edge_index[0]] - pos[edge_index[1]]
        distance = torch.norm(relative_vec, dim=-1, p=2)
        edge_dist = self.distance_expansion(distance)
        return edge_dist, relative_vec, distance


class PosUpdate(Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, use_gate):
        super().__init__()
        self.left_lin_edge = MLP(node_dim, edge_dim, hidden_dim)
        self.right_lin_edge = MLP(node_dim, edge_dim, hidden_dim)
        self.edge_lin = BondFFN(edge_dim, edge_dim, node_dim, use_gate, out_dim=1)

    def forward(self, node_h, edge_h, edge_index, relative_vec, distance, edge_time):
        edge_index_left, edge_index_right = edge_index
        
        left_feat = self.left_lin_edge(node_h[edge_index_left])
        right_feat = self.right_lin_edge(node_h[edge_index_right])
        weight_edge = self.edge_lin(edge_h, left_feat * right_feat, edge_time)
        
        force_edge = weight_edge * relative_vec / distance.unsqueeze(-1) / (distance.unsqueeze(-1) + 1.)
        delta_pos = scatter_sum(force_edge, edge_index_left, dim=0, dim_size=node_h.shape[0])

        return delta_pos

class NodeBondNet(Module):
    def __init__(self, node_dim, edge_dim, bond_dim, key_dim, num_heads, num_blocks, k, cutoff, use_atten, use_gate):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.bond_dim = bond_dim
        self.key_dim = key_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.k = k
        self.cutoff = cutoff
        self.use_atten = use_atten
        self.use_gate = use_gate

        self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=20)
        self.edge_emb = Linear(self.additional_edge_feat+20, edge_dim)
        # node network
        self.lin_node = Linear(node_dim, node_dim)
        self.node_blocks_with_edge = ModuleList()
        self.node_blocks_with_bond = ModuleList()
        self.bond_blocks = ModuleList()
        for _ in range(num_blocks):
            self.node_blocks_with_edge.append(NodeBlock(
                node_dim=node_dim, edge_dim=edge_dim, key_dim=None,
                num_heads=None, use_atten=False, use_gate=use_gate,  # never use atten for edges message becused too many edges
            ))
            self.node_blocks_with_bond.append(NodeBlock(
                node_dim=node_dim, edge_dim=bond_dim, key_dim=None,
                num_heads=None, use_atten=False, use_gate=use_gate,
            ))
            if bond_dim > 0:
                self.bond_blocks.append(BondBlock(
                    bond_dim=bond_dim, node_dim=node_dim, use_gate=use_gate,
                    use_atten=use_atten, key_dim=key_dim, num_heads=num_heads,
                ))

    def forward(self, node_h, node_pos, h_bond, bond_index, batch, is_mol, is_frag, return_edge=False):

        edge_attr, edge_index = self._build_edges(node_pos, batch, is_mol, is_frag)
        for i in range(self.num_blocks):
            # node updates with edges
            node_h_with_edge = self.node_blocks_with_edge[i](node_h, edge_index, edge_attr)
            if self.bond_dim > 0:
                # node updates with bonds
                node_h_with_bond = self.node_blocks_with_bond[i](node_h, bond_index, h_bond)
                # bond updates
                h_bond = h_bond + self.bond_blocks[i](h_bond, bond_index, node_h)
            else:
                node_h_with_bond = 0
            node_h = node_h + self.lin_node(node_h_with_edge + node_h_with_bond)
        if return_edge:
            return {
                'node_h': node_h,
                'h_bond': h_bond,
                'edge_attr': edge_attr,
                'edge_index': edge_index,
            }
        else:
            return {
                'node_h': node_h,
                'h_bond': h_bond,
            }

    @property
    def additional_edge_feat(self):
        return 6

    def _build_edges(self, pos, batch, is_mol, is_frag):
        edge_index = knn_graph(pos, k=self.k, batch=batch, flow='target_to_source') 
        # distance
        distance = torch.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=-1)
        edge_attr = self.distance_expansion(distance)

        # 6-vector represent the two node types (atoms of protein or mol or frag)
        edge_src_feat = is_mol[edge_index[0]].long()
        edge_src_feat = edge_src_feat + 2 * is_frag[edge_index[0]].long()
        edge_dst_feat = is_mol[edge_index[1]].long()
        edge_dst_feat = edge_dst_feat + 2 * is_frag[edge_index[1]].long()
        edge_type_feat = torch.cat([
            F.one_hot(edge_src_feat, num_classes=3),
            F.one_hot(edge_dst_feat, num_classes=3),
        ], axis=-1)

        edge_attr = torch.cat([edge_attr, edge_type_feat], axis=-1)
        edge_attr = self.edge_emb(edge_attr)
        return edge_attr, edge_index

    def _build_bond_atten(self, bond_index):
        left_node, right_node = bond_index
        index_query_bond_left, index_key_bond_left = [], []
        index_query_bond_right, index_key_bond_right = [], []
        for node in torch.unique(left_node):
            ind_connect_left = (left_node == node)
            idx_connect_left = torch.nonzero(ind_connect_left)[:, 0]
            idx_query_bond, idx_key_bond = torch.meshgrid(idx_connect_left, idx_connect_left, indexing='ij')
            index_query_bond_left.append(idx_query_bond.flatten())
            index_key_bond_left.append(idx_key_bond.flatten())

            ind_connect_right = (right_node == node)
            idx_connect_right = torch.nonzero(ind_connect_right)[:, 0]
            idx_query_bond, idx_key_bond = torch.meshgrid(idx_connect_right, idx_connect_right, indexing='ij')
            index_query_bond_right.append(idx_query_bond.flatten())
            index_key_bond_right.append(idx_key_bond.flatten())

        index_query_bond_left = torch.cat(index_query_bond_left)
        index_key_bond_left = torch.cat(index_key_bond_left)
        index_query_bond_right = torch.cat(index_query_bond_right)
        index_key_bond_right = torch.cat(index_key_bond_right)
        return index_query_bond_left, index_key_bond_left, index_query_bond_right, index_key_bond_right



    def _build_bond_atten2(self, bond_index):
        left_node, right_node = bond_index
        index_query_bond_left, index_key_bond_left = [], []
        index_query_bond_right, index_key_bond_right = [], []
        left_node_unique = torch.unique(left_node).cpu().numpy()
        right_node_unique = torch.unique(right_node).cpu().numpy()
        group2node_dict_left = {l:[] for l in left_node_unique}
        group2node_dict_right = {l:[] for l in right_node_unique}
        for i, node in enumerate(left_node.cpu().numpy()):
            group2node_dict_left[node] += [i]
        for i, node in enumerate(right_node.cpu().numpy()):
            group2node_dict_right[node] += [i]
        for node in left_node_unique:
            idx_connect_left = torch.LongTensor(group2node_dict_left[node]).to(bond_index.device)
            idx_query_bond, idx_key_bond = torch.meshgrid(idx_connect_left, idx_connect_left, indexing='ij')
            index_query_bond_left.append(idx_query_bond.flatten())
            index_key_bond_left.append(idx_key_bond.flatten())

            idx_connect_right = torch.LongTensor(group2node_dict_right[node]).to(bond_index.device)
            idx_query_bond, idx_key_bond = torch.meshgrid(idx_connect_right, idx_connect_right, indexing='ij')
            index_query_bond_right.append(idx_query_bond.flatten())
            index_key_bond_right.append(idx_key_bond.flatten())

        index_query_bond_left = torch.cat(index_query_bond_left)
        index_key_bond_left = torch.cat(index_key_bond_left)
        index_query_bond_right = torch.cat(index_query_bond_right)
        index_key_bond_right = torch.cat(index_key_bond_right)
        return index_query_bond_left, index_key_bond_left, index_query_bond_right, index_key_bond_right

class PosPredictor(Module):
    def __init__(self, node_dim, edge_dim, bond_dim, use_gate):
        super().__init__()
        self.left_lin_edge = MLP(node_dim, edge_dim, hidden_dim=edge_dim)
        self.right_lin_edge = MLP(node_dim, edge_dim, hidden_dim=edge_dim)
        self.edge_lin = BondFFN(edge_dim, edge_dim, node_dim, use_gate, out_dim=1)

        self.bond_dim = bond_dim
        if bond_dim > 0:
            self.left_lin_bond = MLP(node_dim, bond_dim, hidden_dim=bond_dim)
            self.right_lin_bond = MLP(node_dim, bond_dim, hidden_dim=bond_dim)
            self.bond_lin = BondFFN(bond_dim, bond_dim, node_dim, use_gate, out_dim=1)

    def forward(self, node_h, node_pos, h_bond, bond_index, edge_h, edge_index, is_frag):
        # 1 pos update through edges
        is_left_frag = is_frag[edge_index[0]]
        edge_index_left, edge_index_right = edge_index[:, is_left_frag]
        
        left_feat = self.left_lin_edge(node_h[edge_index_left])
        right_feat = self.right_lin_edge(node_h[edge_index_right])
        weight_edge = self.edge_lin(edge_h[is_left_frag], left_feat * right_feat)
        force_edge = weight_edge * (node_pos[edge_index_left] - node_pos[edge_index_right])
        delta_pos = scatter_sum(force_edge, edge_index_left, dim=0, dim_size=node_h.shape[0])

        # 2 pos update through bonds
        if self.bond_dim > 0:
            is_left_frag = is_frag[bond_index[0]]
            bond_index_left, bond_index_right = bond_index[:, is_left_frag]

            left_feat = self.left_lin_bond(node_h[bond_index_left])
            right_feat = self.right_lin_bond(node_h[bond_index_right])
            weight_bond = self.bond_lin(h_bond[is_left_frag], left_feat * right_feat)
            force_bond = weight_bond * (node_pos[bond_index_left] - node_pos[bond_index_right])
            delta_pos = delta_pos + scatter_sum(force_bond, bond_index_left, dim=0, dim_size=node_h.shape[0])
        
        pos_update = node_pos + delta_pos / 10.
        return pos_update #TODO: use only frag pos instead of all pos to save memory
