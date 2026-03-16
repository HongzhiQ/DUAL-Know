import os
import sys
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import (
    DGHMA_NUM_LAYERS,
    DGHMA_NUM_HEADS,
    DGHMA_HIDDEN_DIM,
    DGHMA_DROPOUT,
    EMBEDDING_DIM,
)


REV_PREFIX = "rev_"


def _has_scatter_reduce() -> bool:
    return hasattr(torch.Tensor, "scatter_reduce_")


def segment_softmax_per_head(
    logits: torch.Tensor,
    dst_index: torch.Tensor,
    num_nodes: int,
    eps: float = 1e-9,
) -> torch.Tensor:

    device = logits.device
    E, H = logits.shape

    if E == 0:
        return logits

    if _has_scatter_reduce():
        max_per_dst = torch.full(
            (num_nodes, H),
            fill_value=-1e30,
            dtype=logits.dtype,
            device=device,
        )
        index_expand = dst_index.unsqueeze(-1).expand(-1, H)
        max_per_dst.scatter_reduce_(
            0, index_expand, logits, reduce="amax", include_self=True
        )
        shifted = logits - max_per_dst[dst_index]
        exp_logits = torch.exp(shifted)

        denom = torch.zeros((num_nodes, H), dtype=logits.dtype, device=device)
        denom.index_add_(0, dst_index, exp_logits)
        softmax = exp_logits / (denom[dst_index] + eps)
        return softmax


    outs = []
    for h in range(H):
        x = logits[:, h]
        max_per_dst = torch.full(
            (num_nodes,),
            fill_value=-1e30,
            dtype=x.dtype,
            device=device,
        )
        max_per_dst = max_per_dst.scatter_reduce(
            0, dst_index, x, reduce="amax", include_self=True
        )
        shifted = x - max_per_dst[dst_index]
        exp_x = torch.exp(shifted)
        denom = torch.zeros((num_nodes,), dtype=x.dtype, device=device)
        denom.index_add_(0, dst_index, exp_x)
        outs.append(exp_x / (denom[dst_index] + eps))
    return torch.stack(outs, dim=-1)


class DGHMALayer(nn.Module):


    def __init__(self, input_dim, hidden_dim, num_heads, type_list, relation_list, dropout=0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0,

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** 0.5


        self.W_Q = nn.ModuleDict({t: nn.Linear(hidden_dim, hidden_dim, bias=False) for t in type_list})
        self.W_K = nn.ModuleDict({t: nn.Linear(hidden_dim, hidden_dim, bias=False) for t in type_list})
        self.W_V = nn.ModuleDict({t: nn.Linear(hidden_dim, hidden_dim, bias=False) for t in type_list})
        self.W_Q_def = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_K_def = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_V_def = nn.Linear(hidden_dim, hidden_dim, bias=False)


        self.rel_emb = nn.Embedding(len(relation_list) + 1, num_heads * self.head_dim)
        self.rel2idx = {r: i for i, r in enumerate(relation_list)}

        type_dim = hidden_dim // 4
        self.type_emb = nn.Embedding(len(type_list) + 1, type_dim)
        self.type2idx = {t: i for i, t in enumerate(type_list)}


        self.rel_type_emb = nn.Embedding(len(relation_list) + 1, type_dim)


        self.gate = nn.Linear(type_dim * 3 + hidden_dim, num_heads)


        self.static_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.Ws = nn.Linear(hidden_dim, hidden_dim, bias=False)


        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.ln = nn.LayerNorm(hidden_dim)


        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

    def _tidx(self, t):
        return self.type2idx.get(t, len(self.type2idx))

    def _ridx(self, r):
        return self.rel2idx.get(r, len(self.rel2idx))

    def _batch_type_specific_proj(
        self,
        x: torch.Tensor,
        type_names: List[str],
        which: str,
    ) -> torch.Tensor:

        if which == "Q":
            mod_dict, mod_def = self.W_Q, self.W_Q_def
        elif which == "K":
            mod_dict, mod_def = self.W_K, self.W_K_def
        elif which == "V":
            mod_dict, mod_def = self.W_V, self.W_V_def
        else:
            raise ValueError(f"Unsupported projection type: {which}")

        N = x.size(0)
        out = x.new_zeros((N, self.hidden_dim))


        type_to_indices: Dict[str, List[int]] = {}
        for idx, t in enumerate(type_names):
            type_to_indices.setdefault(t, []).append(idx)

        for t, idxs in type_to_indices.items():
            idx_tensor = torch.tensor(idxs, dtype=torch.long, device=x.device)
            W = mod_dict[t] if t in mod_dict else mod_def
            out[idx_tensor] = W(x[idx_tensor])

        return out.view(N, self.num_heads, self.head_dim)

    def forward(self, feats, types, static, query, adj):

        device = query.device


        node_ids = list(feats.keys())
        num_nodes = len(node_ids)
        if num_nodes == 0:
            return feats, query

        nid2idx = {nid: i for i, nid in enumerate(node_ids)}

        x = torch.stack([feats[nid] for nid in node_ids], dim=0)
        x_static = torch.stack([static[nid] for nid in node_ids], dim=0)
        type_names = [types.get(nid, "未知") for nid in node_ids]

        node_type_idx = torch.tensor(
            [self._tidx(t) for t in type_names],
            dtype=torch.long,
            device=device,
        )

        Q = self._batch_type_specific_proj(x, type_names, "Q")
        K = self._batch_type_specific_proj(x, type_names, "K")
        V = self._batch_type_specific_proj(x, type_names, "V")


        dst_list = []
        src_list = []
        rel_list = []

        for dst_nid, neighbors in adj.items():
            if dst_nid not in nid2idx:
                continue
            dst_idx = nid2idx[dst_nid]
            for src_nid, rel in neighbors:
                if src_nid not in nid2idx:
                    continue
                src_idx = nid2idx[src_nid]
                dst_list.append(dst_idx)
                src_list.append(src_idx)
                rel_list.append(self._ridx(rel))


        if len(dst_list) == 0:
            updated_x = self.ln(x + self.ffn(x))
            sc = torch.matmul(updated_x, query.unsqueeze(-1)).squeeze(-1)
            w = F.softmax(sc, dim=0)
            ctx = (w.unsqueeze(-1) * updated_x).sum(dim=0)
            query = self.gru(ctx.unsqueeze(0), query.unsqueeze(0)).squeeze(0)

            updated = {nid: updated_x[i] for i, nid in enumerate(node_ids)}
            return updated, query

        dst_index = torch.tensor(dst_list, dtype=torch.long, device=device)  # (E,)
        src_index = torch.tensor(src_list, dtype=torch.long, device=device)  # (E,)
        rel_index = torch.tensor(rel_list, dtype=torch.long, device=device)  # (E,)
        num_edges = dst_index.numel()


        Qi = Q[dst_index]
        Kj = K[src_index]
        Vj = V[src_index]

        Er = self.rel_emb(rel_index).view(num_edges, self.num_heads, self.head_dim)


        base_logits = torch.sum(Qi * (Kj + Er), dim=-1) / self.scale

        ti_emb = self.type_emb(node_type_idx[dst_index])
        tj_emb = self.type_emb(node_type_idx[src_index])
        tr_emb = self.rel_type_emb(rel_index)
        query_expand = query.unsqueeze(0).expand(num_edges, -1)

        gate_input = torch.cat([ti_emb, tj_emb, tr_emb, query_expand], dim=-1)
        gate_vals = torch.sigmoid(self.gate(gate_input))


        attn_base = segment_softmax_per_head(base_logits, dst_index, num_nodes)
        attn = attn_base * gate_vals

        attn_denom = torch.zeros((num_nodes, self.num_heads), dtype=attn.dtype, device=device)
        attn_denom.index_add_(0, dst_index, attn)
        attn = attn / (attn_denom[dst_index] + 1e-9)


        weighted_vals = attn.unsqueeze(-1) * Vj
        agg = torch.zeros((num_nodes, self.num_heads, self.head_dim), dtype=weighted_vals.dtype, device=device)
        agg.index_add_(0, dst_index, weighted_vals)
        agg = agg.reshape(num_nodes, self.hidden_dim)

        agg = agg + self.Ws(self.static_ffn(x_static))

        updated_x = self.ln(x + self.ffn(agg))


        sc = torch.matmul(updated_x, query.unsqueeze(-1)).squeeze(-1)
        w = F.softmax(sc, dim=0)
        ctx = (w.unsqueeze(-1) * updated_x).sum(dim=0)
        query = self.gru(ctx.unsqueeze(0), query.unsqueeze(0)).squeeze(0)

        updated = {nid: updated_x[i] for i, nid in enumerate(node_ids)}
        return updated, query


class DGHMAModule(nn.Module):


    def __init__(
        self,
        input_dim=EMBEDDING_DIM,
        hidden_dim=DGHMA_HIDDEN_DIM,
        num_layers=DGHMA_NUM_LAYERS,
        num_heads=DGHMA_NUM_HEADS,
        type_list=None,
        relation_list=None,
        dropout=DGHMA_DROPOUT,
    ):
        super().__init__()
        type_list = type_list or []
        relation_list = relation_list or []


        full_relation_list = list(relation_list)

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.query_proj = nn.Linear(input_dim, hidden_dim)

        self.layers = nn.ModuleList([
            DGHMALayer(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                type_list=type_list,
                relation_list=full_relation_list,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        subgraph: nx.DiGraph,
        node_embeddings: Dict[str, np.ndarray],
        query_embedding: np.ndarray,
        entities: Dict[str, dict],
    ):
        device = next(self.parameters()).device

        feats, static, types = {}, {}, {}
        query_dim = query_embedding.shape[0]

        for nid in subgraph.nodes():
            emb_np = node_embeddings.get(nid, np.zeros(query_dim, dtype=np.float32))
            emb = torch.tensor(emb_np, dtype=torch.float32, device=device)

            h0 = self.input_proj(emb)
            feats[nid] = h0
            static[nid] = h0.clone()
            types[nid] = (
                entities.get(nid, {}).get("type")
                or subgraph.nodes[nid].get("type", "未知")
                or "未知"
            )

        query = self.query_proj(
            torch.tensor(query_embedding, dtype=torch.float32, device=device)
        )



        adj = {nid: [] for nid in subgraph.nodes()}
        for s, t, d in subgraph.edges(data=True):
            r = d.get("relation", "相关")
            adj[t].append((s, r))


        for layer in self.layers:
            feats, query = layer(feats, types, static, query, adj)

        return feats, query