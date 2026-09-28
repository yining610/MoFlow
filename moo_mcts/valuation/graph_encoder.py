"""Message-passing encoder over a WorkflowGraph for the f_theta value head."""

import numpy as np
import torch
import torch.nn as nn

from ..workflow import operators as ops
from ..workflow.graph import WorkflowGraph

_CONTROL_KINDS = ("branch", "loop")

def node_feature_dim() -> int:
    """Return the width of the per-node structural feature block (excludes the role embedding)."""
    # operator one-hot | width, log1p(width) | is_control, parallel
    # | control_kind one-hot | normalized position | is_frontier
    return len(ops.names()) + 2 + 2 + len(_CONTROL_KINDS) + 1 + 1


def graph_to_tensors(graph: WorkflowGraph, role_encode) -> dict:
    """Build the per-node feature matrix, role vectors, and bidirectional edge index for a graph.

    - graph: the workflow encoded; sparse node ids are remapped to contiguous [0, N).
    - role_encode: callable mapping a role string to a fixed-length semantic vector
    """
    op_names = ops.names()
    op_index = {name: i for i, name in enumerate(op_names)}
    n_ops = len(op_names)

    nodes = list(graph.nodes)
    n = len(nodes)
    id_to_pos = {node.id: i for i, node in enumerate(nodes)}

    fdim = node_feature_dim()
    feats = np.zeros((n, fdim), dtype=np.float32)
    role_vecs_list = []

    for i, node in enumerate(nodes):
        col = 0
        # operator one-hot
        if node.operator in op_index:
            feats[i, op_index[node.operator]] = 1.0
        col += n_ops
        # width (raw + log1p)
        feats[i, col] = float(node.width)
        feats[i, col + 1] = float(np.log1p(node.width))
        col += 2
        # is_control, parallel
        feats[i, col] = 1.0 if node.is_control else 0.0
        is_parallel = (not node.is_control) and ops.get(node.operator).parallel
        feats[i, col + 1] = 1.0 if is_parallel else 0.0
        col += 2
        # control_kind one-hot ("" -> all zeros)
        for k, kind in enumerate(_CONTROL_KINDS):
            feats[i, col + k] = 1.0 if node.control_kind == kind else 0.0
        col += len(_CONTROL_KINDS)
        # normalized topological position
        feats[i, col] = i / max(1, n - 1)
        col += 1
        # is_frontier
        feats[i, col] = 1.0 if node.id == graph.frontier else 0.0
        col += 1

        role_vecs_list.append(role_encode(node.role))

    role_vecs = (np.stack(role_vecs_list).astype(np.float32, copy=False)
                 if role_vecs_list else np.zeros((0, 0), np.float32))

    # bidirectional edge index (2, 2E); skip edges whose endpoints were dropped
    src, dst = [], []
    for s, d in graph.edges:
        if s in id_to_pos and d in id_to_pos:
            si, di = id_to_pos[s], id_to_pos[d]
            src += [si, di]
            dst += [di, si]
    edge_index = np.asarray([src, dst], dtype=np.int64) if src else np.zeros((2, 0), np.int64)

    frontier_idx = id_to_pos.get(graph.frontier, -1) if graph.frontier is not None else -1
    return {
        "node_feats": feats,
        "role_vecs": role_vecs,
        "edge_index": edge_index,
        "frontier_idx": frontier_idx,
        "num_nodes": n,
    }


def collate(samples: list[dict]):

    feats, rvecs, edges, batch, frontier_pos = [], [], [], [], []
    offset = 0
    role_dim = 0
    for gi, s in enumerate(samples):
        nn = s["num_nodes"]
        if nn == 0:
            frontier_pos.append(-1)
            continue
        feats.append(s["node_feats"])
        rvecs.append(s["role_vecs"])
        role_dim = s["role_vecs"].shape[1]
        if s["edge_index"].shape[1] > 0:
            edges.append(s["edge_index"] + offset)
        batch.append(np.full((nn,), gi, dtype=np.int64))
        fi = s["frontier_idx"]
        frontier_pos.append(offset + fi if fi >= 0 else -1)
        offset += nn

    return {
        "node_feats": np.concatenate(feats, 0) if feats else np.zeros((0, node_feature_dim()), np.float32),
        "role_vecs": np.concatenate(rvecs, 0) if rvecs else np.zeros((0, role_dim), np.float32),
        "edge_index": np.concatenate(edges, 1) if edges else np.zeros((2, 0), np.int64),
        "batch": np.concatenate(batch, 0) if batch else np.zeros((0,), np.int64),
        "frontier_pos": np.asarray(frontier_pos, dtype=np.int64),
        "num_graphs": len(samples),
    }


def build_net(hidden: int, role_input_dim: int, role_dim: int, output_dim: int,
              dropout: float = 0.1, num_layers: int = 2):

    def scatter_mean(src, index, dim_size):
        # mean of src rows grouped by index; groups with no members stay 0.
        out = torch.zeros((dim_size, src.shape[1]), dtype=src.dtype, device=src.device)
        cnt = torch.zeros((dim_size, 1), dtype=src.dtype, device=src.device)
        idx = index.unsqueeze(-1).expand_as(src)
        out = out.scatter_add(0, idx, src)
        cnt = cnt.scatter_add(0, index.unsqueeze(-1), torch.ones_like(index, dtype=src.dtype).unsqueeze(-1))
        return out / cnt.clamp(min=1.0)

    class MPLayer(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.w_self = nn.Linear(dim, dim)
            self.w_neigh = nn.Linear(dim, dim)

        def forward(self, h, edge_index, n):
            if edge_index.shape[1] > 0:
                src, dst = edge_index[0], edge_index[1]
                agg = scatter_mean(h[src], dst, n)
            else:
                agg = torch.zeros_like(h)
            return torch.relu(self.w_self(h) + self.w_neigh(agg))

    class GraphValueNet(nn.Module):
        def __init__(self):
            super().__init__()
            # learnable projection
            self.role_proj = nn.Linear(role_input_dim, role_dim)
            in_dim = node_feature_dim() + role_dim
            self.proj = nn.Linear(in_dim, hidden)
            
            self.mp_layers = nn.ModuleList([MPLayer(hidden) for _ in range(num_layers)])
            self.drop = nn.Dropout(dropout)
            self.head = nn.Sequential(
                nn.Linear(hidden * 2, hidden),
                nn.ReLU(),
                # one output per PREDICTED axis in the chosen set (cost & latency are
                # pinned analytically, never learned). Sigmoid keeps each in [0,1].
                nn.Linear(hidden, output_dim),
                nn.Sigmoid(),
            )

        def forward(self, node_feats, role_vecs, edge_index, batch, frontier_pos, num_graphs):
            n = node_feats.shape[0]
            role_h = torch.relu(self.role_proj(role_vecs))
            h = torch.cat([node_feats, role_h], dim=1)
            h = torch.relu(self.proj(h))
            for i, mp in enumerate(self.mp_layers):
                h = mp(h, edge_index, n)
                if i < len(self.mp_layers) - 1:
                    h = self.drop(h)  # dropout between layers

            # per-graph mean pool
            pooled = scatter_mean(h, batch, num_graphs)
            # frontier node embedding (fall back to the graph's mean pool if none)
            front = torch.zeros((num_graphs, h.shape[1]), dtype=h.dtype, device=h.device)
            for g in range(num_graphs):
                fp = int(frontier_pos[g].item())
                front[g] = h[fp] if fp >= 0 else pooled[g]
            graph_emb = torch.cat([pooled, front], dim=1)
            return self.head(graph_emb)

    return GraphValueNet()
