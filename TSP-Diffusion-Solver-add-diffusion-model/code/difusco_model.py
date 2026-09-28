"""Compact dense categorical DIFUSCO-style TSP model.

The diffusion and gated edge-GNN follow Edward Sun and Yiming Yang's DIFUSCO
design (NeurIPS 2023), with a self-contained PyTorch implementation so Windows
users do not need PyTorch Geometric, torch-sparse, or the Cython decoder.
"""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class GatedEdgeLayer(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.node_self = nn.Linear(hidden, hidden)
        self.node_message = nn.Linear(hidden, hidden)
        self.edge_src = nn.Linear(hidden, hidden)
        self.edge_dst = nn.Linear(hidden, hidden)
        self.edge_self = nn.Linear(hidden, hidden)
        self.node_norm = nn.LayerNorm(hidden)
        self.edge_norm = nn.LayerNorm(hidden)
        self.edge_update = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))

    def forward(self, nodes, edges):
        src = self.edge_src(nodes).unsqueeze(2)
        dst = self.edge_dst(nodes).unsqueeze(1)
        edge = F.silu(src + dst + self.edge_self(edges))
        gate = torch.sigmoid(edge)
        messages = self.node_message(nodes).unsqueeze(1) * gate
        nodes = self.node_norm(nodes + self.node_self(nodes) + messages.sum(dim=2) / max(1, nodes.shape[1]))
        edges = self.edge_norm(edges + self.edge_update(edge))
        return F.silu(nodes), F.silu(edges)


class DIFUSCOTSP(nn.Module):
    """Denoise a noisy (undirected) tour-edge matrix conditioned on city points."""
    def __init__(self, hidden=16, layers=2):
        super().__init__()
        self.node_in = nn.Sequential(nn.Linear(2, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.edge_in = nn.Sequential(nn.Linear(2, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.time_in = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.layers = nn.ModuleList([GatedEdgeLayer(hidden) for _ in range(layers)])
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.SiLU(), nn.Linear(hidden, 2))

    def forward(self, points, noisy_edges, timesteps):
        points = points - points.mean(dim=1, keepdim=True)
        scale = points.std(dim=1, keepdim=True).clamp_min(1e-6)
        nodes = self.node_in(points / scale)
        distances = torch.cdist(points, points)
        distances = distances / distances.mean(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        edge_input = torch.stack((noisy_edges.float(), distances), dim=-1)
        edges = self.edge_in(edge_input)
        half = self.time_in.__getitem__(0).in_features // 2
        freq = torch.exp(torch.arange(half, device=points.device) * (-math.log(10000.0) / max(1, half - 1)))
        phase = timesteps.float().unsqueeze(-1) * freq.unsqueeze(0)
        time = torch.cat((phase.sin(), phase.cos()), dim=-1)
        if time.shape[-1] < self.time_in[0].in_features:
            time = F.pad(time, (0, self.time_in[0].in_features - time.shape[-1]))
        time = self.time_in(time)
        for layer in self.layers:
            nodes, edges = layer(nodes, edges)
            edges = edges + time[:, None, None, :]
        logits = self.out(edges)
        logits = (logits + logits.transpose(1, 2)) * 0.5
        return logits.permute(0, 3, 1, 2).contiguous()


class CategoricalEdgeDiffusion:
    """Uniform categorical corruption and DIFUSCO-style x0 posterior sampling."""
    def __init__(self, steps=32, device="cpu"):
        self.steps = steps
        beta = np.linspace(1e-4, 0.02, steps, dtype=np.float64)
        eye = np.eye(2)
        self.q = np.stack([(1 - b) * eye + (b / 2) * np.ones((2, 2)) for b in beta])
        qbar = [eye]
        for matrix in self.q:
            qbar.append(qbar[-1] @ matrix)
        self.qbar = torch.tensor(np.stack(qbar), dtype=torch.float32, device=device)

    def corrupt(self, clean):
        t = torch.randint(1, self.steps + 1, (clean.shape[0],), device=clean.device)
        matrix = self.qbar[t]
        probs = matrix[:, None, None, :, :].expand(-1, clean.shape[1], clean.shape[2], -1, -1)
        probs = probs.gather(-2, clean[..., None, None].expand(-1, -1, -1, 1, 2)).squeeze(-2)
        noisy = torch.bernoulli(probs[..., 1]).long()
        return noisy, t

    @torch.no_grad()
    def sample(self, model, points, inference_steps=8):
        b, n, _ = points.shape
        state = torch.randint(0, 2, (b, n, n), device=points.device)
        schedule = np.linspace(self.steps, 0, min(inference_steps, self.steps) + 1).round().astype(int)
        schedule = np.unique(schedule)[::-1]
        for current, target in zip(schedule[:-1], schedule[1:]):
            t = torch.full((b,), int(current), device=points.device)
            x0 = model(points, state, t).softmax(dim=1).permute(0, 2, 3, 1)
            qbar_now = self.qbar[current]
            qbar_target = self.qbar[target]
            q_step = torch.linalg.solve(qbar_target.T, qbar_now.T).T
            q_step = q_step.clamp_min(0)
            q_step = q_step / q_step.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            onehot = F.one_hot(state, 2).float()
            likelihood = torch.einsum("...i,ki->...k", onehot, q_step)
            denom = torch.einsum("ci,...i->...c", qbar_now, onehot).clamp_min(1e-12)
            posterior = qbar_target[None, None, None, :, :] * likelihood[..., None, :] / denom[..., :, None]
            probs = (x0[..., :, None] * posterior).sum(dim=-2).clamp_min(0)
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            if target == 0:
                state = probs[..., 1]
            else:
                state = torch.bernoulli(probs[..., 1]).long()
        state = (state + state.transpose(1, 2)) * 0.5
        diagonal = torch.arange(n, device=points.device)
        state[:, diagonal, diagonal] = 0
        return state.clamp(0, 1)
