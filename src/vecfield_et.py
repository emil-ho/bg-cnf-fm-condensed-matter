import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MessagePassing

from src.vecfield_pegnn import GaussianSmearing, scatter_sum


class ExpNormalSmearing(nn.Module):
    """Exponential normal radial basis function expansion."""

    def __init__(self, rmin: float = 0.0, rmax: float = 5.0, num_rbf: int = 50, trainable: bool = True):
        super().__init__()
        self.rmin = rmin
        self.rmax = rmax
        self.num_rbf = num_rbf
        self.trainable = trainable

        self.cutoff_fn = CosineCutoff(0, rmax)
        self.alpha = 5.0 / (rmax - rmin)

        means, betas = self._initial_params()
        if trainable:
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("betas", nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self) -> Tuple[Tensor, Tensor]:
        start_value = torch.exp(torch.scalar_tensor(-self.rmax + self.rmin))
        means = torch.linspace(start_value, 1, self.num_rbf)
        betas = torch.tensor([(2 / self.num_rbf * (1 - start_value)) ** -2] * self.num_rbf)
        return means, betas

    def reset_parameters(self):
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist: Tensor) -> Tensor:
        dist = dist.unsqueeze(-1)
        return self.cutoff_fn(dist) * torch.exp(
            -self.betas * (torch.exp(self.alpha * (-dist + self.rmin)) - self.means) ** 2
        )


class ShiftedSoftplus(nn.Module):
    """Shifted softplus activation function."""

    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x: Tensor) -> Tensor:
        return F.softplus(x) - self.shift


class CosineCutoff(nn.Module):
    """Smooth cosine cutoff function for interatomic distances."""

    def __init__(self, rmin: float = 0.0, rmax: float = 5.0):
        super().__init__()
        self.rmin = rmin
        self.rmax = rmax

    def forward(self, distances: Tensor) -> Tensor:
        if self.rmin > 0:
            cutoffs = 0.5 * (
                torch.cos(math.pi * (2 * (distances - self.rmin) / (self.rmax - self.rmin) + 1.0)) + 1.0
            )
            cutoffs = cutoffs * (distances < self.rmax).float()
            cutoffs = cutoffs * (distances > self.rmin).float()
            return cutoffs
        else:
            cutoffs = 0.5 * (torch.cos(distances * math.pi / self.rmax) + 1.0)
            cutoffs = cutoffs * (distances < self.rmax).float()
            return cutoffs


rbf_class_mapping = {"gauss": GaussianSmearing, "expnorm": ExpNormalSmearing}

act_class_mapping = {
    "ssp": ShiftedSoftplus,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}


class CoorsNorm(nn.Module):
    """Coordinate feature normalization layer."""

    def __init__(self, eps: float = 1e-8, scale_init: float = 1.0):
        super().__init__()
        self.eps = eps
        scale = torch.zeros(1).fill_(scale_init)
        self.scale = nn.Parameter(scale)

    def forward(self, coors_feat: Tensor) -> Tensor:
        norm = coors_feat.norm(dim=1, keepdim=True)
        normed_coors = coors_feat / norm.clamp(min=self.eps)
        return normed_coors * self.scale


class GatedEquivariantBlock(nn.Module):
    """Gated Equivariant Block (Schütt et al., 2021) with functorch/vmap-safe vector norms."""

    def __init__(
        self,
        hidden_channels: int,
        out_channels: int,
        intermediate_channels: Optional[int] = None,
        activation: str = "silu",
        scalar_activation: bool = False,
        vector_output: bool = False,
        layer_norm: bool = True,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.vector_output = vector_output
        self.layer_norm = layer_norm

        proj_out_channels = 1 if self.vector_output else out_channels
        if intermediate_channels is None:
            intermediate_channels = hidden_channels

        self.vec1_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.vec2_proj = nn.Linear(hidden_channels, proj_out_channels, bias=False)

        act_class = act_class_mapping[activation]
        final_out_dim = (out_channels + 1) if vector_output else (out_channels * 2)

        if layer_norm:
            self.update_net = nn.Sequential(
                nn.Linear(hidden_channels * 2, intermediate_channels),
                nn.LayerNorm(intermediate_channels),
                act_class(),
                nn.Linear(intermediate_channels, final_out_dim),
            )
        else:
            self.update_net = nn.Sequential(
                nn.Linear(hidden_channels * 2, intermediate_channels),
                act_class(),
                nn.Linear(intermediate_channels, final_out_dim),
            )

        self.act = act_class() if scalar_activation else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.vec1_proj.weight)
        nn.init.xavier_uniform_(self.vec2_proj.weight)
        nn.init.xavier_uniform_(self.update_net[0].weight)
        self.update_net[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.update_net[-1].weight)
        self.update_net[-1].bias.data.fill_(0)

    def forward(self, x: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
        vec1_buffer = self.vec1_proj(v)
        vec2 = self.vec2_proj(v)

        # Vectorized, vmap-safe norm to avoid NaN gradients on zero rows without boolean indexing
        all_nonzero = (vec1_buffer != 0).view(vec1_buffer.size(0), -1).all(dim=1)
        eps = torch.finfo(vec1_buffer.dtype).eps
        gate = all_nonzero.to(vec1_buffer.dtype).unsqueeze(-1)
        eps_row = (1.0 - gate) * eps

        vec1 = torch.sqrt(vec1_buffer.pow(2).sum(dim=-2) + eps_row) * gate
        x = torch.cat([x, vec1], dim=-1)

        if self.vector_output:
            out = self.update_net(x)
            x, v = out[:, : self.out_channels], out[:, self.out_channels :]
        else:
            x, v = torch.split(self.update_net(x), self.out_channels, dim=-1)

        v = v.unsqueeze(1) * vec2
        if self.act is not None:
            x = self.act(x)
        return x, v


class EquivariantVectorOutput(nn.Module):
    """Equivariant vector output head composed of two GatedEquivariantBlocks."""

    def __init__(
        self,
        hidden_channels: int,
        activation: str = "silu",
        layer_norm: bool = True,
    ):
        super().__init__()
        self.output_network = nn.ModuleList(
            [
                GatedEquivariantBlock(
                    hidden_channels,
                    hidden_channels // 2,
                    activation=activation,
                    scalar_activation=True,
                    layer_norm=layer_norm,
                ),
                GatedEquivariantBlock(
                    hidden_channels // 2,
                    hidden_channels,
                    activation=activation,
                    vector_output=True,
                    layer_norm=layer_norm,
                ),
            ]
        )
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.output_network:
            layer.reset_parameters()

    def pre_reduce(self, h: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
        for layer in self.output_network:
            h, v = layer(h, v)
        return h, v.squeeze(-1)


class EquivariantMultiHeadAttention(MessagePassing):
    """Equivariant multi-head attention layer for TorchMD-ET."""

    def __init__(
        self,
        hidden_channels: int,
        num_rbf: int,
        distance_influence: str,
        num_heads: int,
        activation,
        attn_activation: str,
        rmin: float,
        rmax: float,
        node_attr_dim: int = 0,
        time_dim: int = 1,
        qk_norm: bool = False,
        norm_coors: bool = False,
        norm_coors_scale_init: float = 1e-2,
        cutoff_func=CosineCutoff,
        vector_cutoff: bool = False,
    ):
        super().__init__(aggr="add", node_dim=0)
        assert hidden_channels % num_heads == 0, (
            f"hidden_channels ({hidden_channels}) must be evenly divisible by num_heads ({num_heads})"
        )

        self.distance_influence = distance_influence
        self.num_heads = num_heads
        self.hidden_channels = hidden_channels
        self.head_dim = hidden_channels // num_heads

        self.layernorm = nn.LayerNorm(hidden_channels)
        self.node_attr_dim = node_attr_dim
        self.norm_coors = norm_coors
        self.coors_norm = CoorsNorm(scale_init=norm_coors_scale_init) if norm_coors else nn.Identity()
        self.act = activation()
        self.attn_activation = act_class_mapping[attn_activation]()
        self.cutoff = cutoff_func(rmin, rmax)
        self.vector_cutoff = vector_cutoff
        self.qk_norm = qk_norm

        input_channels = hidden_channels + time_dim + (hidden_channels if node_attr_dim > 0 else 0)
        self.mixing_mlp = nn.Sequential(
            nn.Linear(input_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

        if qk_norm:
            self.q_proj = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.LayerNorm(hidden_channels),
            )
            self.k_proj = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.LayerNorm(hidden_channels),
            )
        else:
            self.q_proj = nn.Linear(hidden_channels, hidden_channels)
            self.k_proj = nn.Linear(hidden_channels, hidden_channels)

        self.v_proj = nn.Linear(hidden_channels, hidden_channels * 3)
        self.o_proj = nn.Linear(hidden_channels, hidden_channels * 3)
        self.vec_proj = nn.Linear(hidden_channels, hidden_channels * 3, bias=False)

        self.dk_proj = nn.Linear(num_rbf, hidden_channels)
        self.dv_proj = nn.Linear(num_rbf, hidden_channels * 3)

        self.reset_parameters()

    def reset_parameters(self):
        self.layernorm.reset_parameters()
        if self.qk_norm:
            self.q_proj[0].bias.data.fill_(0)
            nn.init.xavier_uniform_(self.q_proj[0].weight)
            self.k_proj[0].bias.data.fill_(0)
            nn.init.xavier_uniform_(self.k_proj[0].weight)
        else:
            self.q_proj.bias.data.fill_(0)
            nn.init.xavier_uniform_(self.q_proj.weight)
            self.k_proj.bias.data.fill_(0)
            nn.init.xavier_uniform_(self.k_proj.weight)

        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.vec_proj.weight)
        nn.init.xavier_uniform_(self.dk_proj.weight)
        self.dk_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.dv_proj.weight)
        self.dv_proj.bias.data.fill_(0)

    def forward(
        self,
        h: Tensor,
        vec: Tensor,
        edge_index: Tensor,
        r_ij: Tensor,
        f_ij: Tensor,
        d_ij: Tensor,
        t: Tensor,
        node_attr: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        h = self.mixing_mlp(torch.cat([h, t, node_attr], dim=1))
        h = self.layernorm(h)

        q = self.q_proj(h).reshape(-1, self.num_heads, self.head_dim)
        k = self.k_proj(h).reshape(-1, self.num_heads, self.head_dim)
        v = self.v_proj(h).reshape(-1, self.num_heads, self.head_dim * 3)

        vec1, vec2, vec3 = torch.split(self.vec_proj(vec), self.hidden_channels, dim=-1)
        vec = vec.reshape(-1, 3, self.num_heads, self.head_dim)
        vec_dot = (vec1 * vec2).sum(dim=1)

        dk = self.act(self.dk_proj(f_ij)).reshape(-1, self.num_heads, self.head_dim)
        dv = self.act(self.dv_proj(f_ij)).reshape(-1, self.num_heads, self.head_dim * 3)

        h, vec = self.propagate(
            edge_index,
            q=q,
            k=k,
            v=v,
            vec=vec,
            dk=dk,
            dv=dv,
            r_ij=r_ij,
            d_ij=d_ij,
            size=None,
        )
        h = h.reshape(-1, self.hidden_channels)
        vec = vec.reshape(-1, 3, self.hidden_channels)
        vec = self.coors_norm(vec)

        o1, o2, o3 = torch.split(self.o_proj(h), self.hidden_channels, dim=1)
        dvec = vec3 * o1.unsqueeze(1) + vec
        dx = vec_dot * o2 + o3
        return dx, dvec

    def message(
        self,
        q_i: Tensor,
        k_j: Tensor,
        v_j: Tensor,
        vec_j: Tensor,
        dk: Tensor,
        dv: Tensor,
        r_ij: Tensor,
        d_ij: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        attn = (q_i * k_j * dk).sum(dim=-1)
        cutoff = self.cutoff(r_ij).unsqueeze(1)
        attn = self.attn_activation(attn)

        if self.vector_cutoff:
            v_j = v_j * cutoff.unsqueeze(2)
        else:
            attn = attn * cutoff

        v_j = v_j * dv
        x, vec1, vec2 = torch.split(v_j, self.head_dim, dim=2)

        x = x * attn.unsqueeze(2)
        vec = vec_j * vec1.unsqueeze(1) + vec2.unsqueeze(1) * d_ij.unsqueeze(2).unsqueeze(3)
        return x, vec

    def aggregate(
        self,
        features: Tuple[Tensor, Tensor],
        index: Tensor,
        ptr: Optional[Tensor],
        dim_size: Optional[int],
    ) -> Tuple[Tensor, Tensor]:
        x, vec = features
        index = index.to(torch.int64)
        dim_size = int(index.max().item()) + 1 if dim_size is None else dim_size

        x_flat = x.reshape(x.size(0), -1)
        x_agg = scatter_sum(x_flat, index, dim_size).reshape(dim_size, self.num_heads, self.head_dim)

        vec_flat = vec.reshape(vec.size(0), -1)
        vec_agg = scatter_sum(vec_flat, index, dim_size).reshape(dim_size, 3, self.num_heads, self.head_dim)
        return x_agg, vec_agg

    def update(self, inputs: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        return inputs


class ET_dynamics(nn.Module):
    """Periodic Equivariant Transformer vector field (RFM-ET).

    Parameters
    ----------
    manifold : FlatTorus
        Periodic manifold handling box wrapping (`projx`), geodesic vectors (`logmap`),
        and mean-free tangent projection (`proju`).
    eq_pos : torch.Tensor
        Equilibrium lattice positions of shape ``(n_particles, s_dim)``.
    box_edges_unit_cell : torch.Tensor, optional
        Unit cell edge lengths of shape ``(s_dim,)`` for sinusoidal equilibrium position encoding.
    eq_pos_labeling_features : int, optional
        Number of sinusoidal frequency components for equilibrium position labeling.
    n_features_t : int, optional
        Dimension of the linear time embedding, by default 10.
    hidden_channels : int, optional
        Hidden channel dimension, by default 24.
    num_layers : int, optional
        Number of equivariant multi-head attention layers, by default 12.
    num_heads : int, optional
        Number of attention heads, by default 3.
    num_rbf : int, optional
        Number of radial basis functions, by default 16.
    rbf_type : str, optional
        Radial basis function type (``"gauss"`` or ``"expnorm"``), by default ``"gauss"``.
    trainable_rbf : bool, optional
        Whether RBF parameters are trainable, by default False.
    rmin : float, optional
        Lower distance cutoff, by default 0.0.
    rmax : float, optional
        Upper distance cutoff for graph construction and attention, by default 0.55.
    activation : str, optional
        Activation function name, by default ``"silu"``.
    attn_activation : str, optional
        Attention activation function name, by default ``"silu"``.
    distance_influence : str, optional
        Where distance features are applied (``"both"``, ``"keys"``, ``"values"``), by default ``"both"``.
    qk_norm : bool, optional
        Whether to apply LayerNorm to query and key projections, by default True.
    output_layer_norm : bool, optional
        Whether to apply LayerNorm in the gated equivariant output blocks, by default True.
    clip_during_norm : bool, optional
        Whether to clamp small distances when normalizing edge vectors, by default True.
    vector_cutoff : bool, optional
        Whether to apply the cosine cutoff to vector messages as well as attention weights, by default True.
    projx : bool, optional
        Whether to wrap input positions into the box via ``manifold.projx``, by default True.
    proju : bool, optional
        Whether to project output velocities via ``manifold.proju``, by default True.
    """

    def __init__(
        self,
        manifold,
        eq_pos: Tensor,
        box_edges_unit_cell: Optional[Tensor] = None,
        eq_pos_labeling_features: Optional[int] = 1,
        n_features_t: int = 10,
        hidden_channels: int = 24,
        num_layers: int = 12,
        num_heads: int = 3,
        num_rbf: int = 16,
        rbf_type: str = "gauss",
        trainable_rbf: bool = False,
        rmin: float = 0.0,
        rmax: float = 0.55,
        activation: str = "silu",
        attn_activation: str = "silu",
        distance_influence: str = "both",
        qk_norm: bool = True,
        output_layer_norm: bool = True,
        clip_during_norm: bool = True,
        vector_cutoff: bool = True,
        projx: bool = True,
        proju: bool = True,
    ):
        super().__init__()
        self.manifold = manifold
        self.projx = projx
        self.proju = proju
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_rbf = num_rbf
        self.rmin = rmin
        self.rmax = rmax
        self.clip_during_norm = clip_during_norm

        self.register_buffer("eq_pos", torch.as_tensor(eq_pos), persistent=False)

        if n_features_t > 1:
            self.t_embed = nn.Linear(1, n_features_t)
            time_dim = n_features_t
        else:
            self.t_embed = nn.Identity()
            time_dim = 1

        # Sinusoidal unit-cell equilibrium position labeling
        node_attr_dim = 0
        eqp_emb = None
        if eq_pos_labeling_features is not None and eq_pos_labeling_features > 0:
            if box_edges_unit_cell is None:
                raise ValueError("box_edges_unit_cell must be provided when eq_pos_labeling_features > 0.")
            box_uc = box_edges_unit_cell.view(1, self.eq_pos.shape[1], -1).to(self.eq_pos)
            ks = torch.arange(1, eq_pos_labeling_features + 1, dtype=self.eq_pos.dtype, device=self.eq_pos.device)
            freqs = 2 * math.pi * box_uc.pow(-1) * ks.view(1, 1, -1)
            omegas = self.eq_pos.unsqueeze(-1) * freqs
            eqp_emb = torch.cat([torch.sin(omegas), torch.cos(omegas)], dim=-1)
            eqp_emb = eqp_emb.reshape(self.eq_pos.shape[0], -1)
            node_attr_dim = eqp_emb.shape[-1]
        self.register_buffer("eqp_emb", eqp_emb, persistent=False)
        self.node_attr_dim = node_attr_dim

        act_class = act_class_mapping[activation]
        self.embedding = nn.Embedding(1, hidden_channels)
        self.distance_expansion = rbf_class_mapping[rbf_type](rmin, rmax, num_rbf, trainable_rbf)

        if self.node_attr_dim > 0:
            self.node_mlp = nn.Sequential(
                nn.Linear(node_attr_dim, hidden_channels),
                act_class(),
                nn.LayerNorm(hidden_channels),
                nn.Linear(hidden_channels, hidden_channels),
            )

        self.attention_layers = nn.ModuleList(
            [
                EquivariantMultiHeadAttention(
                    hidden_channels=hidden_channels,
                    num_rbf=num_rbf,
                    distance_influence=distance_influence,
                    num_heads=num_heads,
                    activation=act_class,
                    attn_activation=attn_activation,
                    rmin=rmin,
                    rmax=rmax,
                    node_attr_dim=node_attr_dim,
                    time_dim=time_dim,
                    qk_norm=qk_norm,
                    cutoff_func=CosineCutoff,
                    vector_cutoff=vector_cutoff,
                )
                for _ in range(num_layers)
            ]
        )
        self.out_norm = nn.LayerNorm(hidden_channels)
        self.output_model = EquivariantVectorOutput(
            hidden_channels=hidden_channels,
            activation=activation,
            layer_norm=output_layer_norm,
        )

        rows, cols = self._create_edges()
        edge_index = torch.stack([rows, cols], dim=0)
        self.register_buffer("edge_index", edge_index, persistent=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.distance_expansion.reset_parameters()
        for attn in self.attention_layers:
            attn.reset_parameters()
        self.out_norm.reset_parameters()
        self.output_model.reset_parameters()

    def _create_edges(self) -> Tuple[Tensor, Tensor]:
        rows, cols = [], []
        r2max = float(self.rmax) * float(self.rmax)
        for i in range(self.manifold.n_particles):
            xi = self.eq_pos[i]
            for j in range(i + 1, self.manifold.n_particles):
                xj = self.eq_pos[j]
                vij = self.manifold.logmap(xj, xi)
                if torch.dot(vij, vij) <= r2max:
                    rows.append(i)
                    cols.append(j)
                    rows.append(j)
                    cols.append(i)
        return torch.LongTensor(rows), torch.LongTensor(cols)

    def _edge_index_batched(self, n_batch: int) -> Tensor:
        n = self.manifold.n_particles
        offsets = torch.arange(n_batch, device=self.edge_index.device).view(1, -1, 1) * n
        ei_b = self.edge_index.unsqueeze(1) + offsets
        return ei_b.reshape(2, -1)

    def coord2radial(self, edge_index: Tensor, coord: Tensor) -> Tuple[Tensor, Tensor]:
        row, col = edge_index
        x_i_flat = coord[row].reshape(-1)
        x_j_flat = coord[col].reshape(-1)

        diff_flat = self.manifold.logmap(x_i_flat, x_j_flat)
        coord_diff = diff_flat.view(-1, self.manifold.s_dim)
        radial = torch.sum(coord_diff**2, dim=1, keepdim=True)
        norm = torch.sqrt(radial + 1e-8)
        return norm, coord_diff

    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        """Compute tangent vector field at time ``t`` and configuration ``x``."""
        n_batch = x.shape[0]
        N = self.manifold.n_particles
        s_dim = self.manifold.s_dim

        if self.projx:
            x = self.manifold.projx(x)

        t = torch.as_tensor(t, device=x.device, dtype=x.dtype).reshape(-1)
        if t.numel() == 1:
            t = t.expand(n_batch)
        elif t.numel() != n_batch:
            raise ValueError(f"Expected t with 1 or {n_batch} elements, got {t.numel()}.")
        t_feats = self.t_embed(t.view(n_batch, 1)).repeat_interleave(N, dim=0)

        h = self.embedding(torch.zeros(n_batch * N, dtype=torch.long, device=x.device))

        if self.node_attr_dim > 0 and self.eqp_emb is not None:
            node_attr = self.eqp_emb.to(x).repeat(n_batch, 1)
            node_attr = self.node_mlp(node_attr)
        else:
            node_attr = torch.zeros(n_batch * N, 0, device=x.device, dtype=h.dtype)

        edges = self._edge_index_batched(n_batch)
        edge_weight, edge_vec = self.coord2radial(edges, x.reshape(n_batch * N, s_dim))
        edge_attr = self.distance_expansion(edge_weight).squeeze(1)

        if self.clip_during_norm:
            edge_weight = edge_weight.clamp(min=1.0e-2)
        edge_vec = edge_vec / edge_weight

        vec = torch.zeros(n_batch * N, 3, self.hidden_channels, device=h.device, dtype=h.dtype)
        for attn in self.attention_layers:
            dh, dvec = attn(
                h=h,
                vec=vec,
                edge_index=edges,
                r_ij=edge_weight.squeeze(1),
                f_ij=edge_attr,
                d_ij=edge_vec,
                t=t_feats,
                node_attr=node_attr,
            )
            h = h + dh
            vec = vec + dvec

        h = self.out_norm(h)
        _, v = self.output_model.pre_reduce(h, vec)
        v = v.reshape(n_batch, N * s_dim)

        if self.proju:
            v = self.manifold.proju(x, v)
        return v

