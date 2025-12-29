import torch
import torch.nn as nn
from torch_geometric.utils import scatter
import math
from torch import Tensor

class MLP_dynamics(nn.Module):
    def __init__(self, dim: int = 2, h: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, dim))
    
    def forward(self, t: Tensor, x_t: Tensor) -> Tensor:
        return self.net(torch.cat((t, x_t), -1))
    
    def step(self, x_t: Tensor, t_start: Tensor, t_end: Tensor) -> Tensor:
        t_start = t_start.view(1, 1).expand(x_t.shape[0], 1)
        
        return x_t + (t_end - t_start) * self(t=t_start + (t_end - t_start) / 2, x_t= x_t + self(x_t=x_t, t=t_start) * (t_end - t_start) / 2)


def scatter_sum(src, index, dim_size):
    out = torch.zeros(dim_size, src.size(-1), device=src.device, dtype=src.dtype)
    # Out-of-place op to stay compatible with functorch transforms (vmap/jacrev).
    return out.index_add(0, index, src)


class ExponentialTimeEmbedding(nn.Module):
    """Sinusoidal time embedding with exponentially spaced frequencies."""
    def __init__(self, emb_dim: int, max_period: float = 100.0, max_freq=32.0):
        super().__init__()
        if emb_dim % 2 != 0 or emb_dim <= 0:
            raise ValueError("Exponential time embedding dimension must be a positive even number.")
        half_dim = emb_dim // 2
        exponent = torch.arange(half_dim, dtype=torch.float32)
        scale = -math.log(max_period) / max(half_dim - 1, 1)
        inv_freq = torch.exp(exponent * scale) * max_freq
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.view(-1, 1)
        freqs = self.inv_freq.to(t.dtype)
        args = t * freqs * math.pi
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return emb


class GaussianSmearing(nn.Module):
    def __init__(self, rmin=0.0, rmax=5.0, num_rbf=50, trainable=True):
        super().__init__()
        self.rmin = rmin
        self.rmax = rmax
        self.num_rbf = num_rbf
        self.trainable = trainable

        offset, coeff = self._initial_params()
        if trainable:
            self.register_parameter("coeff", nn.Parameter(coeff))
            self.register_parameter("offset", nn.Parameter(offset))
        else:
            self.register_buffer("coeff", coeff)
            self.register_buffer("offset", offset)

    def _initial_params(self):
        offset = torch.linspace(self.rmin, self.rmax, self.num_rbf)
        coeff = -0.5 / (offset[1] - offset[0]) ** 2
        return offset, coeff

    def reset_parameters(self):
        offset, coeff = self._initial_params()
        self.offset.data.copy_(offset)
        self.coeff.data.copy_(coeff)

    def forward(self, dist):
        dist = dist.unsqueeze(-1) - self.offset
        return torch.exp(self.coeff * torch.pow(dist, 2))


class PE_GCL(nn.Module):
    def __init__(
        self,
        box_edges,
        input_nf,
        output_nf,
        hidden_nf,
        edges_in_d=0,
        act_fn=nn.SiLU(),
        agg="sum",
        gain=0.001,
        norm_diff=True,
        distance_expansion=nn.Identity(),
    ):
        super().__init__()
        self.input_nf = input_nf
        self.output_nf = output_nf
        self.hidden_nf = hidden_nf
        self.agg = agg
        self.norm_diff = norm_diff
        self.register_buffer('box_edges', box_edges)

        self.distance_expansion = distance_expansion
        if isinstance(self.distance_expansion, nn.Identity):
            edge_coords_nf = 1  # r_ij^2
        else:
            edge_coords_nf = self.distance_expansion.num_rbf
            
        edge_in = 2 * input_nf + edge_coords_nf + edges_in_d

        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_in, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf),
        )

        layer = nn.Linear(hidden_nf, 1, bias=False)
        nn.init.xavier_uniform_(layer.weight, gain=gain)

        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            layer,
        )

    def forward(self, h, x, edge_index, edge_attr=None, node_attr=None):
        row, col = edge_index  # [E]

        # geometry
        radial2, coord_diff = self.coord2radial(edge_index, x)  # [E,1], [E,s_dim]
        radial = self.distance_expansion(torch.sqrt(radial2))
        radial = radial.view(radial.size(0), -1)
        
        # build edge input
        if edge_attr is None:
            edge_in = torch.cat([h[row], h[col], radial], dim=-1)
        else:
            edge_in = torch.cat([h[row], h[col], radial, edge_attr], dim=-1)

        edge_feat = self.edge_mlp(edge_in)  # [E, hidden_nf]

        # node feature update
        # msg = scatter(edge_feat, row, dim=0, dim_size=h.size(0), reduce=self.agg)
        msg = scatter_sum(edge_feat, row, h.size(0))
        if node_attr is not None:
            node_in = torch.cat([h, msg, node_attr], dim=-1)
        else:
            node_in = torch.cat([h, msg], dim=-1)
        h_out = self.node_mlp(node_in)

        # coordinate update
        scale = self.coord_mlp(edge_feat)    # [E, 1]

        trans = coord_diff * scale
        # coord_update = scatter(trans, row, dim=0, dim_size=x.size(0), reduce=self.agg)
        coord_update = scatter_sum(trans, row, x.size(0))
        x_out = x + coord_update

        return h_out, x_out
    
    def coord2radial(self, edge_index, coord):
        """Compute radial features and normalized coordinate differences.
        
        Parameters
        ----------
        edge_index : LongTensor, shape (2, n_edges)
            Edge indices [source, target]
        coord : Tensor, shape (n_nodes, s_dim)
            Node coordinates
            
        Returns
        -------
        radial : Tensor, shape (n_edges, 1)
            Squared distances
        coord_diff : Tensor, shape (n_edges, s_dim)
            Normalized coordinate differences
        """
        coord_diff =  coord[edge_index[0]] - coord[edge_index[1]]
        coord_diff =  coord_diff - self.box_edges * torch.round(coord_diff / self.box_edges)
        
        # Compute squared distance
        radial = torch.sum(coord_diff ** 2, dim=1, keepdim=True)  # (n_edges, 1)
        
        # Normalize coordinate differences
        norm = torch.sqrt(radial + 1e-8)
        coord_diff = coord_diff / (norm + 1)
        
        return radial, coord_diff


class PEGNN(nn.Module):
    """Periodic EGNN using PyTorch Geometric.
    
    Same as EGNN, but with PE_GCL layers instead of E_GCL.
    """
    def __init__(
        self,
        in_node_nf: int,
        in_edge_nf: int,
        hidden_nf: int,
        box_edges = torch.tensor([1.,1.,1.]),
        act_fn=nn.SiLU(),
        n_layers: int = 4,
        norm_diff: bool = True,
        out_node_nf: int = None,
        coords_range: float = 15.0,
        agg: str = "sum",
        gain: float = 0.001,
        distance_expansion=nn.Identity(),
    ):
        super().__init__()
        
        if out_node_nf is None:
            out_node_nf = in_node_nf
        
        self.box_edges = box_edges
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range) / self.n_layers
        
        # if agg == 'mean':
        #     self.coords_range_layer = self.coords_range_layer * box_length.n_particles
        
        # Input embedding
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        
        # GCL layers
        self.gcl_layers = nn.ModuleList()
        for i in range(n_layers):
            self.gcl_layers.append(
                PE_GCL(
                    box_edges=box_edges,
                    input_nf=self.hidden_nf,
                    output_nf=self.hidden_nf,
                    hidden_nf=self.hidden_nf,
                    edges_in_d=in_edge_nf,
                    act_fn=act_fn,
                    norm_diff=norm_diff,
                    agg=agg,
                    gain=gain,
                    distance_expansion=distance_expansion,
                )
            )

        # Output embedding
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
    
    def forward(self, h, x, edge_index, edge_attr=None):
        h = self.embedding(h)
        
        for gcl in self.gcl_layers:
            h, x = gcl(h, x, edge_index, edge_attr=edge_attr)
        
        h = self.embedding_out(h)
        
        return h, x


class PEGNN_dynamics(nn.Module):
    """Periodic EGNN dynamics model for a cubic box with box edges in x,y,z direction.
    
    Like the original EGNN by Satorras, but distance calculations are adapted to the box_edges.
    Additionally, some tweaks are added, as e.g. embeddings.
    Graph is built once in the beginning. Either fully connected or with cutoff based on 
    supplied equilibrium positions.

    eq_pos_embed: none or tuple(float, int) with first number being the unit cell box length and second number being the number of frequencies
    """
    def __init__(
        self,
        eq_pos: torch.Tensor,
        box_edges = torch.tensor([1.,1.,1.]),
        rmin: float = 0.0,
        rmax: float = 0.5,
        n_features_t = 32,
        box_edges_unit_cell = None,
        eq_pos_labeling_features = None,
        rbf_features = 0,
        rbf_trainable = False,
        hidden_nf: int = 64,
        act_fn=nn.SiLU(),
        n_layers: int = 4,
        agg: str = "sum",
        gain: float = 0.001,
    ):
        super().__init__()

        self.register_buffer("eq_pos", eq_pos, persistent=False)
        self.register_buffer("box_edges", box_edges, persistent=False)

        if n_features_t == 0:
            self.t_embed = nn.Identity()
        else:
            self.t_embed = ExponentialTimeEmbedding(n_features_t)

        n_features_eqp = 0
        eqp_emb = None
        if eq_pos_labeling_features is not None:
            box_edges_unit_cell = box_edges_unit_cell.view(1, eq_pos.shape[1], -1).to(self.eq_pos)
            ks = torch.arange(1, eq_pos_labeling_features + 1, dtype=self.eq_pos.dtype, device=self.eq_pos.device)
            freqs = 2 * math.pi * box_edges_unit_cell.pow(-1) * ks.view(1, 1, -1)
            omegas = self.eq_pos.unsqueeze(-1) * freqs
            eqp_emb = torch.cat([torch.sin(omegas), torch.cos(omegas)], dim=-1)
            eqp_emb = eqp_emb.reshape(self.eq_pos.shape[0], -1)
            n_features_eqp = eqp_emb.shape[-1]
        self.register_buffer("eqp_emb", eqp_emb, persistent=False)

        self.rmin = rmin
        self.rmax = rmax
        if rbf_features == 0:
            self.rbf = nn.Identity()
        else:
            self.rbf = GaussianSmearing(
                rmin=rmin, rmax=rmax, num_rbf=rbf_features, trainable=rbf_trainable
            )

        self.pegnn = PEGNN(
            box_edges=self.box_edges,
            in_node_nf=n_features_t+n_features_eqp,
            in_edge_nf=1,
            hidden_nf=hidden_nf,
            act_fn=act_fn,
            n_layers=n_layers,
            agg=agg,
            gain=gain,
            distance_expansion=self.rbf,
        )

        # Build edge index
        rows, cols = self._create_edges()
        edge_index = torch.stack([rows, cols], dim=0)  # (2, n_edges)
        self.register_buffer("edge_index", edge_index, persistent=False)

    def forward(self, t, x):
        """
        Parameters
        ----------
        t : Tensor, shape (B,) or (B, 1)
            Time values
        x : Tensor, shape (B, n_particles * s_dim)
            Flat configurations
            
        Returns
        -------
        vel : Tensor, shape (B, n_particles * s_dim)
            Flat velocity (tangent vector)
        """
        n_batch = x.shape[0]
        x = torch.remainder(x.reshape(n_batch, self.eq_pos.shape[0], self.eq_pos.shape[1]), self.box_edges)

        # Embed time and broadcast to every node (B*P, d_t)
        # time comes either as batch (training) or as 0-dim tensor (ODE integration)
        t = torch.as_tensor(t, device=x.device, dtype=x.dtype).reshape(-1)
        if t.numel() == 1:
            t = t.expand(n_batch)
        elif t.numel() != n_batch:
            raise ValueError(f"Expected t with 1 or {n_batch} elements, got {t.numel()}.")
        t = t.view(n_batch, 1)

        t_feats = self.t_embed(t)
        h = t_feats.repeat_interleave(self.eq_pos.shape[0], dim=0)  # (B*P, d_t)
        if self.eqp_emb is not None:
            eqp = self.eqp_emb.unsqueeze(0).expand(n_batch, -1, -1)
            eqp = eqp.reshape(n_batch * self.eq_pos.shape[0], -1)
            h = torch.cat([h, eqp], dim=-1)

        # Batched edge indices
        edge_index = self._edge_index_batched(n_batch)  # (2, B*n_edges)

        # Reshape coordinates
        x0 = x.reshape(n_batch * self.eq_pos.shape[0], self.eq_pos.shape[1]).clone()

        edge_attr = None
        # Compute edge attributes (squared distances using box_edges)
        coord_diff =  x0[edge_index[0]] - x0[edge_index[1]]
        coord_diff =  coord_diff - self.box_edges * torch.round(coord_diff / self.box_edges)
        edge_attr = torch.sum(coord_diff ** 2, dim=1, keepdim=True)  # (B*n_edges, 1)
        # edge_attr = self.rbf(edge_attr)

        # Forward pass through PEGNN
        _, x_final = self.pegnn(h, x0, edge_index, edge_attr=edge_attr)

        # Compute velocity (in tangent space, which is Euclidean)
        vel = x_final - x0
        vel = vel.view(n_batch, -1)

        return vel

    def _create_edges(self):
        """Create edge list (either fully connected or with cutoff)."""
        rows, cols = [], []
        for i in range(self.eq_pos.shape[0]):
            xi = self.eq_pos[i]
            for j in range(i + 1, self.eq_pos.shape[0]):
                xj = self.eq_pos[j]
                coord_diff = xi - xj
                coord_diff = coord_diff - self.box_edges * torch.round(coord_diff / self.box_edges)
                radial = torch.sqrt(torch.sum(coord_diff ** 2))
                if self.rmin <= radial.item() <= self.rmax:
                    rows.append(i)
                    cols.append(j)
                    rows.append(j)
                    cols.append(i)

        return torch.LongTensor(rows), torch.LongTensor(cols)

    def _edge_index_batched(self, n_batch: int):
        """Create batched edge indices for multiple graphs.
        
        Parameters
        ----------
        n_batch : int
            Batch size
            
        Returns
        -------
        edge_index_batched : LongTensor, shape (2, B*n_edges)
            Batched edge indices
        """
        n = self.eq_pos.shape[0]
        offsets = torch.arange(n_batch, device=self.edge_index.device).view(1, -1, 1) * n
        ei_b = self.edge_index.unsqueeze(1) + offsets  # (2, B, n_edges)
        return ei_b.reshape(2, -1)


if __name__ == '__main__':
    eq_pos = torch.tensor(
        [   # The first 4 atoms are the same as for FCC.
            [0.0, 0.0, 0.0],
            [0.0, 2.0, 2.0],
            [2.0, 0.0, 2.0],
            [2.0, 2.0, 0.0],
            # The additional are located within the unit cell.
            [3.0, 3.0, 3.0],
            [3.0, 1.0, 1.0],
            [1.0, 3.0, 1.0],
            [1.0, 1.0, 3.0],
        ]) / 4

    torch.manual_seed(123)
    vf = PEGNN_dynamics(
        box_edges=torch.tensor([1,1,1]),
        hidden_nf=32,
        n_layers=3,
        rmax=0.48,
        eq_pos=eq_pos,
        # now extra stuff
        box_edges_unit_cell = torch.tensor([0.5, 0.5, 0.5]),
        eq_pos_labeling_features = 1,
        rbf_features = 10,
        rbf_trainable = False,

    )

    pos = eq_pos + torch.randn_like(eq_pos) * 0.01
    v = vf(t=torch.tensor(0), x=pos.reshape(1, 8 * 3)).reshape(8,3)
    print(v)
    print(vf.edge_index.shape)