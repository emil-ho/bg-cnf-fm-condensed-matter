import torch
from torch.func import jvp, vjp
import math

class Euclidean:
    r"""
    s_dim-Euclidean = ℝ^{s_dim}.

    Each configuration contains ``n_particles`` particles,
    so the ambient dimension is ``dim = n_particles * s_dim``.
    
    All public methods expect a flattened input:
    """

    # --------------------------------------------------------------------- #
    # construction helpers
    # --------------------------------------------------------------------- #
    def __init__(self,
                 n_particles: int = 1,
                 s_dim: int = 1,
                 *args,
                 **kwargs):
        """
        Parameters
        ----------
        n_particles : int
        s_dim       : int
        """
        super().__init__()

        self.n_particles = n_particles
        self.s_dim       = s_dim
        
    
    @property
    def dim(self):
        return self.n_particles * self.s_dim

    def _to_particles(self, x: torch.Tensor) -> torch.Tensor:
        """(..., n_particles * s_dim) → (..., n_particles, s_dim)"""
        return x.reshape(*x.shape[:-1], self.n_particles, self.s_dim)

    def _from_particles(self, x: torch.Tensor) -> torch.Tensor:
        """(..., n_particles, s_dim) → (...,  n_particles * s_dim)"""
        return x.reshape(*x.shape[:-2], self.dim)

    def projx(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def proju(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return u

    def expmap(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return x + u

    def logmap(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return y - x
    
    def transp(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return v
    
    def retr(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return x + u


class FlatTorus(Euclidean):
    """
    This is a class that represents the riemannian manifold flat-torus.
    It assumes the simulation box to be cubic.
    
    This is adapted from flowmm
    """
    def __init__(self, n_particles, s_dim, box=torch.tensor([1.,1.,1.])):
        super().__init__()
        if isinstance(box, float) or isinstance(box, int):
            box = torch.tensor([box, box, box])
        elif box.shape == (3,):
            self.box_edges = box
        elif box.shape == (3,3,):
            # check if box is ortho and aligned with cartesian coordinates
            assert torch.diag(box).pow(2).sum() == box.pow(2).sum()
            self.box_edges = torch.diag(box)
            
        self.box_vectors = torch.eye(3) * box
        self.n_particles = n_particles
        self.s_dim = s_dim

    @property
    def dim(self):
        return self.n_particles * self.s_dim

    def _to_particles(self, x: torch.Tensor) -> torch.Tensor:
        """
        (..., dim)  →  (..., n_*, s_dim)

        n_* is inferred from the last axis so the routine also works for
        edge-wise tensors where dim = n_edges * s_dim.
        """
        assert x.shape[-1] % self.s_dim == 0
        n_something = x.shape[-1] // self.s_dim
        return x.reshape(*x.shape[:-1], n_something, self.s_dim)

    def _from_particles(self, x: torch.Tensor) -> torch.Tensor:
        """(..., n_particles, s_dim) → (...,  n_particles * s_dim)"""
        return x.reshape(*x.shape[:-2], x.shape[-2] * x.shape[-1])

    def projx(self, x: torch.Tensor) -> torch.Tensor:
        x_particles = self._to_particles(x)
        x_proj = torch.remainder(x_particles, self.box_edges.to(x))
        return self._from_particles(x_proj)

    def proju(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """This does nothing here, but something in the FlatTorusMasked"""
        return u
    
    def expmap(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        x_particles = self._to_particles(x)
        u_particles = self._to_particles(u)
        result = torch.remainder(x_particles + u_particles, self.box_edges.to(x))
        return self._from_particles(result)

    def logmap(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_particles = self._to_particles(x)
        y_particles = self._to_particles(y)
        box_edges = self.box_edges.to(x)
        z = 2 * math.pi * (y_particles - x_particles) / box_edges
        result = box_edges * torch.atan2(torch.sin(z), torch.cos(z)) / (2 * math.pi)
        return self._from_particles(result)
    


def _calculate_target_batch_dim(*dims: int):
    return max(dims) - 1


class ProductManifold:
    def __init__(self, *manifolds):
        if len(manifolds) < 1:
            raise ValueError(
                "There should be at least one manifold in a product manifold"
            )
        super().__init__()
        self.slices = []
        name_parts = []
        self.manifolds = list(manifolds)
        dim = 0
        pos0 = 0
        for manifold in self.manifolds:
            size = manifold.dim
            dim += size
            name_parts.append(getattr(manifold, "name", manifold.__class__.__name__))
            pos1 = pos0 + size
            self.slices.append(slice(pos0, pos1))
            pos0 = pos1
        self.name = "x".join([f"({name})" for name in name_parts])
        self.n_elements = pos0
        self.n_manifolds = len(self.manifolds)
        self.dim = dim

    def _check_dim(self, x):
        assert x.shape[1] == self.dim

    def take_submanifold_value(self, x: torch.Tensor, i: int) -> torch.Tensor:
        """
        Take i'th slice of the ambient tensor and .

        Parameters
        ----------
        x : tensor
            Ambient tensor
        i : int
            submanifold index
        Returns
        -------
        torch.Tensor
        """
        slc = self.slices[i]
        part = x.narrow(-1, slc.start, slc.stop - slc.start)
        return part

    def projx(self, x: torch.Tensor) -> torch.Tensor:
        self._check_dim(x)
        projected = []
        for i, manifold in enumerate(self.manifolds):
            point = self.take_submanifold_value(x, i)
            proj = manifold.projx(point)
            proj = proj.view(*x.shape[: len(x.shape) - 1], -1)
            projected.append(proj)
        return torch.cat(projected, -1)

    def proju(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        target_batch_dim = _calculate_target_batch_dim(x.dim(), u.dim())
        projected = []
        for i, manifold in enumerate(self.manifolds):
            point = self.take_submanifold_value(x, i)
            tangent = self.take_submanifold_value(u, i)
            proj = manifold.proju(point, tangent)
            proj = proj.reshape((*proj.shape[:target_batch_dim], -1))
            projected.append(proj)
        return torch.cat(projected, -1)

    def expmap(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        target_batch_dim = _calculate_target_batch_dim(x.dim(), u.dim())
        mapped_tensors = []
        for i, manifold in enumerate(self.manifolds):
            point = self.take_submanifold_value(x, i)
            tangent = self.take_submanifold_value(u, i)
            mapped = manifold.expmap(point, tangent)
            mapped = mapped.reshape((*mapped.shape[:target_batch_dim], -1))
            mapped_tensors.append(mapped)
        return torch.cat(mapped_tensors, -1)

    def transp(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        target_batch_dim = _calculate_target_batch_dim(x.dim(), y.dim(), v.dim())
        transported_tensors = []
        for i, manifold in enumerate(self.manifolds):
            point = self.take_submanifold_value(x, i)
            point1 = self.take_submanifold_value(y, i)
            tangent = self.take_submanifold_value(v, i)
            transported = manifold.transp(point, point1, tangent)
            transported = transported.reshape(
                (*transported.shape[:target_batch_dim], -1)
            )
            transported_tensors.append(transported)
        return torch.cat(transported_tensors, -1)

    def logmap(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        target_batch_dim = _calculate_target_batch_dim(x.dim(), y.dim())
        logmapped_tensors = []
        for i, manifold in enumerate(self.manifolds):
            point = self.take_submanifold_value(x, i)
            point1 = self.take_submanifold_value(y, i)
            logmapped = manifold.logmap(point, point1)
            logmapped = logmapped.reshape((*logmapped.shape[:target_batch_dim], -1))
            logmapped_tensors.append(logmapped)
        return torch.cat(logmapped_tensors, -1)