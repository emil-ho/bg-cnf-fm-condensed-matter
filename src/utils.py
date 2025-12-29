import torch
from numba import njit, prange
import numpy as np
import ot as pot

@njit(parallel=True, fastmath=True, cache=True)
def cdist2_numba_ortho_unrolled(x0, x1, n_particles, box_edges):
    """
    Fastest squared config distance for cubic periodic box.

    Parameters
    ----------
    x0, x1 : (N, P*3) flat Cartesian coordinates
    box_length : float (assumes cubic box with side L)
    """
    N0, N1 = x0.shape[0], x1.shape[0]
    d2 = np.empty((N0, N1), dtype=x0.dtype)
    Lx, Ly, Lz = box_edges

    for i in prange(N0):
        for j in range(N1):
            acc = 0.0
            for p in range(n_particles):
                base = p * 3

                # x, y, z deltas
                dx = x0[i, base]     - x1[j, base]
                dy = x0[i, base + 1] - x1[j, base + 1]
                dz = x0[i, base + 2] - x1[j, base + 2]

                # minimum image convention
                dx -= Lx * np.round(dx / Lx)
                dy -= Ly * np.round(dy / Ly)
                dz -= Lz * np.round(dz / Lz)

                # accumulate squared distance
                acc += dx * dx + dy * dy + dz * dz

            d2[i, j] = acc
    return d2

def ot_reordering(x0, x1, box):
    B = x0.shape[0]
    n_particles = x0.shape[1] // 3

    x0, x1, box = x0.cpu().numpy(), x1.cpu().numpy(), box.cpu().numpy()

    a = pot.unif(B)
    b = pot.unif(B)

    cost = cdist2_numba_ortho_unrolled(x0, x1, n_particles, box)
    cost = (cost / cost.max())

    pi = pot.emd(a, b, cost)

    flat_pi = pi.ravel()
    flat_pi /= flat_pi.sum()
    choices = np.random.choice(pi.size, size=B, p=flat_pi)
    i, j = np.divmod(choices, B)
    i = torch.as_tensor(i, dtype=torch.long)
    j = torch.as_tensor(j, dtype=torch.long)

    return i, j