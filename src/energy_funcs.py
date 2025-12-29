import torch
import math
Array = torch.Tensor

# all energy funcs are expected to work the following way:
# - give the same result for positions mapped into the box or not
# - work with batched and non-batched input
# - automatically adjust to the device of the input tensor
# - automatically convert the box argument (list, tensor, etc.) into a tensor on the correct device

def pairwise_difference_pbc(coordinates: Array, box_length: Array) -> Array:
    """Computes pairwise distance vectors obeying periodic boundary conditions.

    Args:
      coordinates: array with shape [..., num_particles, dim] containing particle
        coordinates.
      box_length: array with shape [..., dim], the edge lengths of the box.

    Returns:
      Array with shape [..., num_particles, num_particles, dim], the pairwise
      distance vectors with respect to periodic boundary conditions.
    """
    deltas = _pairwise_difference(coordinates)
    box_length = box_length[..., None, None, :]
    return deltas - box_length * torch.round(deltas / box_length)


def _pairwise_difference(coordinates: Array) -> Array:
    """Computes pairwise difference vectors.

    Args:
      coordinates: array with shape [..., num_particles, dim] containing particle
        coordinates.

    Returns:
      Array with shape [..., num_particles, num_particles, dim], difference
      vectors for all pairs of particles.
    """
    if coordinates.ndim < 2:
        raise ValueError(
            f"Expected at least 2 array dimensions, got {coordinates.ndim}."
        )
    x = coordinates[..., :, None, :]
    y = coordinates[..., None, :, :]
    return x - y


def pairwise_squared_distance_pbc(coordinates: Array, box_length: Array) -> Array:
    """Computes pairwise squared distance obeying periodic boundary conditions.

    Args:
      coordinates: array with shape [..., num_particles, dim] containing particle
        coordinates.
      box_length: array with shape [..., dim], the edge lengths of the box.

    Returns:
      Array with shape [..., num_particles, num_particles] with pairwise squared
      distances.
    """
    coordinate_deltas = _pairwise_difference(coordinates)
    # chex.assert_is_broadcastable(box_length.shape[:-1], coordinates.shape[:-2])
    return squared_distance_pbc(coordinate_deltas, box_length[..., None, None, :])


def squared_distance_pbc(coordinate_deltas: Array, box_length: Array) -> Array:
    """Computes squared distance obeying periodic boundary conditions.

    Args:
      coordinate_deltas: array with shape [..., dim] containing difference
        vectors.
      box_length: array with shape [..., dim], the edge lengths of the box.

    Returns:
      Array with shape [...], the squared distances with respect to periodic
      boundary conditions.
    """
    coordinate_deltas_pbc = coordinate_deltas - box_length * torch.round(
        coordinate_deltas / box_length
    )
    return torch.sum(coordinate_deltas_pbc**2, axis=-1)




def mw_energy(coordinates, box_length=[1], unit_length='angstroem', unit_energy='kcal/mol'): # TODO: make this work for a single configuration again
    """Compute the energy of multiple configurations of monoatomic water with pbc.
    
    Args:
        coordinates: tensor of shape (batch_size, particle_num, position_dim)
        box_length: box length for periodic boundary conditions
        
    Returns:
        tensor of shape (batch_size,) containing energies for each configuration
    """
    if isinstance(box_length, (float, int)):
        box_length = [box_length]
    box_length = torch.as_tensor(box_length, device=coordinates.device, dtype=coordinates.dtype)

    # Constants
    MW_A = 7.049556277
    MW_B = 0.6022245584
    MW_GAMMA = 1.2
    if unit_energy == 'kcal/mol':
      MW_EPSILON = 6.189
    elif unit_energy == 'kJ/mol':
      MW_EPSILON = 6.189 * 4.184
    else:
      raise ValueError(f"Unit '{unit_energy}' not implemented")
    if unit_length == 'angstroem':
      MW_SIGMA = 2.3925
    elif unit_length == 'nm':
      MW_SIGMA = 2.3925 / 10
    else:
      raise ValueError(f"Unit '{unit_length}' not implemented")
    
    MW_REDUCED_CUTOFF = 1.8
    MW_COS = math.cos(109.47 / 180.0 * math.pi)
    MW_LAMBDA = 23.15

    dr = pairwise_difference_pbc(coordinates, box_length)
    dr /= MW_SIGMA

    def _three_body_energy_batched(dr):
        """Compute three-body term for multiple samples."""
        
        def _one_particle_contribution(dri):
            raw_norms = torch.linalg.norm(dri, dim=-1)
            keep = raw_norms < MW_REDUCED_CUTOFF
            norms = torch.where(keep, raw_norms, 1e20)
            norm_energy = torch.exp(MW_GAMMA / (norms - MW_REDUCED_CUTOFF))
            norm_energy = torch.where(keep, norm_energy, 0.0)
            normprods = norms[..., None, :] * norms[..., :, None]
            
            dotprods = torch.sum(dri[..., None, :, :] * dri[..., :, None, :], dim=-1)
            cos_ijk = dotprods / normprods
            
            energy = MW_LAMBDA * MW_EPSILON * (MW_COS - cos_ijk) ** 2
            energy *= norm_energy[..., None, :]
            energy = torch.triu(energy, 1)
            energy = torch.sum(energy, dim=-1)
            return torch.sum(energy * norm_energy, dim=-1)

        # Handle batch dimension when cleaning diagonal elements
        clean_dr = torch.moveaxis(
            torch.triu(torch.moveaxis(dr, -1, 0), 1)[..., 1:]
            + torch.tril(torch.moveaxis(dr, -1, 0), -1)[..., :-1],
            0,
            -1,
        )
        
        # Sum over particles for each configuration in batch
        batch_energy = torch.zeros(dr.shape[0], device=dr.device)
        for i in range(clean_dr.shape[1]):
            batch_energy += _one_particle_contribution(clean_dr[:, i])
        
        return batch_energy

    def _two_body_energy(r2):
        r2 = r2 / (MW_SIGMA ** 2)
        r = torch.sqrt(r2)
        mask = r < MW_REDUCED_CUTOFF
        # Distances on or above the cutoff can cause NaNs in the gradient of
        # `term_2` below, even though they're masked out in the forward computation.
        # To avoid this, we set these distances to a safe value.
        r = torch.where(mask, r, 2.0 * MW_REDUCED_CUTOFF)
        term_1 = MW_A * MW_EPSILON * (MW_B / r2**2 - 1.0)
        term_2 = torch.where(mask, torch.exp(1.0 / (r - MW_REDUCED_CUTOFF)), 0.0)
        return term_1 * term_2

    # Calculate three-body energy for all configurations
    three_body_energy = _three_body_energy_batched(dr)

    # Calculate two-body energy for all configurations
    r2 = pairwise_squared_distance_pbc(coordinates, box_length)
    r2 = r2 + torch.eye(r2.shape[-1], device=r2.device)[None, :, :]
    
    energies = _two_body_energy(r2)
    two_body_energy = torch.sum(torch.triu(energies, diagonal=1), dim=[-2, -1])

    total_energy = three_body_energy + two_body_energy
    return total_energy


# TODO: check that device handling works like with mw and also batched vs not
def lj_energy(coordinates,
  cutoff, 
  box_length: torch.Tensor, 
  epsilon=1., 
  sigma=1., 
  lambda_lj=1., 
  min_distance=0., 
  linearize_below=None, 
  shift_energy=False,
  device='cpu'):

  box_length.to(coordinates)
  if type(box_length) == float or type(box_length) == int:
      box_length = [box_length]
  box_length = torch.tensor(box_length, device=coordinates.device)

  def _soft_core_lj_potential(r2: Array, sigma, lambda_lj, epsilon) -> Array:
    r6 = r2**3 / sigma**6
    r6 += 0.5 * (1. - lambda_lj)**2
    r6inv = 1. / r6
    energy = r6inv * (r6inv - 1.)
    energy *= 4. * lambda_lj * epsilon
    return energy

  def _unclipped_pairwise_potential(r2: Array, sigma, lambda_lj, epsilon, cutoff, shift) -> Array:
    energy = _soft_core_lj_potential(r2, sigma, lambda_lj, epsilon)
    # Apply radial cutoff and shift.
    energy = torch.where(r2 <= cutoff**2, energy - shift, 0.)
    return energy

  shift = _soft_core_lj_potential(cutoff**2, sigma, lambda_lj, epsilon) if shift_energy else 0

  r2 = pairwise_squared_distance_pbc(coordinates, box_length)
  r2 += torch.eye(r2.shape[-1], device=coordinates.device)

  energies = _unclipped_pairwise_potential(r2, sigma, lambda_lj, epsilon, cutoff, shift)
  return torch.sum(torch.triu(energies, diagonal=1), axis=[-2, -1])