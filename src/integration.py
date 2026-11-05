import torch
from torch.func import vmap, jacrev, vjp, jvp
from tqdm.auto import tqdm


@torch.no_grad()
def projx_integrator_return_last(manifold, odefunc, x0, t, method="euler", pbar=True):
    """Adapted from rfm. Also handles dlogp (on Euclidean manifold)"""
    step_fn = {
        "euler": euler_step,
        "midpoint": midpoint_step,
        "rk4": rk4_step,
    }[method]

    xt = x0
    t0s = t[:-1]
    if pbar:
        t0s = tqdm(t0s)

    for t0, t1 in zip(t0s, t[1:]):
        dt = t1 - t0
        vt = odefunc(t=t0, x=xt)
        xt = step_fn(odefunc, xt, vt, t0, dt, manifold=manifold)
        xt = manifold.projx(xt)
    return xt


def euler_step(odefunc, xt, vt, t0, dt, manifold=None):
    if manifold is not None:
        return manifold.expmap(xt, dt * vt)
    else:
        return xt + dt * vt


def midpoint_step(odefunc, xt, vt, t0, dt, manifold=None):
    half_dt = 0.5 * dt
    if manifold is not None:
        x_mid = xt + half_dt * vt
        v_mid = odefunc(t0 + half_dt, x_mid)
        v_mid = manifold.transp(x_mid, xt, v_mid)
        return manifold.expmap(xt, dt * v_mid)
    else:
        x_mid = xt + half_dt * vt
        return xt + dt * odefunc(t0 + half_dt, x_mid)


def _move(xt, v, scale, manifold):
    if manifold is None:
        return xt + scale * v
    return manifold.expmap(xt, scale * v)


def _transport(x_from, x_to, v, manifold):
    if manifold is None:
        return v
    return manifold.transp(x_from, x_to, v)


def rk4_step(odefunc, xt, vt, t0, dt, manifold=None):
    # k1 in T_{xt}M
    k1 = vt if vt is not None else odefunc(t0, xt)

    # Stage 2: x2 = exp_x( (dt/2) * k1 )
    x2 = _move(xt, k1, 0.5 * dt, manifold)
    v2 = odefunc(t0 + 0.5 * dt, x2)
    k2 = _transport(x2, xt, v2, manifold)  # bring to T_{xt}M

    # Stage 3: x3 = exp_x( (dt/2) * k2 )
    x3 = _move(xt, k2, 0.5 * dt, manifold)
    v3 = odefunc(t0 + 0.5 * dt, x3)
    k3 = _transport(x3, xt, v3, manifold)

    # Stage 4: x4 = exp_x( dt * k3 )
    x4 = _move(xt, k3, dt, manifold)
    v4 = odefunc(t0 + dt, x4)
    k4 = _transport(x4, xt, v4, manifold)

    # Combine in T_{xt}M, then retract/exp back to the manifold
    v_comb = (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    return _move(xt, v_comb, dt, manifold)


def output_and_div(vecfield, x: torch.Tensor, div_mode: str = "exact", k: int = 5):
    """Compute vector field output, divergence estimate, and estimator variance.

    Parameters
    ----------
    vecfield : Callable[[torch.Tensor], torch.Tensor]
        Function mapping configurations of shape ``[B, D]`` to velocities ``[B, D]``.
    x : torch.Tensor
        Input configurations of shape ``[B, D]``.
    div_mode : str, optional
        Divergence calculation mode (``"exact"``, ``"rademacher"``, or
        ``"rademacher_raw_probes"``), by default ``"exact"``.
    k : int, optional
        Number of Hutchinson Rademacher probes when ``div_mode`` starts with
        ``"rademacher"``, by default 5.

    Returns
    -------
    dx : torch.Tensor
        Vector field output of shape ``[B, D]``.
    div : torch.Tensor
        Divergence estimate of shape ``[B, 1]``, or raw probe traces of shape
        ``[B, k]`` when ``div_mode="rademacher_raw_probes"``.
    var_div : torch.Tensor or None
        Unbiased variance of the divergence estimator of shape ``[B, 1]``
        (zeros when ``div_mode="exact"`` or ``k <= 1``, ``None`` when
        ``div_mode="rademacher_raw_probes"``).
    """
    B, D = x.shape

    if div_mode == "exact":
        dx = vecfield(x)

        def single_vecfield(xi: torch.Tensor) -> torch.Tensor:
            return vecfield(xi.unsqueeze(0)).squeeze(0)

        div = vmap(lambda xi: torch.trace(jacrev(single_vecfield)(xi)))(x)
        div = div.unsqueeze(-1)
        var_div = torch.zeros_like(div)
        return dx, div, var_div

    elif div_mode.startswith("rademacher"):
        # One forward pass; keep its activations for all probes
        v = (torch.randint(0, 2, (k, *x.shape), device=x.device) * 2 - 1).to(x)
        v_list = [v[i] for i in range(k)]

        trace_estimates = []
        dx, vjpfunc = vjp(vecfield, x)
        for v_b in v_list:                      # v_b: [B, D]
            vJ = vjpfunc(v_b)[0]                # [B, D] = (v^T J)
            trace_estimates.append((vJ * v_b).sum(-1))  # [B]

        trace_estimates = torch.stack(trace_estimates)  # [K, B]
        if div_mode == "rademacher_raw_probes":
            return dx, trace_estimates.T, None

        div = trace_estimates.mean(dim=0).unsqueeze(-1)  # [B, 1]

        if k > 1:
            var_div = trace_estimates.var(dim=0, unbiased=True).unsqueeze(-1) / k  # [B, 1]
        else:
            var_div = torch.zeros_like(div)

        return dx, div, var_div

    else:
        raise ValueError(f"div mode '{div_mode}' is not known/implemented")


def reconstruct_rk4_stats(
    probess: torch.Tensor,
    n_subset: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct accumulated log-density change and estimator variance from raw RK4 probes.

    Assumes integration over ``t in [0, 1]`` with uniform step size ``dt = 1 / steps``.

    Parameters
    ----------
    probess : torch.Tensor
        Raw trace estimates of shape ``(batch, steps, 4, k)``, where ``4`` corresponds
        to the four RK4 stages ``(k1, k2, k3, k4)`` and ``k`` is the total number of probes.
    n_subset : int or None, optional
        If provided, only the first ``n_subset`` probes are used, by default ``None``.

    Returns
    -------
    dlogp : torch.Tensor
        Accumulated log-density change of shape ``(batch,)``.
    variance : torch.Tensor
        Total variance of the log-density change estimator of shape ``(batch,)``.
    """
    steps = probess.shape[1]
    dt = 1.0 / steps

    if n_subset is not None:
        probes = probess[..., :n_subset]
        k_used = n_subset
    else:
        probes = probess
        k_used = probes.shape[-1]

    means = probes.mean(dim=-1)  # [Batch, Steps, 4]

    if k_used > 1:
        vars_population = probes.var(dim=-1, unbiased=True)
        vars_mean = vars_population / k_used
    else:
        vars_mean = torch.zeros_like(means)

    weighted_means = (
        means[..., 0] * 1.0
        + means[..., 1] * 2.0
        + means[..., 2] * 2.0
        + means[..., 3] * 1.0
    ) / 6.0
    dlogp = (weighted_means * dt).sum(dim=1)

    weighted_vars = (
        vars_mean[..., 0] * 1.0
        + vars_mean[..., 1] * 4.0
        + vars_mean[..., 2] * 4.0
        + vars_mean[..., 3] * 1.0
    ) / 36.0
    variance = (weighted_vars * (dt**2)).sum(dim=1)

    return dlogp, variance

