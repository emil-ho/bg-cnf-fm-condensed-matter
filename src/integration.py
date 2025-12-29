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


def output_and_div(vecfield, x, div_mode="exact", k=5):
    """
    vecfield: a function that, given x of shape [B, D], returns [B, D]
    x: shape [B, D]
    We want to compute dx = vecfield(x), plus the per-sample divergence.
    """
    B, D = x.shape

    if div_mode == 'exact':
        dx = vecfield(x)

        def single_vecfield(xi: torch.Tensor) -> torch.Tensor:
            return vecfield(xi.unsqueeze(0)).squeeze(0)
        
        div = vmap(lambda xi: torch.trace(jacrev(single_vecfield)(xi)))(x)
        return dx, div.unsqueeze(-1)

    elif div_mode == "rademacher":
        # One forward pass; keep its activations for all probes
        v_list = []
        v = (torch.randint(0, 2, (k, *x.shape), device=x.device) * 2 - 1).to(x)
        v_list = [v[k].to(x) for k in range(v.shape[0])]

        acc = torch.zeros(B, device=x.device, dtype=x.dtype)
    
        dx, vjpfunc = vjp(vecfield, x)
        for v_b in v_list:                      # v_b: [B, D]
            vJ = vjpfunc(v_b)[0]                # [B, D] = (v^T J)
            acc = acc + (vJ * v_b).sum(-1)      # [B]

        div = (acc / len(v_list)).unsqueeze(-1)  # [B, 1]
        return dx, div

    else:
        raise ValueError(f"div mode '{div_mode}' is not known/implemented")