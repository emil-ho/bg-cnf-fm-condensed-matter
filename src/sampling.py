from functools import partial
import torch
from src.integration import output_and_div
from src.manifolds import Euclidean, ProductManifold
from src.integration import projx_integrator_return_last, reconstruct_rk4_stats

def sample_with_dlogp(
    prior_samples,
    vecfield,
    manifold,
    integration_steps=11,
    integration_method='euler',
    div_mode='exact',
    k=1,
    device="cuda",
    pbar=True,
    return_probes=False,
):
    vecfield.to(device)
    prior_samples = prior_samples.to(device)
    stage_probes = []

    def odefunc(t, x, vecfield):
        x = x[..., :-2]  # leave out div and var
        vecfield_func = lambda x: vecfield(t, x)
        if div_mode == 'rademacher':
            dx, raw_probes, _ = output_and_div(
                vecfield_func, x, div_mode='rademacher_raw_probes', k=k
            )
            stage_probes.append(raw_probes.detach().cpu())
            div = raw_probes.mean(dim=-1, keepdim=True)
            var = (
                raw_probes.var(dim=-1, unbiased=True, keepdim=True) / k
                if k > 1
                else torch.zeros_like(div)
            )
        else:
            dx, div, var = output_and_div(vecfield_func, x, div_mode=div_mode, k=k)
            div = div.reshape(-1, 1)
        return torch.cat([dx, div, var], dim=-1)

    product_manifold = ProductManifold(manifold, Euclidean(2, 1))
    state0 = torch.cat([prior_samples, torch.zeros_like(prior_samples[..., :2])], dim=-1)
    ts = torch.linspace(0, 1, integration_steps+1, device=device)

    state1 = projx_integrator_return_last(
                        manifold=product_manifold,
                        odefunc=partial(odefunc, vecfield=vecfield),
                        x0=state0,
                        t=ts,
                        method=integration_method,
                        pbar=pbar,
                    )

    transformed_samples, dlogp, raw_var = state1[..., :-2], state1[..., -2:-1], state1[..., -1:]

    if div_mode == 'rademacher' and integration_method == 'rk4':
        probes = torch.stack(stage_probes, dim=1).reshape(
            len(prior_samples), integration_steps, 4, k
        )
        dlogp, var = reconstruct_rk4_stats(probes)
        dlogp = dlogp.unsqueeze(-1)
        var = var.unsqueeze(-1)
    else:
        probes = torch.stack(stage_probes, dim=1) if stage_probes else None
        dt = 1.0 / integration_steps
        var = raw_var * dt
        if integration_method == 'rk4':
            var = var * (5.0 / 18.0)

    if return_probes:
        return transformed_samples, dlogp, var, probes
    return transformed_samples, dlogp, var


def sample_with_dlogp_batched(
    prior_samples,
    vecfield,
    manifold,
    integration_steps=11,
    integration_method='euler',
    div_mode='exact',
    k=1,
    device="cuda",
    pbar=True,
    return_probes=False,
    batchsize=10
):
    batches = [
        sample_with_dlogp(
            batch,
            vecfield,
            manifold,
            integration_steps=integration_steps,
            integration_method=integration_method,
            div_mode=div_mode,
            k=k,
            device=device,
            pbar=pbar,
            return_probes=return_probes,
        )
        for batch in prior_samples.split(batchsize)
    ]
    return tuple(
        torch.cat(items, dim=0) if items[0] is not None else None
        for items in zip(*batches)
    )
