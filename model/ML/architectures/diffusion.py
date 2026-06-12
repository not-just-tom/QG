import jax
import jax.numpy as jnp
import equinox as eqx

from model.ML.architectures.fno import SpectralConv2d


class _ScoreBackbone(eqx.Module):
    """FNO-style network that approximates s_theta(tau, U_tau, y)."""

    input_proj: eqx.nn.Conv2d
    spec_layers: list
    point_layers: list
    proj1: eqx.nn.Conv2d
    proj2: eqx.nn.Conv2d
    depth: int = eqx.field(static=True)

    def __init__(
        self,
        in_channels,
        out_channels,
        width,
        modes1,
        modes2,
        depth,
        key,
    ):
        self.depth = int(depth)
        keys = jax.random.split(key, 2 * self.depth + 4)

        self.input_proj = eqx.nn.Conv2d(
            in_channels=in_channels,
            out_channels=width,
            kernel_size=1,
            key=keys[0],
            padding_mode="CIRCULAR",
        )

        spec_layers = []
        point_layers = []
        for i in range(self.depth):
            spec_layers.append(SpectralConv2d(width, width, modes1, modes2, key=keys[1 + i]))
            point_layers.append(
                eqx.nn.Conv2d(
                    in_channels=width,
                    out_channels=width,
                    kernel_size=1,
                    key=keys[1 + self.depth + i],
                    padding_mode="CIRCULAR",
                )
            )

        self.spec_layers = spec_layers
        self.point_layers = point_layers

        self.proj1 = eqx.nn.Conv2d(
            in_channels=width,
            out_channels=width,
            kernel_size=1,
            key=keys[-2],
            padding_mode="CIRCULAR",
        )
        self.proj2 = eqx.nn.Conv2d(
            in_channels=width,
            out_channels=out_channels,
            kernel_size=1,
            key=keys[-1],
            padding_mode="CIRCULAR",
        )

    def __call__(self, x):
        h = self.input_proj(x)
        for i, (spec, point) in enumerate(zip(self.spec_layers, self.point_layers)):
            h = spec(h) + point(h)
            if i < self.depth - 1:
                h = jax.nn.gelu(h)
        h = jax.nn.gelu(self.proj1(h))
        return self.proj2(h)


class Diffusion(eqx.Module):
    """Conditional score-based closure model using VE-SDE reverse sampling.

    Input condition y is the resolved state (here the normalized PV field).
    Output is a sampled closure increment U ~ p(U|y) approximated by integrating
    the reverse VE-SDE with an adaptive schedule.

    Sparse-input branches are intentionally omitted.
    """
    
    tau_fourier_weights: jnp.ndarray = eqx.field(static=True)

    channels: int = eqx.field(static=True)
    time_embed_dim: int = eqx.field(static=True)
    sampler_steps: int = eqx.field(static=True)
    sampler_rho: float = eqx.field(static=True)
    tau_min: float = eqx.field(static=True)
    tau_max: float = eqx.field(static=True)
    sigma: float = eqx.field(static=True)
    stochastic_sampling: bool = eqx.field(static=True)
    score_clip: float = eqx.field(static=True)
    time_embed_scale: float = eqx.field(static=True)

    x_lift: eqx.nn.Conv2d
    y_lift: eqx.nn.Conv2d
    x_spec_layers: list
    x_point_layers: list
    y_spec_layers: list
    y_point_layers: list
    fusion1: eqx.nn.Conv2d
    fusion2: eqx.nn.Conv2d
    fusion3: eqx.nn.Conv2d
    out_proj1: eqx.nn.Conv2d
    out_proj2: eqx.nn.Conv2d
    time_dense1: eqx.nn.Linear
    time_dense2: eqx.nn.Linear

    def __init__(
        self,
        width=32,
        modes1=16,
        modes2=16,
        n_layers=4,
        time_embed_dim=16,
        sampler_steps=24,
        sampler_rho=7.0,
        tau_min=1e-3,
        tau_max=0.1,
        sigma=25.0,
        stochastic_sampling=False,
        score_clip=10.0,
        time_embed_scale=30.0,
        key=jax.random.PRNGKey(0),
        cfg=None,
        **kwargs,
    ):
        channels = cfg.params.nz if cfg is not None else 1
        channels = int(channels)

        self.channels = channels
        self.time_embed_dim = int(time_embed_dim)
        self.sampler_steps = int(sampler_steps)
        self.sampler_rho = float(sampler_rho)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.sigma = float(sigma)
        self.stochastic_sampling = bool(stochastic_sampling)
        self.score_clip = float(score_clip)
        self.time_embed_scale = float(time_embed_scale)

        keys = jax.random.split(key, 6 * int(n_layers) + 16)

        self.x_lift = eqx.nn.Conv2d(
            in_channels=channels + 2,
            out_channels=int(width),
            kernel_size=1,
            key=keys[0],
            padding_mode="CIRCULAR",
        )
        self.y_lift = eqx.nn.Conv2d(
            in_channels=channels + 2,
            out_channels=int(width),
            kernel_size=1,
            key=keys[1],
            padding_mode="CIRCULAR",
        )

        x_spec_layers = []
        x_point_layers = []
        y_spec_layers = []
        y_point_layers = []
        for i in range(int(n_layers)):
            x_spec_layers.append(SpectralConv2d(int(width), int(width), int(modes1), int(modes2), key=keys[2 + i]))
            x_point_layers.append(
                eqx.nn.Conv2d(
                    in_channels=int(width),
                    out_channels=int(width),
                    kernel_size=1,
                    key=keys[2 + int(n_layers) + i],
                    padding_mode="CIRCULAR",
                )
            )
            y_spec_layers.append(SpectralConv2d(int(width), int(width), int(modes1), int(modes2), key=keys[2 + 2 * int(n_layers) + i]))
            y_point_layers.append(
                eqx.nn.Conv2d(
                    in_channels=int(width),
                    out_channels=int(width),
                    kernel_size=1,
                    key=keys[2 + 3 * int(n_layers) + i],
                    padding_mode="CIRCULAR",
                )
            )

        self.x_spec_layers = x_spec_layers
        self.x_point_layers = x_point_layers
        self.y_spec_layers = y_spec_layers
        self.y_point_layers = y_point_layers

        offset = 2 + 4 * int(n_layers)
        self.fusion1 = eqx.nn.Conv2d(2 * int(width), 2 * int(width), kernel_size=1, key=keys[offset], padding_mode="CIRCULAR")
        self.fusion2 = eqx.nn.Conv2d(2 * int(width), int(width), kernel_size=1, key=keys[offset + 1], padding_mode="CIRCULAR")
        self.fusion3 = eqx.nn.Conv2d(int(width), int(width), kernel_size=1, key=keys[offset + 2], padding_mode="CIRCULAR")
        self.out_proj1 = eqx.nn.Conv2d(int(width), 128, kernel_size=1, key=keys[offset + 3], padding_mode="CIRCULAR")
        self.out_proj2 = eqx.nn.Conv2d(128, channels, kernel_size=1, key=keys[offset + 4], padding_mode="CIRCULAR")

        self.time_dense1 = eqx.nn.Linear(self.time_embed_dim, self.time_embed_dim, key=keys[offset + 5])
        self.time_dense2 = eqx.nn.Linear(self.time_embed_dim, int(width), key=keys[offset + 6])

        # Fixed Gaussian random features for time embedding.
        self.tau_fourier_weights = jax.random.normal(
            keys[offset + 7],
            (self.time_embed_dim // 2,),
            dtype=jnp.float32,
        )

    def _tau_embed(self, tau, height, width):
        tau = jnp.asarray(tau, dtype=jnp.float32)
        phases = 2.0 * jnp.pi * self.tau_fourier_weights * tau * self.time_embed_scale
        emb = jnp.concatenate([jnp.sin(phases), jnp.cos(phases)], axis=0)
        if emb.shape[0] < self.time_embed_dim:
            emb = jnp.pad(emb, (0, self.time_embed_dim - emb.shape[0]))
        elif emb.shape[0] > self.time_embed_dim:
            emb = emb[: self.time_embed_dim]
        emb = jax.nn.silu(self.time_dense1(emb))
        emb = jax.nn.silu(self.time_dense2(emb))
        return jnp.broadcast_to(emb[:, None, None], (emb.shape[0], height, width))

    def _coord_grid(self, height, width):
        gx = jnp.linspace(0.0, 1.0, width, dtype=jnp.float32)
        gy = jnp.linspace(0.0, 1.0, height, dtype=jnp.float32)
        grid_x = jnp.broadcast_to(gx[None, :], (height, width))
        grid_y = jnp.broadcast_to(gy[:, None], (height, width))
        return jnp.stack([grid_x, grid_y], axis=0)

    def _score_single(self, u_tau, y, tau):
        if u_tau.ndim != 3 or y.ndim != 3:
            raise ValueError("u_tau and y must both have shape (C,H,W)")
        if u_tau.shape != y.shape:
            raise ValueError("u_tau and y must have identical shape")
        if u_tau.shape[0] != self.channels:
            raise ValueError("Channel count does not match configured nz")

        height, width = u_tau.shape[-2:]
        t_embed = self._tau_embed(tau, height, width)
        grid = self._coord_grid(height, width)

        x_in = jnp.concatenate([u_tau, grid], axis=0)
        y_in = jnp.concatenate([y, grid], axis=0)

        x_feat = self.x_lift(x_in)
        y_feat = self.y_lift(y_in)

        for i, (spec, point) in enumerate(zip(self.x_spec_layers, self.x_point_layers)):
            x_feat = spec(x_feat) + point(x_feat) + t_embed
            if i < len(self.x_spec_layers) - 1:
                x_feat = jax.nn.gelu(x_feat)

        for i, (spec, point) in enumerate(zip(self.y_spec_layers, self.y_point_layers)):
            y_feat = spec(y_feat) + point(y_feat)
            if i < len(self.y_spec_layers) - 1:
                y_feat = jax.nn.gelu(y_feat)

        feat = jnp.concatenate([x_feat, y_feat], axis=0)
        feat = jax.nn.gelu(self.fusion1(feat))
        feat = jax.nn.gelu(self.fusion2(feat))
        feat = jax.nn.gelu(self.fusion3(feat))
        score = jax.nn.gelu(self.out_proj1(feat))
        score = self.out_proj2(score)

        # Match the original SBM scaling: network predicts score * std, then divide by std.
        std_tau = self._ve_std(tau)
        score = score / jnp.maximum(std_tau, 1e-6)
        return jnp.clip(score, -self.score_clip, self.score_clip)

    def score(self, u_tau, y, tau):
        """Evaluate s_theta(tau, U_tau, y).

        Supported shapes:
        - u_tau, y: (C,H,W), tau scalar
        - u_tau, y: (B,C,H,W), tau (B,) or scalar
        """
        if u_tau.ndim == 3:
            return self._score_single(u_tau.astype(jnp.float32), y.astype(jnp.float32), tau)

        if u_tau.ndim == 4:
            tau = jnp.asarray(tau, dtype=jnp.float32)
            if tau.ndim == 0:
                tau = jnp.broadcast_to(tau, (u_tau.shape[0],))
            if tau.shape[0] != u_tau.shape[0]:
                raise ValueError("Batch tau must have shape (B,)")
            return jax.vmap(self._score_single)(
                u_tau.astype(jnp.float32),
                y.astype(jnp.float32),
                tau,
            )

        raise ValueError("u_tau must have shape (C,H,W) or (B,C,H,W)")

    def _ve_std(self, tau):
        # Var(tau) = (sigma^(2*tau) - 1) / (2 * log(sigma))
        tau = jnp.asarray(tau, dtype=jnp.float32)
        var = (jnp.power(self.sigma, 2.0 * tau) - 1.0) / (2.0 * jnp.log(self.sigma))
        return jnp.sqrt(jnp.maximum(var, 1e-12))

    def _time_schedule(self):
        # Karras-style schedule from tau_max down to tau_min, then append 0.
        n = max(self.sampler_steps, 2)
        i = jnp.arange(n, dtype=jnp.float32)
        pmax = self.tau_max ** (1.0 / self.sampler_rho)
        pmin = self.tau_min ** (1.0 / self.sampler_rho)
        taus = (pmax + (i / (n - 1.0)) * (pmin - pmax)) ** self.sampler_rho
        return jnp.concatenate([taus, jnp.array([0.0], dtype=jnp.float32)], axis=0)

    def _sample_single(self, y, key=None, stochastic=None):
        if y.ndim != 3:
            raise ValueError("Condition y must have shape (C,H,W)")
        if y.shape[0] != self.channels:
            raise ValueError("Condition channel count does not match configured nz")

        use_stochastic = self.stochastic_sampling if stochastic is None else bool(stochastic)
        taus = self._time_schedule()

        if key is None:
            # Deterministic fallback key derived from condition snapshot.
            seed = jnp.asarray(jnp.abs(jnp.sum(y) * 1e6), dtype=jnp.uint32)
            key = jax.random.fold_in(jax.random.PRNGKey(0), seed)

        key, k0 = jax.random.split(key)
        u = self._ve_std(self.tau_max) * jax.random.normal(k0, y.shape, dtype=y.dtype)

        for i in range(taus.shape[0] - 1):
            tau_i = taus[i]
            tau_next = taus[i + 1]
            dtau = tau_i - tau_next

            score_i = self._score_single(u, y, tau_i)
            drift = (self.sigma ** (2.0 * tau_i)) * score_i * dtau

            if use_stochastic and i < (taus.shape[0] - 2):
                key, kn = jax.random.split(key)
                noise = jax.random.normal(kn, y.shape, dtype=y.dtype)
            else:
                noise = jnp.zeros_like(u)
            diffusion = (self.sigma ** tau_i) * jnp.sqrt(jnp.maximum(dtau, 0.0)) * noise

            u = u + drift + diffusion

        return u

    def sample(self, y, key=None, stochastic=None):
        """Sample closure increment U ~ p(U|y) approximately via reverse VE-SDE."""
        if y.ndim == 3:
            return self._sample_single(y.astype(jnp.float32), key=key, stochastic=stochastic)

        if y.ndim == 4:
            batch = y.shape[0]
            if key is None:
                keys = [None] * batch
            else:
                keys = list(jax.random.split(key, batch))
            outputs = [
                self._sample_single(y[i].astype(jnp.float32), key=keys[i], stochastic=stochastic)
                for i in range(batch)
            ]
            return jnp.stack(outputs, axis=0)

        raise ValueError("Condition y must have shape (C,H,W) or (B,C,H,W)")

    def denoising_score_matching_loss(self, u0, y, key):
        """VE denoising score matching objective (conditional form)."""
        if u0.ndim != 4 or y.ndim != 4:
            raise ValueError("u0 and y must both be shaped (B,C,H,W)")
        if u0.shape != y.shape:
            raise ValueError("u0 and y must have identical shape")

        batch = u0.shape[0]
        k_tau, k_noise = jax.random.split(key)
        tau = jax.random.uniform(
            k_tau,
            shape=(batch,),
            minval=self.tau_min,
            maxval=self.tau_max,
            dtype=jnp.float32,
        )

        std = self._ve_std(tau)[:, None, None, None]
        eps = jax.random.normal(k_noise, u0.shape, dtype=u0.dtype)
        u_tau = u0 + std * eps
        pred_score = self.score(u_tau, y, tau)
        # Same form as reference implementation: E || s_theta * std + eps ||^2
        per_sample = jnp.sum((pred_score * std + eps) ** 2, axis=(1, 2, 3))
        return jnp.mean(per_sample)

    def __call__(self, q):
        # Compatible with the existing closure interface in the solver.
        return self._sample_single(q.astype(jnp.float32), key=None, stochastic=None)
