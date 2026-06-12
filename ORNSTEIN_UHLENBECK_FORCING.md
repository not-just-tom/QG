# Ornstein-Uhlenbeck Forcing Discussion

## Current Implementation: White Noise Forcing

The current implementation uses **white noise forcing** at each timestep:
- Each timestep gets independent random forcing from `jax.random.normal`
- The forcing is generated deterministically from a seed and timestep counter
- This is equivalent to a delta-correlated forcing in time: `<F(t)F(t')> = σ² δ(t-t')`

## Why Ornstein-Uhlenbeck (OU) Process?

An **Ornstein-Uhlenbeck process** provides **time-correlated** stochastic forcing, which is more physically realistic for many fluid dynamics applications:

### Physical Motivation
1. **Realistic turbulence**: Real turbulent forcing has temporal correlations
2. **Energy cascade**: Time-correlated forcing better represents energy input at specific scales
3. **Spectral control**: OU processes allow control of both spatial (via forcing_mask) and temporal (via correlation time) scales

### Mathematical Formulation

The OU process for forcing evolves as:
```
dξ/dt = -ξ/τ + √(2σ²/τ) dW/dt
```

Where:
- `ξ(t)` is the forcing field (complex in spectral space)
- `τ` is the **correlation time** (decorrelation timescale)
- `σ²` is the variance of the stationary distribution
- `dW/dt` is white noise

Key properties:
- **Stationary variance**: `<ξ²> = σ²`
- **Autocorrelation**: `<ξ(t)ξ(t+s)> = σ² exp(-|s|/τ)`
- **Recovers white noise**: As `τ → 0`, the process becomes delta-correlated

### Discrete-Time Update (for numerical integration)

For a timestep `dt`, the exact discretization is:
```
ξ(t+dt) = exp(-dt/τ) ξ(t) + σ√(1 - exp(-2dt/τ)) η
```

Where `η ~ N(0,1)` is standard normal noise.

This ensures:
1. Correct stationary variance `σ²`
2. Exact exponential decay of correlations
3. Numerical stability for all `dt/τ` ratios

## Comparison with FourierFlows.jl

FourierFlows.jl (specifically GeophysicalFlows.jl) implements OU forcing with:

### Key Features to Match:
1. **Per-wavenumber OU processes**: Each Fourier mode has its own OU process
2. **Time-linked evolution**: The forcing state evolves consistently with model time
3. **Spectral band control**: Forcing concentrated in specified wavenumber bands
4. **Deterministic reproducibility**: Uses random seeds for reproducible runs

### FourierFlows.jl Approach:
```julia
# Forcing state maintained separately
struct ForcingState
    ξ::Array{ComplexF64, 3}  # Current forcing field (nz, ny, nk)
    τ::Float64               # Correlation time
    σ::Float64               # Forcing amplitude
end

# Update forcing at each step
function update_forcing!(F, dt, rng)
    decay = exp(-dt/F.τ)
    noise_amp = F.σ * sqrt(1 - decay^2)
    
    for i in eachindex(F.ξ)
        F.ξ[i] = decay * F.ξ[i] + noise_amp * randn(rng, ComplexF64)
    end
    
    # Apply spectral mask
    F.ξ .*= forcing_mask
    # Ensure k=0 mode is real
    F.ξ[:, :, 1] .= real(F.ξ[:, :, 1])
end
```

## Implementation Recommendations for JAX

### Option 1: Add OU State to Model State (Cleanest for FourierFlows.jl similarity)

Add forcing state to the `State` dataclass:
```python
@dataclasses.dataclass(frozen=True, kw_only=True)
class State:
    qh: jnp.ndarray           # PV in spectral space
    _q_shape: tuple[int, int]
    forcing_xi: jnp.ndarray = None  # OU forcing state (same shape as qh)
```

Update logic in kernel:
```python
def _do_ornstein_uhlenbeck_forcing(self, state, key, dt):
    if state.forcing_xi is None or self.forcing_amplitude == 0:
        return state  # No forcing
    
    # OU process parameters
    tau = self.forcing_correlation_time  # New parameter
    decay = jnp.exp(-dt / tau)
    noise_amp = self.forcing_amplitude * jnp.sqrt(1 - decay**2)
    
    # Generate noise
    key_real, key_imag = jax.random.split(key)
    noise_real = jax.random.normal(key_real, state.forcing_xi.shape)
    noise_imag = jax.random.normal(key_imag, state.forcing_xi.shape)
    noise = noise_real + 1j * noise_imag
    noise = noise.at[..., :, 0].set(jnp.real(noise[..., :, 0]))
    
    # OU update
    xi_new = decay * state.forcing_xi + noise_amp * noise
    
    # Apply spatial mask
    forcing_mask = jnp.expand_dims(self.forcing_mask, 0)
    xi_new = xi_new * forcing_mask
    
    # Add to PV tendency
    dqhdt_forced = state.dqhdt + xi_new
    
    return state.update(dqhdt=dqhdt_forced, forcing_xi=xi_new)
```

### Option 2: Time-Linked via Stepper (Current Approach, Extended)

Keep forcing generation in the stepper but with OU memory:
```python
# In SteppedModel, maintain OU state
@Pytree.register_pytree_dataclass
@dataclasses.dataclass
class OUForcingAux:
    xi: jnp.ndarray  # Current OU state
    
# Update step_model to evolve OU state
```

### Option 3: Stateless Approximation (Simplest, but less accurate)

Use a stateless approximation where we seed the noise based on time but apply a smoothing kernel:
```python
# Generate independent noise at each step
# Apply temporal smoothing via exponential moving average
# Not truly OU but simpler for JAX jit
```

## Recommended Implementation Path

**For best FourierFlows.jl compatibility**: Use **Option 1**

### Benefits:
1. ✅ Exactly matches FourierFlows.jl OU forcing
2. ✅ Time-linked: forcing evolves with physical time
3. ✅ Statistically correct: proper OU autocorrelation
4. ✅ JAX-jittable: all state in pytree
5. ✅ Reproducible: deterministic from seed + timestep

### Required Changes:
1. Add `forcing_xi` field to `State` dataclass
2. Add `forcing_correlation_time` parameter to model
3. Replace `_do_stochastic_forcing` with `_do_ornstein_uhlenbeck_forcing`
4. Initialize `forcing_xi` with random noise in `initialise` method
5. Update State initialization to include forcing_xi

### Configuration Parameters:
```yaml
params:
  # Forcing parameters
  forcing_type: 'annulus'          # 'band' or 'annulus'
  forcing_amplitude: 1.0e-4        # σ: forcing strength
  forcing_center: 6.0              # k_f: center wavenumber
  forcing_width: 2.0               # Gaussian width
  forcing_correlation_time: 5.0    # τ: decorrelation time (in model time units)
  forcing_seed: 12345              # Random seed for reproducibility
```

## Energy Input and Stationarity

For a statistically stationary turbulent state with forcing and dissipation:

### Energy Balance:
```
Energy input from forcing ≈ Energy dissipation
```

With OU forcing:
```
ε_forcing = σ² ∫ k² Mask(k) dk
```

Where `Mask(k)` is the forcing mask in wavenumber space.

### Tuning Guidelines:
1. **Forcing amplitude**: Set `σ` to balance dissipation rate
2. **Correlation time**: Typically `τ ~ 1/k_f` for Rossby wave timescale
3. **Annulus width**: Narrower = more scale-selective forcing
4. **Center wavenumber**: Usually near energy injection scale

## Testing and Validation

To verify correct OU implementation:

1. **Temporal autocorrelation**: Should decay as `exp(-t/τ)`
2. **Stationary variance**: `<ξ²>` should equal `σ²`
3. **Energy injection rate**: Monitor `<ξ · ∂q/∂t>` over time
4. **Spectral concentration**: Forcing energy concentrated at `k_f ± Δk`

## Performance Considerations

### JAX JIT Compatibility:
- ✅ OU state as array: fully jittable
- ✅ Exponential decay: pure function
- ✅ Random noise: use `jax.random` with key passing
- ✅ Conditional forcing: use `jax.lax.cond` for zero-amplitude case

### Memory Overhead:
- Additional state array: same size as `qh`
- Minimal compared to full state storage

## Conclusion

The current implementation provides **white noise forcing** (instantaneous random forcing at each step) which works but lacks temporal correlations.

For **physically realistic forcing** matching FourierFlows.jl:
1. Implement **Ornstein-Uhlenbeck process** with state variable `forcing_xi`
2. Add **correlation time** parameter `τ` 
3. Evolve forcing field with exponential decay + noise
4. Maintain time-linked state through stepping process

This gives smooth, correlated forcing that better represents physical turbulent energy injection while remaining fully compatible with JAX jit compilation.
