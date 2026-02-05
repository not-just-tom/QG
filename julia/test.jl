using Pkg
pkg"add GeophysicalFlows, CairoMakie, Printf, Statistics, Random"

using GeophysicalFlows, CairoMakie, Printf, Random

using Statistics: mean

dev = CPU()     # Device (CPU/GPU)


n = 128            # 2D resolution: n² grid points
stepper = "AB3"  # timestepper
dt = 0.001           # timestep
nsteps = 1000           # total number of time-steps
nsubs  = 100             # number of time-steps for intermediate logging/plotting (nsteps must be multiple of nsubs)


L = 6.283185307179586        # domain size
β = 10.0      # planetary PV gradient
μ = 0.0       # bottom drag


prob = SingleLayerQG.Problem(dev; nx=n, Lx=L, β, μ, dt, stepper, aliased_fraction=0)



sol, clock, vars, params, grid = prob.sol, prob.clock, prob.vars, prob.params, prob.grid
x, y = grid.x, grid.y


E₀ = 0.08 # energy of initial condition

K = @. sqrt(grid.Krsq)                          # a 2D array with the total wavenumber

Random.seed!(1234)
q₀h = device_array(dev)(randn(Complex{eltype(grid)}, size(sol)))
@. q₀h = ifelse(K < 6  * 2π/L, 0, q₀h)
@. q₀h = ifelse(K > 10 * 2π/L, 0, q₀h)
@. q₀h[1, :] = 0    # remove any power from zonal wavenumber k=0
q₀h *= sqrt(E₀ / SingleLayerQG.energy(q₀h, vars, params, grid)) # normalize q₀ to have energy E₀
q₀ = irfft(q₀h, grid.nx)

SingleLayerQG.set_q!(prob, q₀)


fig = Figure(size = (800, 360))

axq = Axis(fig[1, 1];
           xlabel = "x",
           ylabel = "y",
           title = "initial potential vorticity (PV)",
           aspect = 1,
           limits = ((-grid.Lx/2, grid.Lx/2), (-grid.Ly/2, grid.Ly/2))
           )

hm = heatmap!(axq, x, y, Array(vars.q); colormap = :balance)
Colorbar(fig[1, 2], hm, label = "PV (units)")

E = Diagnostic(SingleLayerQG.energy, prob; nsteps)
Z = Diagnostic(SingleLayerQG.enstrophy, prob; nsteps)
diags = [E, Z] # A list of Diagnostics types passed to "stepforward!" will  be updated every timestep.


filepath = "."
plotpath = "./plots_decayingbetaturb"
plotname = "snapshots"
filename = joinpath(filepath, "decayingbetaturb.jld2")

if isfile(filename); rm(filename); end
if !isdir(plotpath); mkdir(plotpath); end



get_sol(prob) = Array(prob.sol) # extracts the Fourier-transformed solution (spectral)
get_q(prob) = Array(irfft(prob.sol, grid.nx)) # physical-space PV
out = Output(prob, filename, (:qh, get_sol), (:q, get_q))

# Save problem metadata and initial state so Python can read snapshots/qh or snapshots/q
saveproblem(out)
saveoutput(out)


Lx, Ly = grid.Lx, grid.Ly

title_q = Observable(@sprintf("vorticity, t = %.2f", clock.t))


fig = Figure(size = (900, 720))

axis_kwargs = (xlabel = "x",
               ylabel = "y",
               aspect = 1,
               limits = ((-Lx/2, Lx/2), (-Ly/2, Ly/2)))

axq = Axis(fig[1, 1]; title = title_q, axis_kwargs...)

q  = Observable(Array(vars.q))

hm = heatmap!(axq, x, y, q;
         colormap = :balance, colorrange = (-12, 12))

Colorbar(fig[1, 2], hm, label = "PV (units)")

fig


startwalltime = time()

frames = 0:round(Int, nsteps / nsubs)

record(fig, "singlelayerqg_betadecay.mp4", frames, framerate = 12) do j
  if j % round(Int, nsteps/nsubs / 4) == 0
    cfl = clock.dt * maximum([maximum(vars.u) / grid.dx, maximum(vars.v) / grid.dy])

    log = @sprintf("step: %04d, t: %d, cfl: %.2f, E: %.4f, Q: %.4f, walltime: %.2f min",
      clock.step, clock.t, cfl, E.data[E.i], Z.data[Z.i], (time()-startwalltime)/60)

    println(log)
  end

  q[] = vars.q

  title_q[] = @sprintf("vorticity, t = %.2f", clock.t)

  stepforward!(prob, diags, nsubs)
  SingleLayerQG.updatevars!(prob)
  # save snapshot after updating variables so JLD2 contains each recorded frame
  saveoutput(out)
end

savename = @sprintf("%s_%09d.png", joinpath(plotpath, plotname), clock.step)
savefig(savename)