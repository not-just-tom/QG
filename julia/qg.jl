using Pkg
pkg"add GeophysicalFlows, CUDA, JLD2, CairoMakie, Statistics"

using GeophysicalFlows, CUDA, JLD2, CairoMakie, Random, Printf

using Statistics: mean
using LinearAlgebra: ldiv!

parsevalsum = FourierFlows.parsevalsum
record = CairoMakie.record                # disambiguate between CairoMakie.record and CUDA.record

dev = CPU()     # Device (CPU/GPU)

n = 256            # 2D resolution: n² grid points
stepper = "FilteredRK4"  # timestepper
dt = 0.005           # timestep
nsteps = 10000           # total number of timesteps
save_substeps = 100      # number of timesteps after which output is saved

L = 2π        # domain size
β = 10.0      # planetary PV gradient
μ = 0.01      # bottom drag

forcing_wavenumber = 8.0 * 2π/L  # the forcing wavenumber, `k_f`, for a spectrum that is a ring in wavenumber space
forcing_bandwidth  = 2.0  * 2π/L  # the width of the forcing spectrum, `δ_f`
ε = 0.001                         # energy input rate by the forcing

grid = TwoDGrid(dev; nx=n, Lx=L)

K = @. sqrt(grid.Krsq)            # a 2D array with the total wavenumber

forcing_spectrum = @. exp(-(K - forcing_wavenumber)^2 / (2 * forcing_bandwidth^2))
@CUDA.allowscalar forcing_spectrum[grid.Krsq .== 0] .= 0 # ensure forcing has zero domain-average

ε0 = parsevalsum(forcing_spectrum .* grid.invKrsq / 2, grid) / (grid.Lx * grid.Ly)
@. forcing_spectrum *= ε/ε0       # normalize forcing to inject energy at rate ε

if dev==CPU(); Random.seed!(1234); else; CUDA.seed!(1234); end

function calcF!(Fh, sol, t, clock, vars, params, grid)
  randn!(Fh)
  @. Fh *= sqrt(forcing_spectrum) / sqrt(clock.dt)
  return nothing
end

prob = SingleLayerQG.Problem(dev; nx=n, Lx=L, β, μ, dt, stepper,
                             calcF=calcF!, stochastic=true)
    
sol, clock, vars, params, grid = prob.sol, prob.clock, prob.vars, prob.params, prob.grid
x,  y  = grid.x,  grid.y
Lx, Ly = grid.Lx, grid.Ly

SingleLayerQG.set_q!(prob, device_array(dev)(zeros(grid.nx, grid.ny)))

E = Diagnostic(SingleLayerQG.energy, prob; nsteps, freq=save_substeps)
Z = Diagnostic(SingleLayerQG.enstrophy, prob; nsteps, freq=save_substeps)
diags = [E, Z] # A list of Diagnostics types passed to "stepforward!" will  be updated every timestep.

filepath = "."
plotpath = "./plots_forcedbetaturb"
plotname = "snapshots"
filename = joinpath(filepath, "singlelayerqg_forcedbeta.jld2")

if isfile(filename); rm(filename); end
if !isdir(plotpath); mkdir(plotpath); end

get_sol(prob) = Array(prob.sol) # extracts the Fourier-transformed solution

function get_u(prob)
  vars, grid, sol = prob.vars, prob.grid, prob.sol

  @. vars.qh = sol

  SingleLayerQG.streamfunctionfrompv!(vars.ψh, vars.qh, params, grid)

  ldiv!(vars.u, grid.rfftplan, -im * grid.l .* vars.ψh)

  return Array(vars.u)
end

output = Output(prob, filename, (:qh, get_sol), (:u, get_u))
saveproblem(output)
saveoutput(output)

startwalltime = time()

while clock.step <= nsteps
  if clock.step % 50save_substeps == 0
    cfl = clock.dt * maximum([maximum(vars.u) / grid.dx, maximum(vars.v) / grid.dy])

    log = @sprintf("step: %04d, t: %d, cfl: %.2f, E: %.4f, Q: %.4f, walltime: %.2f min",
    clock.step, clock.t, cfl, E.data[E.i], Z.data[Z.i], (time()-startwalltime)/60)

    println(log)
  end

  stepforward!(prob, diags, save_substeps)
  SingleLayerQG.updatevars!(prob)

  if clock.step % save_substeps == 0
    saveoutput(output)
  end
end

savediagnostic(E, "energy", output.path)
savediagnostic(Z, "enstrophy", output.path)
file = jldopen(output.path)

iterations = parse.(Int, keys(file["snapshots/t"]))
t = [file["snapshots/t/$i"] for i ∈ iterations]

qh = [file["snapshots/qh/$i"] for i ∈ iterations]
u  = [file["snapshots/u/$i"] for i ∈ iterations]

E_t = file["diagnostics/energy/t"]
Z_t = file["diagnostics/enstrophy/t"]
E_data = file["diagnostics/energy/data"]
Z_data = file["diagnostics/enstrophy/data"]

x,  y  = file["grid/x"],  file["grid/y"]
nx, ny = file["grid/nx"], file["grid/ny"]
Lx, Ly = file["grid/Lx"], file["grid/Ly"]

close(file)
n = Observable(1)

qₙ = @lift irfft(qh[$n], nx)
ψₙ = @lift irfft(- Array(grid.invKrsq) .* qh[$n], nx)
q̄ₙ = @lift real(ifft(qh[$n][1, :] / ny))
ūₙ = @lift vec(mean(u[$n], dims=1))

title_q = @lift @sprintf("vorticity, μt = %.2f", μ * t[$n])

energy    = Observable([Point2f(E_t[1], E_data[1])])
enstrophy = Observable([Point2f(Z_t[1], Z_data[1])])

fig = Figure(size = (1000, 600))

axis_kwargs = (xlabel = "x",
               ylabel = "y",
               aspect = 1,
               limits = ((-Lx/2, Lx/2), (-Ly/2, Ly/2)))


frames = 1:length(t)
record(fig, "yeah.mp4", frames, framerate = 16) do i
  n[] = i

  energy[]    = push!(energy[],    Point2f(μ * E_t[i], E_data[i]))
  enstrophy[] = push!(enstrophy[], Point2f(μ * Z_t[i], Z_data[i]))
end