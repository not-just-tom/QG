
import jax 
import jax.numpy as jnp
import numpy as np
from abc import ABC, abstractmethod
from functools import cached_property
import keras
from jax.numpy.fft import rfftn, irfftn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm

from .states import States, IOInterface
from .grid import Grid
from ..utils.pytree import PytreeNode

optional = object()
required = object()

class Solver(PytreeNode, ABC):
    """
    My model to try and get this thing started
    """

    _registry = {}

    _defaults = {"Lx": required,  # x \in [-Lx, Lx]
                 "nx": required,
                 "beta": required,
                 "dt": required}
    
    def __init__(self, params, initial: jnp.ndarray | None = None, grid_input:Grid| None = None, field_states=None, prescribed_field_states=None):
        # ==== Parameter Initialisation ===  
        self._parameters = params #= Parameters(params, defaults=self._defaults)
        # PRNG key for the solver; stored and advanced each step
        self._key = params.get("key", jax.random.PRNGKey(0))

        # ==== Grid and Initial Field Initialisation === 
        """here im not sure we will need the grid input generation as it should probs be match high/low res w intitial cond"""
        if grid_input is None:
            self.grid = grid = Grid(params)
        else:
            self.grid = grid = grid_input


        if initial is None:
            initial = self._make_initial(params)
        self._initial = initial

        # forcing stuff
        self.k_f = params.get('k_f', 8.0)        # central forcing wavenumber
        self.k_width = params.get('k_width', 2.0) # width of the ring

        # ---  (setting energy input for forcing) ---
        L_rh = params.get('L_rh', 1.0) 
        beta = params.get('beta', 10.0)
        self.epsilon = 1e-6 *beta**3*L_rh**5 # target energy injection rate using C_eps = 1e-3

        forcing_spectrum = jnp.exp(- (grid.Kmag - self.k_f)**2 / (2 * self.k_width**2)) # standard Gaussian annulus - im happy w this
        forcing_spectrum = jnp.where(grid.K2 == 0, 0.0, forcing_spectrum)

        eps0 = jnp.sum(forcing_spectrum * grid.invK2 / 2)
        self.forcing_spectrum = forcing_spectrum * (self.epsilon / eps0) # check this

        # === Downsampling initial field for low res === #does this work at all lmao
        if params.get('downsample', None) is None: 
            pass
        else:
            div = params['downsample']
            if grid.nx % div !=0:
                raise ValueError("Subsampling is not indexed correctly")
            initial= initial[::div, ::div]
        self._initial = jax.device_put(initial)

        if field_states is None:
            field_states = set()
        else:
            field_states = set(field_states)
        field_states.update({"psi", "zeta", "Q"})
        if prescribed_field_states is None:
            prescribed_field_states = {"Q"}
        
        self._fields = States(grid, field_states)
        self._prescribed_field_states = tuple(sorted(prescribed_field_states))

        self.zero_prescribed() # zeros the things we're interested in
        self.initialize() # removes irrelevant states from another run

    @property
    def parameters(self):
        return self._parameters

    @property
    def grid(self):
        return self._grid

    @grid.setter
    def grid(self, g):
        self._grid = g

    @property
    def fields(self):
        return self._fields
    
    @property
    def n(self):
        return self._n
    
    @property
    def initial(self):
        """Return the initial condition in physical space."""
        return self._initial

    @n.setter
    def n(self, n):
        self._n = n

    @cached_property
    def beta(self):
        return jnp.array(self.parameters["beta"])
    
    @cached_property
    def dt(self):
        return jnp.array(self.parameters["dt"])

    @property
    def prescribed_field_states(self):
        return self._prescribed_field_states

    def zero_prescribed(self):
        """Zero prescribed fields.
        """

        self.fields.zero(*self.prescribed_field_states)

    @abstractmethod
    def initialize(self, zeta=None):
        """Initialize the model"""

        self.fields.clear(keep_states=self.prescribed_field_states)
        self._n = 0

    @abstractmethod 
    def step(self):
        """Take a timestep.
        """

        self._n += 1

    @staticmethod
    @jax.jit 
    def _step(_, model):
        model.step()
        return model

    def steps(self, n, *, unroll=8): 
        model = jax.lax.fori_loop(0, n, self._step, self, unroll=unroll)
        self.update(model)

    def _make_initial(self, params):
        """Create a band-passed random initial condition in physical space."""
        key = params.get('key', jax.random.PRNGKey(0)) # check this call is fine if the initial key is not inputted so the downsampling relies on the same .get 
        key, k1 = jax.random.split(key, 2)

        grid = self.grid
        k_f = params.get('k_f', 16.0)
        kmin = params.get('kmin', 4.0)
        kmax = params.get('kmax', 10.0)

        # pseudo-random initial spectrum peaked at k_f
        qh = Solver.pseudo_randomiser(grid, k_f, k1)
        qh = Solver.dealias(qh, grid, s=8)

        # --- band-pass filter ---
        band_mask = (grid.Kmag >= kmin) & (grid.Kmag <= kmax)
        qh = qh * band_mask
        qh = qh.at[:, 0].set(0.0)

        #  physical-space initial field
        initial = irfftn(qh, axes=(-2, -1), norm='ortho').real
        assert initial.shape == (grid.ny, grid.nx)
        return initial

    
    @staticmethod
    @jax.jit
    def dealias(y, grid, s=8):
        """Apply a precomputed dealias mask from the grid if available.

        Falls back to a component-wise 2/3 rule mask if `grid.dealias_mask` is
        not present.
        """
        # Use precomputed mask when available (fast and JAX-friendly)
        mask = getattr(grid, "dealias_mask", None)
        if mask is None:
            # Fallback 2/3 rule mask- this needs to be exponential soon
            kxmax = jnp.max(jnp.abs(grid.kx))
            kymax = jnp.max(jnp.abs(grid.ky))
            kxcut = 2/3 * kxmax
            kycut = 2/3 * kymax
            mask_x = jnp.where(jnp.abs(grid.kx) <= kxcut, 1.0, 0.0)
            mask_y = jnp.where(jnp.abs(grid.ky) <= kycut, 1.0, 0.0)
            mask = mask_y[:, None] * mask_x[None, :]
        return y * mask
    
    @staticmethod
    @jax.jit
    def RK4(state, grid, params, forcing_spectrum, ml_forcing=None, dt=None):
        """RK4 stepping for a single timestep.

        Parameters
        - state: (qh, key)
        - params: parameter dict (may contain default dt)
        - forcing_spectrum: spectral forcing ring
        - ml_forcing: optional callable(qh, key, grid, params) -> spectral forcing (qh-shaped)
        - dt: optional timestep override
        """
        qh, key = state
        # Draw one noise key per timestep and reuse it for all RK stages
        key, k_noise = jax.random.split(key, 2)
        dt = params['dt']

        rhs1, key = Solver.rhs(qh, k_noise, grid, params, forcing_spectrum)
        rhs2, key = Solver.rhs(qh + 0.5 * dt * rhs1, k_noise, grid, params, forcing_spectrum)
        rhs3, key = Solver.rhs(qh + 0.5 * dt * rhs2, k_noise, grid, params, forcing_spectrum)
        rhs4, key = Solver.rhs(qh + dt * rhs3, k_noise, grid, params, forcing_spectrum)
        qh_new = qh + (dt/6.0)*(rhs1 + 2*rhs2 + 2*rhs3 + rhs4)
        return (qh_new, key)
    
    @staticmethod
    @jax.jit
    def rhs(qh, key, grid, params, forcing_spectrum):
        # Use the provided key to draw a single real-space noise field per timestep
        # and reuse it for all spectral forcing to ensure Hermitian symmetry.
        key, key_noise = jax.random.split(key, 2)

        # PV to velocity in Fourier space
        psih = -qh * grid.invK2
        uh = -1j * grid.KY * psih
        vh =  1j * grid.KX * psih

        # Back to physical space, avoiding imaginary components(?)
        u, v, q = irfftn(jnp.stack([uh, vh, qh], axis=0), axes=(-2, -1), norm='ortho').real

        # Compute derivatives in physical space via spectral method
        dqdx = irfftn(1j * grid.KX * rfftn(q, axes=(-2,-1), norm='ortho'), axes=(-2,-1), norm='ortho').real
        dqdy = irfftn(1j * grid.KY * rfftn(q, axes=(-2,-1), norm='ortho'), axes=(-2,-1), norm='ortho').real

        # Skew-symmetric form
        nonlinear_phys = 0.5 * (u * dqdx + v * dqdy + dqdx * u + dqdy * v)  #0.5*(u·∇q + ∇·(uq))

        # Transform back to Fourier space and apply dealias filter
        nonlinear = rfftn(nonlinear_phys, axes=(-2,-1), norm='ortho')
        nonlinear = Solver.dealias(nonlinear, grid, s=8)

        # Beta term
        beta_term = - params.get('beta', 1e-3) * 1j * grid.KX * psih

        # --- forcing (stochastic) ---
        noise_phys = jax.random.normal(key_noise, (grid.ny, grid.nx))
        forcing_h = rfftn(noise_phys, axes=(-2, -1), norm='ortho')

        if forcing_spectrum.shape != qh.shape:
            # (ny, nx) -> (ny, nx//2+1)
            if forcing_spectrum.ndim == 2 and forcing_spectrum.shape[0] == grid.ny and forcing_spectrum.shape[1] == grid.nx:
                forcing_spec = forcing_spectrum[:, : qh.shape[1]]
            else:
                raise ValueError(f"forcing_spectrum shape {forcing_spectrum.shape} incompatible with qh shape {qh.shape}")
        forcing = forcing_h * forcing_spectrum
        forcing = Solver.dealias(forcing, grid, s=8)

        rhs =  nonlinear + beta_term + forcing # forcing needs building out

        # --- Shape check ---
        assert u.shape == (grid.ny, grid.nx)
        assert qh.shape == (grid.ny, grid.nx//2 + 1), f"qh shape {qh.shape} unexpected"
        assert qh.shape == (grid.ny, grid.nx//2 + 1), f"qh shape {qh.shape} unexpected"

        return rhs, key
    
    def pseudo_randomiser(grid, k_peak, key):
        '''This is a method to make noise align better with the forcing wavenumber
        It returns qh right now, IS dealiased!!! No normalisation!!!!
        '''
        k0 = k_peak * 2 * jnp.pi / grid.Lx
        key_r, key_i = jax.random.split(key)

        modpsi = (grid.Kmag**2 * (1 + (grid.Kmag / k0)**4))**(-0.5)
        modpsi = modpsi.at[0,0].set(0.0)
        modpsi = jnp.clip(modpsi, a_min=1e-8, a_max=1e8) # is this clipping necessary?

        # this sets the psuedo-random using modpsi scaling
        phase = jax.random.normal(key_r, modpsi.shape) + 1j * jax.random.normal(key_i, modpsi.shape)
        psih = phase * modpsi
        psih = Solver.dealias(psih, grid, s=8)

        qh = -grid.K2 * psih
        qh = qh.at[:, 0].set(jnp.real(qh[:, 0]))

        return qh

    def write(self, h, path="solver"):
        """Write solver.

        Parameters
        ----------

        h : :class:`h5py.Group` or :class:`zarr.hierarchy.Group`
            Parent group.
        path : str
            Group path.

        Returns
        -------

        :class:`h5py.Group` or :class:`zarr.hierarchy.Group`
            Group storing the solver.
        """

        h = IOInterface(h)
        g = h.create_group(path)
        del h

        g.attrs["type"] = type(self).__name__
        g.attrs["n"] = int(self.n)
        self.parameters.write(g.h, "parameters")
        self.fields.write(g.h, "fields")

        return g.h

# the read method ive blanked bc i removed parameters 
#    @classmethod
#    def read(cls, h, path="solver"):
#        g = h[path]
#        del h
#
#        cls = cls._registry[g.attrs["type"]]
#        model = cls(Parameters.read(g, "parameters"))
#        model.fields.update(States.read(g, "fields", grid=model.grid))
#        model.n = g.attrs["n"]
#
#        return model

    def new(self, *, copy_prescribed=False):
        model = type(self)(self.parameters)
        if copy_prescribed:
            for state in self.prescribed_field_states:
                model.fields[state] = self.fields[state]
        return model

    def update(self, model):
        self.fields.update(model.fields)
        self.n = model.n

    def flatten(self):
        """Return a JAX flattened representation.

        Returns
        -------

        Sequence[Sequence[object, ...], Sequence[object, ...]]
        """

        return ((dict(self.fields), self.n),
                (self.parameters))

    @classmethod
    def unflatten(cls, aux_data, children):
        """Unpack a JAX flattened representation.
        """

        parameters = aux_data
        fields, n = children

        model = cls(parameters)
        model.fields.update({key: value for key, value in fields.items() if type(value) is not object})
        if type(n) is not object:
            model.n = n

        return model

    def get_config(self):
        return {"type": type(self).__name__,
                "parameters": dict(self.parameters),
                "fields": dict(self.fields),
                "n": self.n}

    def get_forcing_stats(self):
        """Return quick diagnostics for the forcing generated from the current key.

        Returns a dict with max/mean in physical and spectral space. Does not
        advance the stored solver key (uses a split of the stored key).
        """
        key = getattr(self, '_key', jax.random.PRNGKey(0))
        key, k_noise = jax.random.split(key, 2)
        noise_phys = jax.random.normal(k_noise, (self.grid.ny, self.grid.nx))
        forcing_h = rfftn(noise_phys, axes=(-2, -1), norm='ortho')
        # align shapes
        fs = self.forcing_spectrum
        if fs.shape != forcing_h.shape:
            fs = fs[:, : forcing_h.shape[1]]
        forcing = forcing_h * (jnp.sqrt(fs) / jnp.sqrt(self.dt))
        forcing_phys = irfftn(forcing, axes=(-2, -1), norm='ortho').real
        return {
            'f_spec_max': float(jnp.max(jnp.abs(forcing))),
            'f_spec_mean': float(jnp.mean(jnp.abs(forcing))),
            'f_phys_max': float(jnp.max(jnp.abs(forcing_phys))),
            'f_phys_mean': float(jnp.mean(jnp.abs(forcing_phys)))
        }

    @classmethod
    def from_config(cls, config):
        config = {key: keras.saving.deserialize_keras_object(value) for key, value in config.items()}
        cls = cls._registry[config["type"]]
        model = cls(config["parameters"])
        model.fields.update(config["fields"])
        model.n = config["n"]
        return model

    def plot_field_at_step(self, step=0, frame_interval=1, vmin=None, vmax=None):
        if step > 0:
            self.steps(step * frame_interval)

        zeta = np.array(self.fields["zeta"])
        grid = self.grid
        nan_count = jnp.isnan(zeta).sum()
        print(f"Number of NaNs (black squares) at step {step}: {nan_count}")

        if vmin is None:
            vmin = jnp.nanmin(zeta)
        if vmax is None:
            vmax = jnp.nanmax(zeta)

        cmap = cm.get_cmap('RdBu_r').copy()
        cmap.set_bad(color='black')
        zeta_masked = np.ma.masked_invalid(zeta)

        plt.figure(figsize=(6,5))
        im = plt.imshow(
            zeta_masked,
            origin='lower',
            cmap=cmap,
            extent=(-grid.Lx/2, grid.Lx/2, -grid.Ly/2, grid.Ly/2),
            vmin=vmin,
            vmax=vmax
        )
        plt.colorbar(im, label=r'$\zeta$')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Vorticity Field at Step {step}')
        plt.show()

    def make_animation(
            self, 
            nsteps=5000, 
            frame_interval=100,
            outname="../outputs/qg_gpu.gif",
            stats=None        # flags for which plots, currently ['zonal', 'energy', 'enstrophy', ke_spec]
        ):

        # Normalize stats
        if stats is None:
            stats = []
        valid_stats = ["zonal", "energy", 'enstrophy', 'kespec']
        stats = [s for s in stats if s in valid_stats]

        # setup figs dynamically 
        n_panels = 1 + len(stats)
        panel_indices = {"vort": 0}

        if "zonal" in stats:
            panel_indices["zonal"] = len(panel_indices)
        if "energy" in stats:
            panel_indices["energy"] = len(panel_indices)
        if "enstrophy" in stats:
            panel_indices["enstrophy"] = len(panel_indices)
        if "kespec" in stats:
            panel_indices["kespec"] = len(panel_indices)

        # Compute subplot grid
        if n_panels == 1:
            nrows, ncols = 1, 1
        else:
            nrows = 2
            ncols = int(np.ceil(n_panels / 2))

        fig, axs = plt.subplots(
            nrows, ncols,
            figsize=(5 * ncols, 5 * nrows),
            constrained_layout=False
        )

        axs = np.ravel(axs)

        # Always ensure axs is a list-like
        if n_panels == 1:
            axs = [axs]

        ax_vort = axs[panel_indices["vort"]]
        y = self.grid.y
        dx = self.grid.dx
        dy = self.grid.dy

        umean_list = []
        energy_list = []
        enstrophy_list = []
        ke_list = []
        time_list_energy = []
        time_list_enstro = []
        time_list_ke = []

        # vorticity (always)
        zeta = np.array(self.fields["zeta"])
        vmin, vmax = float(zeta.min()), float(zeta.max())

        im = ax_vort.imshow(
            zeta, origin="lower",
            cmap="RdBu_r",
            extent=(-self.grid.Lx/2, self.grid.Lx/2,
                    -self.grid.Ly/2, self.grid.Ly/2),
            animated=True #vmin=vmin, vmax=vmax
        )
        ax_vort.set_title("Vorticity")
        ax_vort.set_xlabel("x")
        ax_vort.set_ylabel("y")

        # zonal mean velocity
        if "zonal" in stats:
            ax_umean = axs[panel_indices["zonal"]]
            line_umean, = ax_umean.plot(np.zeros_like(y), y)
            ax_umean.set_title("Zonal-mean U")
            ax_umean.set_xlabel("Ū(y)")
            ax_umean.set_ylabel("y")
            ax_umean.grid(True)
        else:
            line_umean = None

        # energy
        if "energy" in stats:
            ax_energy = axs[panel_indices["energy"]]
            line_energy, = ax_energy.plot([], [])
            ax_energy.set_title("Energy vs Time")
            ax_energy.set_xlabel("Step")
            ax_energy.set_ylabel("Energy")
            ax_energy.grid(True)
        else:
            line_energy = None

        # enstrophy
        if "enstrophy" in stats:
            ax_enstrophy = axs[panel_indices["enstrophy"]]
            line_enstrophy, = ax_enstrophy.plot([], [])
            ax_enstrophy.set_title("Enstrophy vs Time")
            ax_enstrophy.set_xlabel("Step")
            ax_enstrophy.set_ylabel("Enstrophy")
            ax_enstrophy.grid(True)
        else:
            line_enstrophy = None

        # KE spectrum
        if "kespec" in stats:
            ax_spec = axs[panel_indices["kespec"]] if "kespec" in panel_indices else axs[-1]
            line_spec, = ax_spec.loglog([], [])
            ax_spec.set_title("KE Spectrum")
            ax_spec.set_xlabel("k")
            ax_spec.set_ylabel("E(k)")
            ax_spec.grid(True, which='both')
        else:
            line_spec = None

        # update func
        def update(frame):
            nonlocal umean_list, energy_list, enstrophy_list, ke_list, time_list_energy, time_list_enstro, time_list_ke

            if frame == 0:
                zeta = np.array(self.fields["zeta"])
                psi = np.array(self.fields["psi"])
            else:
                self.steps(frame_interval)
                zeta = np.array(self.fields["zeta"])
                psi = np.array(self.fields["psi"])

            # vorticity update
            im.set_array(zeta)
            ax_vort.set_title(f"Vorticity (step {frame * frame_interval})")

            # Compute velocities if needed
            if "zonal" in stats or "energy" in stats or "enstrophy" in stats:
                u = -(np.roll(psi, -1, 0) - np.roll(psi, 1, 0)) / (2 * dy)
                v = (np.roll(psi, -1, 1) - np.roll(psi, 1, 1)) / (2 * dx)
            
            # zonal mean velocity
            if "zonal" in stats:
                ubar = u.mean(axis=1)
                line_umean.set_xdata(ubar)
                ax_umean.relim()
                ax_umean.autoscale_view()
                umean_list.append(ubar)

            # energy
            if "energy" in stats:
                KE = 0.5 * np.mean(u**2 + v**2)*dx*dy
                energy_list.append(KE)
                time_list_energy.append(frame * frame_interval)

                line_energy.set_xdata(time_list_energy)
                line_energy.set_ydata(energy_list)
                ax_energy.relim()
                ax_energy.autoscale_view()

            # enstrophy
            if "enstrophy" in stats:
                enstro = 0.5 * np.mean(zeta**2)*dx*dy
                enstrophy_list.append(enstro)
                time_list_enstro.append(frame * frame_interval)

                line_enstrophy.set_xdata(time_list_enstro)
                line_enstrophy.set_ydata(enstrophy_list)
                ax_enstrophy.relim()
                ax_enstrophy.autoscale_view()

            if "kespec" in stats:
                E_k, kbins = Solver.compute_ke_spectrum(psi, self.grid)
                ke_list.append(E_k)
                time_list_ke.append(kbins)
                line_spec.set_ydata(E_k[1:])
                line_spec.set_xdata(kbins[1:])
                ax_spec.relim()
                ax_spec.autoscale_view()

            return [x for x in [im, line_umean, line_energy, line_enstrophy, line_spec] if x is not None]


        frames = nsteps // frame_interval + 1
        anim = FuncAnimation(fig, update, frames=frames, blit=False)
        anim.save(outname, fps=10)
        plt.close(fig)
        print(f"Saved animation to {outname}")

        # Compute subplot for final plots 
        if n_panels <4:
            nrows, ncols = 1, n_panels
        else:
            nrows = 2
            ncols = int(np.ceil(n_panels / 2))

        fig2, axs2 = plt.subplots(
            nrows, ncols,
            figsize=(5 * ncols, 5 * nrows),
            constrained_layout=True
        )

        axs2 = np.ravel(axs2)

        # --- final: time-averaged zonal velocity ---
        if "zonal" in stats:
            ax2_umean = axs2[panel_indices["zonal"]]
            umean_timeavg = np.mean(np.array(umean_list), axis=0)
            ax2_umean.plot(umean_timeavg, y)
            ax2_umean.set_title("Final Time-averaged Zonal Velocity")
            ax2_umean.set_ylabel("y")
            ax2_umean.set_xlabel("Ū(y)")
            ax2_umean.grid(True)

        # --- final: energy vs time ---
        if "energy" in stats:
            ax2_energy = axs2[panel_indices["energy"]]
            ax2_energy.plot(time_list_energy, energy_list)
            ax2_energy.set_title("Energy vs Time (Final)")
            ax2_energy.set_xlabel("Step")
            ax2_energy.set_ylabel("Energy")
            ax2_energy.grid(True)

        # --- final: enstrophy vs time ---
        if "enstrophy" in stats:
            ax2_enstro = axs2[panel_indices["enstrophy"]]
            ax2_enstro.plot(time_list_enstro, enstrophy_list)
            ax2_enstro.set_title("Enstrophy vs Time (Final)")
            ax2_enstro.set_xlabel("Step")
            ax2_enstro.set_ylabel("Enstrophy")
            ax2_enstro.grid(True)

        if "kespec" in stats:
            ke_stack = np.vstack(ke_list)  
            E_k_timeavg = ke_stack.mean(axis=0)
            k_vals = np.arange(E_k_timeavg.size)

            ax2_kespec = axs2[panel_indices["kespec"]]
            ax2_kespec.loglog(k_vals[1:], E_k_timeavg[1:])
            ax2_kespec.set_title("Time-Averaged Energy Spectra")
            ax2_kespec.set_xlabel("k")
            ax2_kespec.set_ylabel("E(k)")
            ax2_kespec.grid(True)

        # Save all final panels together
        fig2.savefig("../outputs/final_summary.png")
        plt.close(fig2)

    @staticmethod
    def compute_ke_spectrum(psi, grid):
        """
        Compute 1D isotropic KE spectrum E(k) using uh, vh
        """
        # Compute velocity in Fourier space
        psih = rfftn(psi, axes=(-2,-1), norm='ortho')
        uh = -1j * grid.KY * psih
        vh =  1j * grid.KX * psih

        # KE density in spectral space
        KE2D = 0.5 * (jnp.abs(uh)**2 + jnp.abs(vh)**2)

        # bin into isotropic shells
        kmax = int(jnp.max(grid.Kmag))
        kbins = jnp.arange(kmax+1)

        # flatten arrays
        kmag_flat = grid.Kmag.flatten().astype(int)
        KE_flat = KE2D.flatten()

        # accumulate into bins
        E_k = jnp.zeros(kmax+1, dtype=float).at[kbins].add(
            jnp.bincount(kmag_flat, weights=KE_flat, length=kmax+1)
        )

        return np.array(E_k), np.array(kbins)


