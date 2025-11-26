try:
    import h5py
except ModuleNotFoundError:
    h5py = None

try:
    import zarr
except ModuleNotFoundError:
    zarr = None

import numpy as np
from collections.abc import Mapping
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

import importlib
import model.inversion
importlib.reload(model.inversion)
from .inversion import PoissonSolver

from .grid import Grid
from .pytree import PytreeNode

optional = object()
required = object()

class IOInterface:
    def __init__(self, h):
        self._h = h

    @property
    def h(self):
        return self._h

    @property
    def attrs(self):
        return self.h.attrs

    def create_group(self, name):
        return IOInterface(self.h.create_group(name))

    def create_dataset(self, name, shape=None, dtype=None, data=None):
        if shape is None and data is not None:
            shape = data.shape
        if dtype is None and data is not None:
            dtype = data.dtype

        if h5py is not None and isinstance(self.h, (h5py.File, h5py.Group)):
            self.h.create_dataset(name, shape=shape, dtype=dtype, data=data)
        elif zarr is not None and isinstance(self.h, zarr.Group):
            a = self.h.create_array(name, shape=shape, dtype=dtype)
            if data is not None:
                a[:] = data
        else:
            raise TypeError(f"Unexpected type: '{type(self.h)}'")


class Parameters(Mapping):
    def __init__(self, parameters, *, defaults=None):
        """to be edited """
        parameters = dict(parameters)
        if defaults is not None:
            for key, value in defaults.items():
                if key not in parameters and value is required:
                    raise KeyError(f"Missing parameter: '{key}'")
                if value is not optional:
                    parameters.setdefault(key, value)
            for key in parameters:
                if key not in defaults:
                    pass #raise KeyError(f"Extra parameter: '{key}'")
        self._parameters = parameters

    def __getitem__(self, key):
        return self._parameters[key]

    def __iter__(self):
        yield from sorted(self._parameters)

    def __len__(self):
        return len(self._parameters)

    def write(self, h, path="parameters"):
        h = IOInterface(h)
        g = h.create_group(path)
        g.attrs.update(self.items())
        return g.h

    @classmethod
    def read(cls, h, path="parameters"):
        return cls(h[path].attrs)
    

class States(Mapping):
    """Holds the evolving state of the QG model for JAX-functional stepping."""
    def __init__(self, grid, states):
        states = tuple(states)
        if len(set(states)) != len(states):
            raise ValueError("Duplicate state")

        self._grid = grid
        self._states = set(states)
        self._fields = {}

    def __getitem__(self, state):
        if state not in self._states:
            raise KeyError(f"Invalid state: '{state}'")
        if state not in self._fields:
            raise ValueError(f"Uninitialized value for state: '{state}'")
        return self._fields[state]

    def __setitem__(self, state, value):
        if state not in self._states:
            raise KeyError(f"Invalid state: '{state}'")
        if value.shape != (self.grid.ny, self.grid.nx):
            raise ValueError(f"Invalid value for state: '{state}'")
        self._fields[state] = value

    def __iter__(self):
        yield from sorted(self._states)

    def __len__(self):
        return len(self._states)

    @property
    def grid(self) -> Grid:
        """The 2D grid.
        """

        return self._grid

    def zero(self, *states):
        """Set fields equal to a zero-valued field.

        Parameters
        ----------

        states : tuple
            The states of fields to set equal to a zero-valued field.
        """

        for state in states:
            self[state] = jnp.zeros((self.grid.ny, self.grid.nx))

    def clear(self, *, keep_states=None):
        """Clear values for fields.

        Parameters
        ----------

        keep_states : Iterable
            states for fields which should be retained.
        """

        if keep_states is None:
            keep_states = set()
        else:
            keep_states = set(keep_states)
        keep_states = sorted(keep_states)

        fields = {state: self[state] for state in keep_states}
        self._fields.clear()
        self.update(fields)

    def write(self, h, path="fields"):
        """Write fields.

        Parameters
        ----------

        h : :class:`h5py.Group` or :class:`zarr.hierarchy.Group`
            Parent group.
        path : str
            Group path.

        Returns
        -------

        :class:`h5py.Group` or :class:`zarr.hierarchy.Group`
            Group storing the fields.
        """

        if not np.can_cast(self.grid.Lx, float):
            raise ValueError("Serialization not supported")
        if not np.can_cast(self.grid.Ly, float):
            raise ValueError("Serialization not supported")
        if not np.can_cast(self.grid.nx, int):
            raise ValueError("Serialization not supported")
        if not np.can_cast(self.grid.ny, int):
            raise ValueError("Serialization not supported")

        h = IOInterface(h)
        g = h.create_group(path)
        del h
        g.attrs["Lx"] = float(self.grid.Lx)
        g.attrs["Ly"] = float(self.grid.Ly)
        g.attrs["nx"] = int(self.grid.nx)
        g.attrs["ny"] = int(self.grid.ny)

        for state, value in self.items():
            g.create_dataset(
                name=state, data=np.array(value))

        return g.h

    @classmethod
    def read(cls, h, path="fields", *, grid=None):
        """Read fields.

        Parameters
        ----------

        h : :class:`h5py.Group` or :class:`zarr.hierarchy.Group`
            Parent group.
        path : str
            Group path.
        grid : :class:`.Grid`
            The 2D grid.

        Returns
        -------

        :class:`.Fields`
            The fields.
        """

        g = h[path]
        del h

        Lx = g.attrs["Lx"]
        Ly = g.attrs["Ly"]
        nx = g.attrs["nx"]
        ny = g.attrs["ny"]
        if grid is None:
            grid = Grid(Lx, Ly, nx, ny)
        if Lx != grid.Lx or Ly != grid.Ly:
            raise ValueError("Invalid dimension(s)")
        if nx != grid.nx or ny != grid.ny:
            raise ValueError("Invalid number(s) of divisions")

        fields = cls(grid, set(g))
        for state in g:
            fields[state] = g[state][...]

        return fields

    def update(self, d):
        """Update field values from the supplied :class:`Mapping`.

        Parameters
        ----------

        d : Mapping
            Key-value pairs containing the field values.
        """

        for state, value in d.items():
            self[state] = value


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
        self.epsilon = params.get('epsilon', 1e-4) # target energy injection rate


        forcing_spectrum = jnp.exp(- (grid.Kmag - self.k_f)**2 / (2 * self.k_width**2))
        forcing_spectrum = jnp.where(grid.K2 == 0, 0.0, forcing_spectrum)

        eps0 = jnp.sum(forcing_spectrum * grid.invK2 / 2) / (grid.Lx * grid.Ly)
        self.forcing_spectrum = forcing_spectrum * (self.epsilon / eps0)

        # === Downsampling initial field for low res ===
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
        self.initialize() # removes irrelevant states from another run - check this is strictly necessary

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

    @cached_property
    def poisson_solver(self):
        return PoissonSolver(self.grid)

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
        """Take multiple timesteps. Uses :func:`jax.lax.fori_loop`.

        Parameters
        ----------

        n : Integral
            The number of timesteps to take.
        unroll : Integral
            Passed to :func:`jax.lax.fori_loop`.
        """

        model = jax.lax.fori_loop(0, n, self._step, self, unroll=unroll)
        self.update(model)

    def _make_initial(self, params):
        """Create a band-passed random initial condition in physical space."""
        key = params.get('key', jax.random.PRNGKey(0)) # check this call is fine if the initial key is not inputted so the downsampling relies on the same .get 
        key, k1 = jax.random.split(key, 2)

        grid = self.grid
        k_beta = params.get('k_beta', 1.0)

        # actual pseudo-random initial spectrum peaked at k_beta
        qh = Solver.pseudo_randomiser(grid, k_beta, k1, energy=100.0)
        qh = Solver.dealias(qh, grid, s=8)

        kmin = 4
        kmax = 15 #hardcoded
        band_mask = (grid.Kmag >= kmin) & (grid.Kmag <= kmax)
        qh = qh * band_mask
        qh = qh.at[:, 0].set(0.0) 

        E_spec = 0.5 * jnp.sum(jnp.abs(qh)**2 * grid.invK2) / (grid.Lx * grid.Ly)
        E0 = 5e5  # target initial energy (tune as needed)
        scale = jnp.sqrt(E0 / (E_spec + 1e-16))
        qh = qh * scale

        #  physical-space initial field
        initial = irfftn(qh, axes=(-2, -1)).real
        assert initial.shape == (grid.ny, grid.nx)
        return initial

    
    @staticmethod
    @jax.jit
    def dealias(y, grid, s=8):
        '''this is now component-wise cutoff, with exponential decay after 2/3. The amplitude
        is entirely made up and check it. The top corner will have exponential decay twice bc of x then y masks'''
        kxmax = jnp.max(grid.kx)
        kymax = jnp.max(grid.ky)
        kxcut = 2/3 * kxmax
        kycut = 2/3 * kymax
        ax = -jnp.log(1e-15) / ((kxmax - kxcut)**s) # this is the exponential dropoff, check this later
        ay = -jnp.log(1e-15) / ((kymax - kycut)**s) # these should be the same for now.
        mask_x = jnp.where(
            jnp.abs(grid.kx) <= kxcut,
            1.0,
            jnp.exp(-ax * (jnp.abs(grid.kx) - kxcut)**s)
        )

        mask_y = jnp.where(
            jnp.abs(grid.ky) <= kycut,
            1.0,
            jnp.exp(-ay * (jnp.abs(grid.ky) - kycut)**s)
        )

        mask = mask_y[:, None] * mask_x[None, :]

        return y * mask
    
    @staticmethod
    @jax.jit
    def RK4(state, grid, params, forcing_spectrum):
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
        key, k_noise = jax.random.split(key, 2)

        # PV to velocity in Fourier space
        psih = -qh * grid.invK2
        uh = -1j * grid.KY * psih
        vh =  1j * grid.KX * psih

        # Back to physical space; take real part to avoid accumulation of tiny imaginary parts
        u, v, q = irfftn(jnp.stack([uh, vh, qh], axis=0), axes=(-2, -1)).real

        # Compute derivatives in physical space via spectral method
        dqdx = irfftn(1j * grid.KX * rfftn(q, axes=(-2,-1)), axes=(-2,-1)).real
        dqdy = irfftn(1j * grid.KY * rfftn(q, axes=(-2,-1)), axes=(-2,-1)).real

        # Skew-symmetric form
        nonlinear_phys = 0.5 * (u * dqdx + v * dqdy + dqdx * u + dqdy * v)  #0.5*(u·∇q + ∇·(uq))

        # Transform back to Fourier space and apply dealias filter
        nonlinear = rfftn(nonlinear_phys, axes=(-2,-1))
        nonlinear = Solver.dealias(nonlinear, grid, s=8)

        # Beta term
        beta_term = - params.get('beta', 1e-3) * 1j * grid.KX * psih

        # --- Forcing ---
        # Draw a real-space white-noise field and transform to spectral space so the
        # forcing respects Hermitian symmetry (real physical fields after iFFT).
        noise_phys = jax.random.normal(k_noise, (grid.ny, grid.nx))
        forcing_h = rfftn(noise_phys, axes=(-2, -1))

        if forcing_spectrum.shape != qh.shape:
            # (ny, nx) -> (ny, nx//2+1)
            if forcing_spectrum.ndim == 2 and forcing_spectrum.shape[0] == grid.ny and forcing_spectrum.shape[1] == grid.nx:
                forcing_spectrum = forcing_spectrum[:, : qh.shape[1]]
            else:
                raise ValueError(f"forcing_spectrum shape {forcing_spectrum.shape} incompatible with qh shape {qh.shape}")
        forcing = forcing_h * forcing_spectrum
        forcing = Solver.dealias(forcing, grid, s=8)

        # --- Dissipation ---
        nu = params.get("nu", 0.0)
        m  = params.get("m", 4)
        mu = params.get("mu", 0.0)

        hypervisc = nu * (grid.Kmag ** m) * qh # hyperviscosity (small scale)
        drag = mu * qh # linear drag (large scale)
        
        dissipation = drag + hypervisc

        rhs =  nonlinear + beta_term - dissipation + forcing

        # --- Pause to check im happy ---
        # spectral kinetic energy estimate
        ke_spec = 0.5 * jnp.sum(jnp.abs(qh) ** 2 * grid.invK2)
        max_qh = jnp.max(jnp.abs(qh))
        max_forcing = jnp.max(jnp.abs(forcing))
        max_nonlinear = jnp.max(jnp.abs(nonlinear))
        max_beta = jnp.max(jnp.abs(beta_term))
        max_diss = jnp.max(jnp.abs(dissipation))

        cond = jnp.logical_or(jnp.isnan(ke_spec), ke_spec > 1e6)

        def _print(_):
            pass
            #jax.debug.print("DIAG RHS: KE={} max_qh={} max_forcing={} max_nonlinear={} max_beta={} max_diss={}",
            #               ke_spec, max_qh, max_forcing, max_nonlinear, max_beta, max_diss)

        jax.lax.cond(cond, _print, lambda _: None, operand=None)
        assert u.shape == (grid.ny, grid.nx)
        assert qh.shape == (grid.ny, grid.nx//2 + 1), f"qh shape {qh.shape} unexpected"
        # sanity-check shapes after dealias operations
        assert qh.shape == (grid.ny, grid.nx//2 + 1), f"qh shape {qh.shape} unexpected"

        return rhs, key
    
    def pseudo_randomiser(grid, k_peak, key, energy=0.08):
        '''This is a method to make Gaussian noise align better with the forcing wavenumber
        It returns qh right now, NOT dealiased!!!
        '''
        k0 = k_peak * 2 * jnp.pi / grid.Lx
        key_r, key_i = jax.random.split(key)

        modpsi = (grid.Kmag**2 * (1 + (grid.Kmag / k0)**4))**(-0.5)
        modpsi = modpsi.at[0,0].set(0.0)
        modpsi = jnp.clip(modpsi, a_min=1e-8, a_max=1e8) #is this clipping necessary?

        # this sets the psuedo-random using modpsi scaling
        phase = jax.random.normal(key_r, modpsi.shape) + 1j * jax.random.normal(key_i, modpsi.shape)
        psih = phase * modpsi
        psih = Solver.dealias(psih, grid, s=8)

        # --- kinetic energy normalization ---
        # Note: normalize by number of grid points (not its square). Guard against
        # tiny energy_initial to avoid huge amplification.
        energy_initial = jnp.sum(jnp.asarray(grid.K2, dtype=jnp.float32) * jnp.abs(psih)**2) / float(grid.nx * grid.ny)
        energy_initial = jnp.maximum(energy_initial, 1e-16)
        scale = jnp.sqrt(energy / energy_initial)
        scale = jnp.minimum(scale, 1e6)
        psih = psih * scale

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

    @classmethod
    def read(cls, h, path="solver"):
        """Read solver.

        Parameters
        ----------

        h : :class:`h5py.Group` or :class:`zarr.hierarchy.Group`
            Parent group.
        path : str
            Group path.

        Returns
        -------

        :class:`.Solver`
            The solver.
        """

        g = h[path]
        del h

        cls = cls._registry[g.attrs["type"]]
        model = cls(Parameters.read(g, "parameters"))
        model.fields.update(States.read(g, "fields", grid=model.grid))
        model.n = g.attrs["n"]

        return model

    def new(self, *, copy_prescribed=False):
        """Return a new :class:`.Solver` with the same configuration as this
        :class:`.Solver`.

        Parameters
        ----------

        copy_prescribed : bool
            Whether to copy values of prescribed fields to the new
            :class:`.Solver`.

        Returns
        -------

        :class:`.Solver`
            The new :class:`.Solver`.
        """

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
            outname="outputs/qg_gpu.gif",
            stats=None        # flags for which plots, currently ['zonal', 'energy', 'time_av']
        ):

        # Normalize stats
        if stats is None:
            stats = []
        valid_stats = ["zonal", "energy"]
        plot_stats = [s for s in stats if s in valid_stats]

        # setup figs dynamically 
        n_panels = 1 + len(plot_stats)
        panel_indices = {"vort": 0}

        if "zonal" in plot_stats:
            panel_indices["zonal"] = len(panel_indices)
        if "energy" in plot_stats:
            panel_indices["energy"] = len(panel_indices)

        fig, axs = plt.subplots(
            1, n_panels, figsize=(5 * n_panels, 5),
            constrained_layout=True
        )

        # Always ensure axs is a list-like
        if n_panels == 1:
            axs = [axs]

        ax_vort = axs[panel_indices["vort"]]
        y = self.grid.y
        dx = self.grid.dx
        dy = self.grid.dy

        umean_list = []
        energy_list = []
        time_list = []

        # vorticity (always)
        zeta = np.array(self.fields["zeta"])
        vmin, vmax = float(zeta.min()), float(zeta.max())

        im = ax_vort.imshow(
            zeta, origin="lower",
            cmap="RdBu_r",
            extent=(-self.grid.Lx/2, self.grid.Lx/2,
                    -self.grid.Ly/2, self.grid.Ly/2),
            vmin=vmin, vmax=vmax
        )
        ax_vort.set_title("Vorticity")
        ax_vort.set_xlabel("x")
        ax_vort.set_ylabel("y")

        # zonal mean velocity
        if "zonal" in stats:
            ax_umean = axs[panel_indices["zonal"]]
            line_umean, = ax_umean.plot(np.zeros_like(y), y)
            ax_umean.set_title("Zonal-mean U (time-avg building)")
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

        # update func
        def update(frame):
            nonlocal umean_list, energy_list, time_list

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
            if "zonal" in stats or "energy" in stats or 'time_av' in stats:
                u = (np.roll(psi, -1, 0) - np.roll(psi, 1, 0)) / (2 * dy)
                v = -(np.roll(psi, -1, 1) - np.roll(psi, 1, 1)) / (2 * dx)
            
            # Time averaged final 
            if 'time' in stats:
                ubar = u.mean(axis=1)
                umean_list.append(ubar)

            # zonal mean velocity
            if "zonal" in stats:
                ubar = u.mean(axis=1)
                line_umean.set_xdata(ubar)

            # energy
            if "energy" in stats:
                KE = 0.5 * np.mean(u*u + v*v)
                energy_list.append(KE)
                time_list.append(frame * frame_interval)

                line_energy.set_xdata(time_list)
                line_energy.set_ydata(energy_list)
                ax_energy.relim()
                ax_energy.autoscale_view()

            return [x for x in [im, line_umean, line_energy] if x is not None]
        
        
        frames = nsteps // frame_interval + 1
        anim = FuncAnimation(fig, update, frames=frames, blit=False)
        anim.save(outname, fps=10)
        plt.close(fig)
        print(f"Saved animation to {outname}")

        # --- time-averaged final plot ---
        fig2, ax2 = plt.subplots(figsize=(5,4))
        umean_timeavg = np.mean(np.array(umean_list), axis=0)
        ax2.plot(umean_timeavg, y)
        ax2.set_title("Final Time-mean Zonal Velocity")
        ax2.set_ylabel("y")
        ax2.set_xlabel("Ū(y)")
        ax2.grid(True)
        fig2.savefig("outputs/time_averaged_u.png")
        plt.close(fig2)




    ################## come back to this, wanted to plot the energy spectra similar to Cope. 
    def compute_ke_spectrum(qh, grid):
        """
        Compute isotropic kinetic energy spectrum from qh (spectral vorticity).
        Returns:
            k_vals: integer wavenumbers (1D)
            E_k: isotropic KE spectrum (1D)
        """

        # 2D kinetic energy density in spectral space:
        #   E = 0.5 * |q_h|^2 / k^2     because psi_h = -qh/k^2
        E2D = 0.5 * (jnp.abs(qh)**2 * grid.invK2)   # shape (ny, nx//2+1)

        kmag = grid.Kmag                               # same shape
        kmax = int(jnp.max(kmag))
        k_vals = jnp.arange(kmax + 1)

        # Bin indices: integer shell each point belongs to
        bins = jnp.floor(kmag).astype(int)
        bins = jnp.clip(bins, 0, kmax)

        # Sum energy into bins
        E_k = jnp.zeros(kmax + 1)
        E_k = E_k.at[bins].add(E2D)

        return k_vals, E_k


    def plot_ke(self, nsteps=5000, frame_interval=100,outname="ke_spectrum.png", title="Kinetic Energy Spectrum"):
        """
        Just checking for now
        """
        k_vals_list = []
        E_k_list = []
        frames = nsteps // frame_interval + 1
        for i in jnp.arange(frames):
            self.steps(frame_interval)
            zeta = np.array(self.fields["zeta"])

            k_vals, E_k = Solver.compute_ke_spectrum(zeta, self.grid)
            k_vals_list.append(k_vals)
            E_k_list.append(E_k)

        plt.figure(figsize=(6, 5))
        plt.loglog(k_vals_list, E_k_list, marker="o")   # skip k=0
        plt.xlabel("Wavenumber k")
        plt.ylabel("E(k)")
        plt.title(title)
        plt.grid(True, which="both", ls="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(outname)
        plt.close()

        print(f"[KE Spectrum] Saved to {outname}")
        return k_vals, E_k

