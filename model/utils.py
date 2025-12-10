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


        forcing_spectrum = jnp.exp(- (grid.Kmag - self.k_f)**2 / (2 * self.k_width**2)) # check this
        forcing_spectrum = jnp.where(grid.K2 == 0, 0.0, forcing_spectrum)

        eps0 = jnp.sum(forcing_spectrum * grid.invK2 / 2) / (grid.Lx * grid.Ly)
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
        kmin = params.get('kmin', 4.0)
        kmax = params.get('kmax', 10.0)

        # pseudo-random initial spectrum peaked at k_beta
        qh = Solver.pseudo_randomiser(grid, k_beta, k1)
        qh = Solver.dealias(qh, grid, s=8)

        # --- band-pass filter ---
        band_mask = (grid.Kmag >= kmin) & (grid.Kmag <= kmax)
        qh = qh * band_mask
        qh = qh.at[:, 0].set(0.0) 

        # --- normalisation ---
        R_beta = params.get('R_beta', 3.0) # this probably isnt a good value - give it a look later
        epsilon = params.get('epsilon', 1e-5)
        beta = params.get('beta', 10.0)
        target_U = jnp.square(jnp.sqrt(2) * jnp.power(epsilon, 0.2) * R_beta / jnp.power(beta, 0.1))


        E_spec = 0.5 * jnp.sum(jnp.abs(qh)**2 * grid.invK2) / (grid.Lx * grid.Ly)
        E0 = 10  # target initial energy (tune as needed)
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
        is entirely made up and I should check it.'''
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

    @classmethod
    def read(cls, h, path="solver"):
        g = h[path]
        del h

        cls = cls._registry[g.attrs["type"]]
        model = cls(Parameters.read(g, "parameters"))
        model.fields.update(States.read(g, "fields", grid=model.grid))
        model.n = g.attrs["n"]

        return model

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
            vmin=vmin, vmax=vmax
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
                u = (np.roll(psi, -1, 0) - np.roll(psi, 1, 0)) / (2 * dy)
                v = -(np.roll(psi, -1, 1) - np.roll(psi, 1, 1)) / (2 * dx)
            
            # zonal mean velocity
            if "zonal" in stats:
                ubar = u.mean(axis=1)
                line_umean.set_xdata(ubar)
                umean_list.append(ubar)

            # energy
            if "energy" in stats:
                KE = 0.5 * np.mean(u**2 + v**2)
                energy_list.append(KE)
                time_list_energy.append(frame * frame_interval)

                line_energy.set_xdata(time_list_energy)
                line_energy.set_ydata(energy_list)
                ax_energy.relim()
                ax_energy.autoscale_view()

            # enstrophy
            if "enstrophy" in stats:
                enstro = 0.5 * np.mean(zeta**2)
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
        fig2.savefig("outputs/final_summary.png")
        plt.close(fig2)



    @staticmethod
    def compute_ke_spectrum(psi, grid):
        """
        Compute 1D isotropic KE spectrum E(k) using uh, vh
        """
        # Compute velocity in Fourier space
        psih = rfftn(psi, axes=(-2,-1))
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


