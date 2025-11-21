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
        qh = Solver.pseudo_randomiser(grid, k_beta, k1, energy=1.0)
        qh = Solver.dealias(qh, grid, s=8)

        # not used yet but for maybe normalisation
        psih = - qh * grid.invK2
        uh = -1j * grid.KY * psih
        vh =  1j * grid.KX * psih
        u = irfftn(uh, axes=(-2, -1))
        v = irfftn(vh, axes=(-2, -1)) 

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

        # Nonlinear advection 
        qe = q + params.get('eta', 0.0) 
        uq = u * qe 
        vq = v * qe 
        uqh, vqh = rfftn(jnp.stack([uq, vq], axis=0), axes=(-2, -1)) 
        nonlinear = -1j * (grid.KX * uqh + grid.KY * vqh) 
        nonlinear = Solver.dealias(nonlinear, grid, s=8)

        # Beta term
        beta_term = - params.get('beta', 1e-3) * 1j * grid.KX * psih

        # --- Forcing ---
        # Draw a real-space white-noise field and transform to spectral space so the
        # forcing respects Hermitian symmetry (real physical fields after iFFT).
        noise_phys = jax.random.normal(k_noise, (grid.ny, grid.nx))
        forcing_h = rfftn(noise_phys, axes=(-2, -1))

        if forcing_spectrum.shape != qh.shape:
            # common case: (ny, nx) -> (ny, nx//2+1)
            if forcing_spectrum.ndim == 2 and forcing_spectrum.shape[0] == grid.ny and forcing_spectrum.shape[1] == grid.nx:
                forcing_spectrum = forcing_spectrum[:, : qh.shape[1]]
            else:
                raise ValueError(f"forcing_spectrum shape {forcing_spectrum.shape} incompatible with qh shape {qh.shape}")
        forcing = forcing_h * forcing_spectrum

        # --- Dissipation ---
        nu = params.get("nu", 0.0)
        m  = params.get("m", 4)
        mu = params.get("mu", 0.0)

        hypervisc = nu * (grid.Kmag ** m) * qh # hyperviscosity (small scale)
        drag = mu * qh # linear drag (large scale)
        
        dissipation = drag + hypervisc

        rhs =  nonlinear + beta_term - dissipation #+ forcing

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
            jax.debug.print("DIAG RHS: KE={} max_qh={} max_forcing={} max_nonlinear={} max_beta={} max_diss={}",
                           ke_spec, max_qh, max_forcing, max_nonlinear, max_beta, max_diss)

        jax.lax.cond(cond, _print, lambda _: None, operand=None)

        assert u.shape == (grid.ny, grid.nx)
        assert qh.shape == (grid.ny, grid.nx//2 + 1), f"qh shape {qh.shape} unexpected"
        dealiased = Solver.dealias(qh, grid, s=8)
        assert dealiased.shape == qh.shape, f"dealias output shape {dealiased.shape} != qh shape {qh.shape}"

        return rhs, key
    
    def pseudo_randomiser(grid, k_peak, key, energy=1.0):
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

    def make_animation(self, nsteps=5000, frame_interval=100, outname="outputs/qg_gpu.gif"):
        frames = nsteps // frame_interval
        fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)

        # --- set colours from initial zeta field ---
        zeta = np.array(self.fields["zeta"])
        vmin, vmax = float(np.min(zeta)), float(np.max(zeta))

        im = ax.imshow(
            zeta,
            origin='lower',
            cmap='RdBu_r',
            extent=(-self.grid.Lx, self.grid.Lx, -self.grid.Ly, self.grid.Ly),
            vmin=vmin, vmax=vmax
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Relative Vorticity (ζ)")
        im.set_array(zeta) #just so the initial field is also in this bitch 

        # --- Frame update function ---
        def update(frame):
            if frame == 0:
                # frame 0: do NOT advance
                zeta = np.array(self.fields["zeta"])
            else:
                self.steps(frame_interval)
                zeta = np.array(self.fields["zeta"])
            im.set_array(zeta)
            ax.set_title(f"Vorticity (step {frame * frame_interval})")
            return [im]

        frames = frames + 1  # include initial field as frame 0

        anim = FuncAnimation(fig, update, frames=frames, blit=False)
        anim.save(outname, fps=10)
        plt.close(fig)

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



