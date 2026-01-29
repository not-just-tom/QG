"""
Model basis and defintion stuff - make this sound smarter eventually eh

To do:
Check that this is GPU compatible 
On that note, JAX optimise
Figure out the grid formulation properly 
Check Arakawa Jacobian discretisation
Add initialise function which can clear field variables <---
^^^ these are kept IN the function, so thats why we need a thing for them....


"""
from abc import abstractmethod
import jax.numpy as jnp
from jax.numpy.fft import rfftn, irfftn
import importlib
import model.core.TwoLayer
importlib.reload(model.core.TwoLayer)
import model.core.grid
importlib.reload(model.core.grid)
from model.core.grid import Grid
import model.utils.pytree as Pytree
import model.core.kernels
importlib.reload(model.core.kernels)
from model.core.kernels import SingleLayerKernel, TwoLayerKernel
import model.core.states as states


def create_model(params, n_layers=1):
    """Factory function to create QG models based on configuration.
    
    Parameters
    ----------
    params : dict
        Configuration dictionary with model parameters
    n_layers : int, optional
        Number of layers: 1 for single-layer (default), 2 for two-layer model
        
    Returns
    -------
    Model or TwoLayerModel
        Appropriate model instance
    """
    if n_layers == 1:
        return SingleLayerModel(params)
    elif n_layers == 2:
        # Import here to avoid circular imports
        from model.core.TwoLayer import TwoLayerModel
        return TwoLayerModel.from_params(params)
    else:
        raise ValueError(f"Unsupported n_layers={n_layers}. Use 1 or 2.")


def _grid_xy(nx, ny, Lx, Ly):
    dx = Lx / nx
    dy = Ly / ny
    x = jnp.linspace(-Lx/2 + dx/2, Lx/2 - dx/2, nx)
    y = jnp.linspace(-Ly/2 + dy/2, Ly/2 - dy/2, ny)
    return jnp.meshgrid(x, y)

def _grid_kl(kk, ll):
    k, l = jnp.meshgrid(kk, ll)
    return k, l

@Pytree.register_pytree_class_attrs(
    children=[],
    static_attrs=["nx", "ny", "Lx", "Ly", "beta", "kmin", "kmax"],
)
class SingleLayerModel(SingleLayerKernel):
    """Base single-layer QG model with spectral solver.
    
    Requires params dict with keys:
    - nx, ny: grid resolution
    - Lx, Ly: domain size
    - beta: beta-plane parameter
    - kmin, kmax: wavenumber band for initialization
    """
    
    def __init__(self, params):
        """Initialize model from params dict.
        
        Parameters
        ----------
        params : dict
            Configuration dictionary with model parameters
        """
        super().__init__(params=params)

    def get_full_state(self, state: states.State) -> states.TempStates:
        """Compute all the temp values for a step in kernel.
        """
        full_state = super()._get_state(state)
        full_state = self._do_external_forcing(full_state)
        return full_state

    def get_grid(self) -> Grid:
        """Retrieve the grid for this model."""
        return Grid(
            nx=self.nx,
            ny=self.ny,
            Lx=self.Lx,
            Ly=self.Ly,
        )

    def _do_external_forcing(self, state: states.TempStates) -> states.TempStates:
        # put machine leanring here?
        return state
    
    def _get_dealias_filter(self, alpha=36.0, p=8):
        """Apply a precomputed dealias mask from the grid if available.
        """
        return jnp.exp(-alpha * (self.Kmag / jnp.max(self.Kmag)) ** p)
    
    @property
    def _dealias(self):
        """Dealias filter as a property."""
        return self._get_dealias_filter()

    @property
    def dx(self):
        return self.get_grid().dx

    @property
    def dy(self):
        return self.get_grid().dy
    
    @property
    def kk(self):
        return jnp.fft.rfftfreq(self.nx, d=self.dx / (2 * jnp.pi))

    @property
    def ll(self):
        return jnp.fft.fftfreq(self.ny, d=self.dy / (2 * jnp.pi))

    @property
    def Kmag(self):
        KX, KY = jnp.meshgrid(self.kk, self.ll)
        return jnp.sqrt(KX**2 + KY**2)


    @property
    def M(self):
        return self.nx * self.ny

    @property
    def x(self):
        return _grid_xy(
            nx=self.nx,
            ny=self.ny,
            Lx=self.Lx,
            Ly=self.Ly,
        )[0]

    @property
    def y(self):
        return _grid_xy(
            nx=self.nx,
            ny=self.ny,
            Lx=self.Lx,
            Ly=self.Ly,
        )[1]
        
    @property
    def ll(self):
        return jnp.fft.fftfreq(
            self.ny,
            d=(self.Ly / (2 * jnp.pi * self.ny)),
        )

    @property
    def kk(self):
        return jnp.fft.rfftfreq(
            self.nx,
            d=(self.Lx / (2 * jnp.pi * self.nx)),
        )