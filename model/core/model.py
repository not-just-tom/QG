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
import jax.numpy as jnp
from jax.numpy.fft import rfftn, irfftn
import importlib
import model.core.grid
importlib.reload(model.core.grid)
from model.core.grid import Grid
import model.utils.pytree as Pytree
import model.core.kernel
importlib.reload(model.core.kernel)
import model.core.kernel as _kernel
import model.core.states as states


def _grid_xy(nx, ny, Lx, Ly):
    dx = Lx / nx
    dy = Ly / ny
    x = jnp.linspace(-Lx/2 + dx/2, Lx/2 - dx/2, nx)
    y = jnp.linspace(-Ly/2 + dy/2, Ly/2 - dy/2, ny)
    return jnp.meshgrid(x, y)


@Pytree.register_pytree_class_attrs(
    children=["Lx", "Ly"],
    static_attrs=[],
)
class Model(_kernel.Kernel):
    def __init__(
        self,
        params,
    ):
        super().__init__(
            params=params
        )
        self.Lx = params['Lx']
        self.Ly = params['Ly'] if hasattr(params, 'Ly') else params['Lx']
        self._grid = Grid(
            nx=self.nx,
            ny=self.ny,
            Lx=self.Lx,
            Ly=self.Ly,
        )

    def get_full_state(self, state: states.State) -> states.TempStates:
        """Compute all the temp values for a step in kernel.
        """
        full_state = super()._get_state(state)
        full_state = self._do_external_forcing(full_state)
        return full_state

    def get_grid(self) -> Grid:
        """Just a safer way to get info from the grid class
        """
        return self._grid

    def _do_external_forcing(self, state: states.TempStates) -> states.TempStates:
        # put machine leanring here?
        return state
    
    @property
    def _dealias(self, alpha=36.0, p=8):
        """Apply a precomputed dealias mask from the grid if available.
        """
       
        return jnp.exp(-alpha * (self.Kmag / jnp.max(self.Kmag)) ** p)

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


@Pytree.register_pytree_class_attrs(
    children=["beta"],
    static_attrs=[],
)
class QGM(Model):
    """Spectral solver for the 2D barotropic vorticity equation on a
    beta-plane"""

    def __init__(self, params):
        super().__init__(params)

    def intialise(self, key):
        state = super().initialise()
        return state


        









        