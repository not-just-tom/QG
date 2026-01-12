import jax
import jax.numpy as jnp
from functools import cached_property, partial
from jax import tree_util
from ..utils.pytree import PytreeNode



class Grid(PytreeNode):

    """
    I'm working on it ok 

    """
    def __init__(self, cfg):
        """ Initialising the grid """
        self._Lx = cfg.params.Lx
        self._Ly = cfg.params.Ly if hasattr(cfg.params, 'Ly') else cfg.params.Lx
        self._nx = cfg.params.nx
        self._ny = cfg.params.ny if hasattr(cfg.params, 'ny') else cfg.params.nx
        self._dx = self._Lx / self._nx
        self._dy = self._Ly / self._ny

        self._kx = jnp.fft.rfftfreq(self._nx, d=self._dx) * 2*jnp.pi 
        self._ky = jnp.fft.fftfreq(self._ny, d=self._dy) * 2*jnp.pi

        # rfft-shaped meshgrid: shape (ny, nx//2+1)
        self._KX, self._KY = jnp.meshgrid(self._kx, self._ky, indexing='xy')
        self._K2 = self._KX**2 + self._KY**2
        self._invK2 = jnp.where(self._K2 == 0.0, 0.0, 1.0 / self._K2)
        self._Kmag = jnp.sqrt(self._K2)


    @property
    def Lx(self):
        return self._Lx
    
    @property
    def Ly(self):
        return self._Ly
    
    @property
    def nx(self):
        return self._nx
    
    @property
    def ny(self):
        return self._ny
    
    @property
    def dx(self):
        return self._dx
    
    @property
    def dy(self):
        return self._dy
    
    @property
    def kx(self):
        return self._kx
    
    @property
    def ky(self):
        return self._ky

    @property
    def KX(self):
        return self._KX
    
    @property
    def KY(self):
        return self._KY
    
    @property
    def K2(self):
        return self._K2
    
    @property
    def invK2(self):
        return self._invK2
    
    @property
    def Kmag(self):
        return self._Kmag
    
    @cached_property
    def x(self):
        return jnp.linspace(-self.Lx/2, self.Lx/2 - self.dx, self.nx)
    
    @cached_property
    def y(self):
        return jnp.linspace(-self.Ly/2, self.Ly/2 - self.dy, self.ny)
    
    @cached_property
    def X(self):
        # physical grid (ny, nx)
        return jnp.meshgrid(self.x, self.y, indexing='xy')[0]  
    
    @cached_property
    def Y(self):
        return jnp.meshgrid(self.x, self.y, indexing='xy')[1]
    
    @staticmethod
    @partial(jax.jit, static_argnums=(0,))
    def _interpolate(x_g, y_g, u, x, y):
        nx, = x_g.shape
        nx -= 1
        ny, = y_g.shape
        ny -= 1
        Lx = x_g[-1]
        Ly = y_g[-1]

        i = jnp.array((x + Lx) * nx / (2 * Lx))
        i = jnp.maximum(i)
        i = jnp.minimum(i, (nx - 1))
        alpha = (x - x_g[i]) / (x_g[i + 1] - x_g[i])

        j = jnp.array((y + Ly) * ny / (2 * Ly))
        j = jnp.maximum(j)
        j = jnp.minimum(j, (ny - 1))
        beta = (y - y_g[j]) / (y_g[j + 1] - y_g[j])

        return (jnp.outer(1 - alpha, 1 - beta) * u[i, :][:, j]
                + jnp.outer(alpha, 1 - beta) * u[i + 1, :][:, j]
                + jnp.outer(1 - alpha, beta) * u[i, :][:, j + 1]
                + jnp.outer(alpha, beta) * u[i + 1, :][:, j + 1])

    def interpolate(self, u, x, y, *, extrapolate=False):
        """Bilinearly interpolate onto a grid.

        Parameters
        ----------

        u : :class:`jax.Array`
            Array of grid point values.
        x : :class:`jax.Array`
            :math:`x`-coordinates.
        y : :class:`jax.Array`
            :math:`y`-coordinates.
        extrapolate : bool
            Whether to allow extrapolation.

        Returns
        -------

        :class:`jax.Array`
            Array of values on the grid.
        """

        if not extrapolate and ((x < -self.Lx).any() or (x > self.Lx).any()
                                or (y < -self.Ly).any() or (y > self.Ly).any()):
            raise ValueError("Out of bounds")

        return self._interpolate(self.x, self.y, u, x, y)

    @cached_property
    def W(self) -> jax.Array:
        """Integration matrix diagonal.
        """

        w_x = jnp.ones(self.nx + 1) * self.dx
        w_x = w_x.at[0].set(0.5 * self.dx)
        w_x = w_x.at[-1].set(0.5 * self.dx)

        w_y = jnp.ones(self.ny + 1) * self.dy
        w_y = w_y.at[0].set(0.5 * self.dy)
        w_y = w_y.at[-1].set(0.5 * self.dy)

        return jnp.outer(w_x, w_y)

    @cached_property
    def dealias_mask(self) -> jax.Array:
        """Precompute a component-wise 2/3-rule dealias mask in rfft layout.

        Returns an array shaped (ny, nx//2+1) with 1.0 in kept modes and 0.0
        in removed modes. Using a cached property avoids recomputing masks per
        step and is safe for JAX tracing.
        """
        # Maximum k in each direction
        kxmax = jnp.max(jnp.abs(self.kx))
        kymax = jnp.max(jnp.abs(self.ky))
        kxcut = (2.0 / 3.0) * kxmax
        kycut = (2.0 / 3.0) * kymax

        mask_x = jnp.where(jnp.abs(self.kx) <= kxcut, 1.0, 0.0)
        mask_y = jnp.where(jnp.abs(self.ky) <= kycut, 1.0, 0.0)
        mask = mask_y[:, None] * mask_x[None, :]
        return mask

    def integrate(self, u):
        """
        Compute the integral of a field.

        Parameters
        ----------

        u : :class:`jax.Array`
            Field to integrate.

        Returns
        -------

        :class:`jax.Array`
            The integral.
        """

        return jnp.tensordot(u, self.W)

    def flatten(self):
        # no children (static object) <--- check what this means
        children = ()
        aux_data = (self.Lx, self.Ly, self.nx, self.ny)
        return children, aux_data

    @classmethod
    def unflatten(cls, aux_data, children):
        Lx, Ly, nx, ny = aux_data
        assert len(children) == 0
        params = {"Lx": Lx, "Ly": Ly, "nx": nx, "ny": ny}
        return cls(params)

