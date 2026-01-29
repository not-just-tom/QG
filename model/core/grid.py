import jax
import jax.numpy as jnp
from functools import cached_property, partial
import model.utils.pytree as Pytree 

@Pytree.register_pytree_class_attrs(
    children=[],
    static_attrs=["Lx", "Ly", "nx", "ny", "nz"],
)
class Grid:
    """
    I'm working on it ok

    """
    def __init__(self, Lx, nx, Ly=None, ny=None, nz=1, Hi=None):
        """ Initialising the grid """
        self._Lx = Lx
        self._Ly = Ly if Ly is not None else Lx
        self._nx = nx
        self._ny = ny if ny is not None else nx
        self._nz = nz
        self._dx = self._Lx / self._nx
        self._dy = self._Ly / self._ny

        self._Hi = Hi

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
    def nz(self):
        return self._nz
    
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
    
    @property
    def nl(self) -> int:
        return self.ny

    @property
    def nk(self) -> int:
        return (self.nx // 2) + 1

    # vv check these shapes carefully vv
    @property
    def real_state_shape(self) -> tuple:
        if self.nz == 1:
            return (self.ny, self.nx)
        else:
            return (self.nz, self.ny, self.nx)

    @property
    def spectral_state_shape(self) -> tuple:
        if self.nz == 1:
            return (self.nl, self.nk)
        else:
            return (self.nz, self.nl, self.nk)

