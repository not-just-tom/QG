import model.utils.pytree as Pytree 

@Pytree.register_pytree_class_attrs(
    children=[],
    static_attrs=["Lx", "Ly", "nx", "ny", "nz"],
)
class Grid:
    """
    I can't believe im back here after all this time - geometry only. No work with fft. Bare minimum

    """
    def __init__(self, Lx, nx, Ly=None, ny=None, nz=1, Lz=None):
        """ Initialising the grid """
        self._Lx = Lx
        self._Ly = Ly if Ly is not None else Lx
        self._Lz = Lz
        self._nx = nx
        self._ny = ny if ny is not None else nx
        self._nz = nz
        self._dx = self._Lx / self._nx
        self._dy = self._Ly / self._ny
    
    @property
    def Lx(self):
        return self._Lx
    
    @property
    def Ly(self):
        return self._Ly
    
    @property
    def Lz(self):
        return self._Lz
    
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
    def nl(self) -> int:
        return self._ny

    @property
    def nk(self) -> int:
        return (self._nx // 2) + 1

    # vv check these shapes carefully vv
    @property
    def real_state_shape(self) -> tuple:
        # Always include the layer axis for consistency across kernels.
        return (self._nz, self._ny, self._nx)

    @property
    def spectral_state_shape(self) -> tuple:
        # Always include the layer axis for consistency across kernels.
        return (self._nz, self.nl, self.nk)

