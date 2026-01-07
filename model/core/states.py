try:
    import h5py
except ModuleNotFoundError:
    h5py = None

try:
    import zarr
except ModuleNotFoundError:
    zarr = None

import jax.numpy as jnp
import numpy as np
from collections.abc import Mapping
from .grid import Grid

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
