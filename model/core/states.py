try:
    import h5py
except ModuleNotFoundError:
    h5py = None
try:
    import zarr
except ModuleNotFoundError:
    zarr = None
import jax
import jax.numpy as jnp
import dataclasses
import model.utils.pytree as Pytree


def _generic_rfftn(a):
    return jnp.fft.rfftn(a, axes=(-2, -1))

def _generic_irfftn(a, shape):
    return jnp.fft.irfftn(a, axes=(-2, -1), norm='ortho', s=None)


@Pytree.register_pytree_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class State:
    """Holds the evolving state of the QG model for JAX-functional stepping."""
    qh: jnp.ndarray
    _q_shape: tuple[int, int]

    @property
    def q(self) -> jnp.ndarray:
        return _generic_irfftn(self.qh, shape=self._q_shape)


    def update(self, **kwargs) -> "State":
        """Replace the value stored in this state to produces a new* state object, containing the
        replacement value.

        The keyword arguments may be either `q` or `qh` (not both),
        allowing the replacement value to be provided in spectral form
        if desired.
        """
        if not kwargs:
            # Copy the class with no changes
            return dataclasses.replace(self)
        if extra_attrs := (kwargs.keys() - {"q", "qh"}):
            extra_attr_str = ", ".join(extra_attrs)
            pl_suf = "s" if len(extra_attrs) > 1 else ""
            raise ValueError(
                f"tried to update unknown state attribute{pl_suf} {extra_attr_str}"
            )
        if len(kwargs) > 1:
            raise ValueError("duplicate updates for q (specified both q and qh)") 
        # Exactly one attribute specified
        attr, update = next(iter(kwargs.items()))
        new_qh = update if attr == "qh" else _generic_rfftn(update)
        return dataclasses.replace(self, qh=new_qh)

    def zero(self, *states):
        pass


@Pytree.register_pytree_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class TempStates:
    """Temporary states including calculated values, beyond the single truth 
    q/qh which are required for doing a physics step in kernel before being discarded
    """

    state: State
    ph: jnp.ndarray
    u: jnp.ndarray
    v: jnp.ndarray
    dqhdt: jnp.ndarray

    @property
    def qh(self) -> jnp.ndarray:
        return self.state.qh

    @property
    def q(self) -> jnp.ndarray:
        return self.state.q

    @property
    def p(self) -> jnp.ndarray:
        return _generic_irfftn(self.ph, shape=self.state._q_shape)

    @property
    def uh(self) -> jnp.ndarray:
        return _generic_rfftn(self.u)

    @property
    def vh(self) -> jnp.ndarray:
        return _generic_rfftn(self.v)

    @property
    def dqdt(self) -> jnp.ndarray:
        return _generic_irfftn(self.dqhdt, shape=self.state._q_shape)

    def update(self, **kwargs) -> "TempStates":
        """Replace values stored in this state.

        This function produces a *new* state object, with specified
        attributes replaced.

        The keyword arguments may specify any of this class's
        attributes *except* :attr:`state`, but must not apply multiple
        updates to the same attribute. That is, modifying both the
        spectral and real space values at the same time is not
        allowed.

        The object this method is called on is not modified.
        """

        new_values = {}
        if "state" in kwargs:
            raise ValueError(
                "do not update attribute 'state' directly, update individual fields "
                "(for example q or qh)"
            )
        if extra_attrs := (
            kwargs.keys()
            - {"q", "qh", "p", "ph", "u", "uh", "v", "vh", "dqhdt", "dqdt"}
        ):
            extra_attr_str = ", ".join(extra_attrs)
            pl_suf = "s" if len(extra_attrs) > 1 else ""
            raise ValueError(
                f"tried to update unknown state attribute{pl_suf} {extra_attr_str}"
            )
        for name, new_val in kwargs.items():
            match name:
                case "q" | "qh":
                    # Special handling for q and qh, make spectral and assign to state
                    new_val = self.state.update(**{name: new_val})
                    name = "state"
                case "uh" | "vh":
                    # Handle other spectral names, store as non-spectral
                    new_val = _generic_irfftn(new_val, shape=self.state._q_shape)
                    name = name[:-1]
                case "p":
                    new_val = _generic_rfftn(new_val)
                    name = "ph"
                case "dqdt":
                    new_val = _generic_rfftn(new_val)
                    name = "dqhdt"
            # Check that we don't have duplicate destinations
            if name in new_values:
                raise ValueError(f"duplicate updates for {name}")
            # Set up the actual replacement
            new_values[name] = new_val
        # Produce new object with processed values
        return dataclasses.replace(self, **new_values)