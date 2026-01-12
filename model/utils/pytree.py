from abc import ABC, abstractmethod
import dataclasses
import jax

__all__ = \
    [
    ]


class PytreeNode(ABC):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        def flatten(obj):
            return obj.flatten()

        def unflatten(aux_data, children):
            return cls.unflatten(aux_data, children)

        jax.tree_util.register_pytree_node(cls, flatten, unflatten)

    @abstractmethod
    def flatten(self):
        """Return a JAX flattened representation.

        Returns
        -------

        Sequence[Sequence[object, ...], Sequence[object, ...]]
            The JAX flattened representation.
        """

        raise NotImplementedError

    @classmethod
    @abstractmethod
    def unflatten(cls, aux_data, children):
        """Unpack a JAX flattened representation.

        Returns
        -------

        object
            The unpacked object.
        """

        raise NotImplementedError

def register_pytree_dataclass(cls):
    fields = tuple(f.name for f in dataclasses.fields(cls))

    def flatten(obj):
        return [getattr(obj, name) for name in fields], None

    def unflatten(aux_data, flat_contents):
        return cls(**dict(zip(fields, flat_contents, strict=True)))

    jax.tree_util.register_pytree_node(cls, flatten, unflatten)
    return cls