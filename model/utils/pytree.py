import dataclasses
import jax
import inspect
import itertools
import weakref

__all__ = \
    [
    ]


pytree_class_attrs_registry = weakref.WeakKeyDictionary()

def register_pytree_class_attrs(children, static_attrs):
    children = tuple(children)
    static_attrs = tuple(static_attrs)

    def do_registration(cls):
        pytree_class_attrs_registry[cls] = (children, static_attrs)
        # Combine recursively
        cls_children = set()
        cls_static = set()
        for c in inspect.getmro(cls):
            c_children, c_static = pytree_class_attrs_registry.get(c, ((), ()))
            cls_children.update(c_children)
            cls_static.update(c_static)
        if not cls_children.isdisjoint(cls_static):
            raise ValueError("recursive static and dynamic attributes overlap")
        cls_child_fields = tuple(cls_children)
        cls_static_fields = tuple(cls_static)

        def flatten_with_keys(obj):
            key_children = [
                (jax.tree_util.GetAttrKey(name), getattr(obj, name))
                for name in cls_child_fields
            ]
            if cls_static_fields:
                aux = tuple(getattr(obj, name) for name in cls_static_fields)
            else:
                aux = None
            return key_children, aux

        def flatten(obj):
            flatkeys, aux = flatten_with_keys(obj)
            return [c for _, c in flatkeys], aux

        def unflatten(aux_data, children):
            obj = cls.__new__(cls)
            for name, val in itertools.chain(
                zip(cls_child_fields, children, strict=True),
                zip(cls_static_fields, aux_data or (), strict=True),
            ):
                setattr(obj, name, val)
            return obj

        jax.tree_util.register_pytree_with_keys(
            cls, flatten_with_keys, unflatten, flatten
        )
        return cls

    return do_registration


def register_pytree_dataclass(cls):
    cls_fields = []
    cls_static_fields = []
    for field in dataclasses.fields(cls):
        if not field.init:
            continue
        if field.metadata.get("pyqg_jax", {}).get("static", False):
            cls_static_fields.append(field.name)
        else:
            cls_fields.append(field.name)

    jax.tree_util.register_dataclass(
        cls, data_fields=tuple(cls_fields), meta_fields=tuple(cls_static_fields)
    )
    return cls

def register_pytree_dataclass(cls):
    fields = tuple(f.name for f in dataclasses.fields(cls))

    def flatten(obj):
        return [getattr(obj, name) for name in fields], None

    def unflatten(aux_data, flat_contents):
        return cls(**dict(zip(fields, flat_contents, strict=True)))

    jax.tree_util.register_pytree_node(cls, flatten, unflatten)
    return cls