from abc import ABC, abstractmethod
import inspect
import jax.numpy as jnp
import jax

def model_to_args(model):
    # Inspect the class __init__ signature and pull attributes from the
    # instance for any constructor parameters (excluding 'self').
    sig = inspect.signature(type(model).__init__)
    params = [p for p in sig.parameters if p != "self"]
    return {p: getattr(model, p) for p in params if hasattr(model, p)}

def coarsen(hr_model, n_lr):
    # Unwrap if this is a SteppedModel
    if hasattr(hr_model, "model") and not hasattr(hr_model, "nx"):
        hr_physics = hr_model.model
    else:
        hr_physics = hr_model

    if hr_physics.nx < n_lr:
        raise ValueError(f'Coarsening can only be reducing resolution! Attempted {hr_physics.nx} to {n_lr}')
    
    # Get params from the model (handles both Model and QGM)
    cls = type(hr_physics)
    # Inspect the constructor to see if it expects a single `params` dict
    sig = inspect.signature(cls.__init__)
    ctor_params = [p for p in sig.parameters if p != "self"]

    # If constructor expects a single `params` dict, build that dict
    if len(ctor_params) == 1 and ctor_params[0] == "params":
        # Build a params dict from common model attributes
        param_dict = {}
        candidate_attrs = [
            "nx",
            "ny",
            "nz",
            "rek",
            "kmin",
            "kmax",
            "beta",
            "Lx",
            "Ly",
            "Lz",
            "filterfac",
            "g",
            "f",
            "rd",
            "delta",
            "U1",
            "U2",
            "hr_nx",
        ]
        for a in candidate_attrs:
            if hasattr(hr_physics, a):
                param_dict[a] = getattr(hr_physics, a)
        # Ensure square grid at the target resolution
        param_dict["nx"] = n_lr
        param_dict["ny"] = n_lr
        return cls(param_dict)

    # Otherwise fall back to old behaviour of passing individual args
    model_args = model_to_args(hr_physics)
    model_args["nx"] = n_lr
    model_args["ny"] = n_lr
    return cls(**model_args)


class Coarsener(ABC):
    def __init__(self, hr_model, n_lr):
        # Determine physics models
        self.hr_model = hr_model
        
        # Unwrap if this is a SteppedModel
        if hasattr(hr_model, "model") and not hasattr(hr_model, "nx"):
            self.hr_physics = hr_model.model
        else:
            self.hr_physics = hr_model

        self.n_lr = n_lr

        # Initialize LR model and template state ONCE
        self._lr_model = coarsen(self.hr_physics, self.n_lr)
        
        # Create a template state for the low res grid
        dummy_key = jax.random.PRNGKey(0)
        self._lr_template = self._lr_model.initialise(dummy_key)
        
        # Pre-calculate ratio
        self._ratio = self.hr_physics.nx / self.n_lr

    @property
    def lr_model(self):
        return self._lr_model
    
    @property
    def ratio(self):
        return self._ratio

    def coarsen_state(self, state):
        # nk = lr_state.qh.shape[-2] // 2
        nk = self._lr_template.qh.shape[-2] // 2

        # Galerkin Truncation
        trunc = jnp.concatenate(
            [
                state.qh[:, :nk, :nk + 1],
                state.qh[:, -nk:, :nk + 1],
            ],
            axis=-2,
        )
        filtered = trunc * self.spectral_filter / self.ratio**2
        return self._lr_template.update(qh = filtered)

    def sgs_forcing(self, state):
        """Just the difference between coarsening then 
        progressing or progressing then coarsening.
        """
        # Always use physics models for derivative calculations
        hr_updates = self.hr_physics.get_updates(state)
        coarsened_deriv = self.coarsen_state(hr_updates)
        
        lr_updates = self._lr_model.get_updates(self.coarsen_state(state))
        return coarsened_deriv.q - lr_updates.q

    @property
    @abstractmethod
    def spectral_filter(self):
        pass

    def tree_flatten(self):
        # We need to include anything that is a JAX array or another Pytree
        children = [self.hr_model, self._lr_model, self._lr_template]
        aux_data = (self.n_lr, self._ratio)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        n_lr, ratio = aux_data
        # Create a new instance without calling __init__
        obj = cls.__new__(cls)
        obj.hr_model = children[0]
        
        # Determine physics model again
        if hasattr(obj.hr_model, "model") and not hasattr(obj.hr_model, "nx"):
            obj.hr_physics = obj.hr_model.model
        else:
            obj.hr_physics = obj.hr_model

        obj._lr_model = children[1]
        obj._lr_template = children[2]
        obj.n_lr = n_lr
        obj._ratio = ratio
        return obj

@jax.tree_util.register_pytree_node_class
class Coarsen(Coarsener): # i hate this get rid at convenience. 
    @property
    def spectral_filter(self):
        return self.lr_model._dealias