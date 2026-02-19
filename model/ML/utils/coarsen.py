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
    if hr_model.nx<n_lr:
        raise ValueError(f'Coarsening can only be reducing resolution! Attempted {hr_model.n} to {n_lr}')
    
    # Get params from the model (handles both Model and QGM)
    cls = type(hr_model)
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
            if hasattr(hr_model, a):
                param_dict[a] = getattr(hr_model, a)
        # Ensure square grid at the target resolution
        param_dict["nx"] = n_lr
        param_dict["ny"] = n_lr
        return cls(param_dict)

    # Otherwise fall back to old behaviour of passing individual args
    model_args = model_to_args(hr_model)
    model_args["nx"] = n_lr
    model_args["ny"] = n_lr
    return cls(**model_args)


class Coarsener(ABC):
    def __init__(self, hr_model, n_lr):
        self.hr_model = hr_model
        self.n_lr = n_lr

    @property
    def lr_model(self):
        return coarsen(self.hr_model, self.n_lr)
    
    @property
    def ratio(self):
        return self.hr_model.nx / self.n_lr

    def coarsen_state(self, state):
        key = jax.random.PRNGKey(0)  # Seed is irrelevant here since we're just using the grid
        lr_state = self.lr_model.initialise(key)  
        nk = lr_state.qh.shape[-2] // 2

        # Galerkin Truncation - something is really up here
        trunc = jnp.concatenate(
            [
                state.qh[:, :nk, :nk + 1],
                state.qh[:, -nk:, :nk + 1],
            ],
            axis=-2,
        )
        filtered = trunc * self.spectral_filter / self.ratio**2
        return lr_state.update(qh = filtered)

    def sgs_forcing(self, state):
        """Just the difference between coarsening then 
        progressing or progressing then coarsening.
        """
        coarsened_deriv = self.coarsen_state(self.hr_model.get_updates(state))
        lr_deriv = self.lr_model.get_updates(self.coarsen_state(state))
        return coarsened_deriv.q - lr_deriv.q

    @property
    @abstractmethod
    def spectral_filter(self):
        pass

    def tree_flatten(self):
        return [self.hr_model], self.n_lr

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(hr_model=children[0], n_lr=aux_data)
