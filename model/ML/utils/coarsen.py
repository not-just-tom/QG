from abc import ABC, abstractmethod
import inspect
import jax.numpy as jnp
import jax

def model_to_args(model):
    return {
        arg: getattr(model, arg) for arg in inspect.signature(type(model)).parameters
    }

def coarsen(hr_model, n_lr):
    if hr_model.nx<n_lr:
        raise ValueError(f'Coarsening can only be reducing resolution! Attempted {hr_model.n} to {n_lr}')
    
    # Get params from the model (handles both Model and TwoLayerModel)
    model_args = model_to_args(hr_model)
    model_args["nx"] = n_lr
    model_args["ny"] = n_lr
    return type(hr_model)(**model_args)


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
        # Use create_initial_state which is JAX-compatible and doesn't rebuild the grid
        lr_state = self.lr_model.create_initial_state(jax.random.key(0))
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
