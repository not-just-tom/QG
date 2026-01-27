from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp
import copy

def coarsen(hr_model, n_lr):
    if hr_model.nx<n_lr:
        raise ValueError(f'Coarsening can only be reducing resolution! Attempted {hr_model.n} to {n_lr}')
    
    params = copy.deepcopy(hr_model.params)
    params['nx'] = n_lr
    params['ny'] = n_lr
    return type(hr_model)(params)


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
        lr_state = self.lr_model.initialise(42) # this doesn't need to be tracked but should I keep the seed the same as the config?
        nk = lr_state.qh.shape[0] // 2

        # Galerkin Truncation - something is really up here
        trunc = jnp.concatenate(
            [
                state.qh[:nk, : nk + 1],    # positive ky
                state.qh[-nk:, : nk + 1],   # negative ky
            ],
            axis=0,
        )
        filtered = trunc * self.spectral_filter[None, :] / self.ratio**2
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
