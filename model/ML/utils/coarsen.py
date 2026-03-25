import inspect

def model_to_args(model):
    # Inspect the class __init__ signature and pull attributes from the
    # instance for any constructor parameters (excluding 'self').
    sig = inspect.signature(type(model).__init__)
    params = [p for p in sig.parameters if p != "self"]
    return {p: getattr(model, p) for p in params if hasattr(model, p)}

def coarsen(hr_model, n_lr):
    '''
    hr_model here is QGM. 
    '''
    cls = type(hr_model)
    if not hasattr(hr_model, 'nx'):
        raise TypeError(f'hr_model input should be a QGM object, instead it is {cls}')

    if hr_model.nx < n_lr:
        raise ValueError(f'Coarsening can only be reducing resolution! Attempted {hr_model.nx} to {n_lr}')
    
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