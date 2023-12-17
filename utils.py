import numpy as np
import torch

def sm_array(shared_mem, shape, dtype=np.float32):
    return np.ndarray(shape, dtype=dtype, buffer=shared_mem.buf)


def aux_function(func):
    def inner_aux(*args):
        out = func(*args)
        return out, out

    return inner_aux


def batched_jacobian(f, in_dims, *args):
    with torch.no_grad():
        return torch.func.vmap(torch.func.jacrev(aux_function(f), argnums=1, has_aux=True), in_dims=in_dims)(*args)


