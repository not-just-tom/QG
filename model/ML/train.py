import time
import logging
import jax.numpy as jnp
import jax
import functools
import equinox as eqx

def do_epoch(train_state, batch_iter, batch_fn):
    logger = logging.getLogger("train_epoch")
    epoch_start = time.perf_counter()
    losses = []
    for batch in batch_iter:
        train_state, batch_loss = batch_fn(batch, train_state)
        losses.append(batch_loss)
    epoch_end = time.perf_counter()
    mean_loss = jax.device_get(jnp.mean(jnp.stack(losses)))
    final_loss = jax.device_get(losses[-1])
    logger.info("Finished epoch in %f sec", epoch_end - epoch_start)
    logger.info("Epoch mean loss %f", mean_loss)
    logger.info("Epoch final loss %f", final_loss)
    return train_state, {"mean_loss": mean_loss.item(), "final_loss": final_loss.item(), "duration_sec": epoch_end - epoch_start}, rng_ctr

def make_batch_computer(params, loss_fn):

    def batch_loss(net, input_chunk, target_chunk):
        losses = jax.vmap(
            functools.partial(
                loss_fn,
                net=net,
            )
        )(input_chunk, target_chunk)
        return jnp.mean(losses)

    def do_batch(batch, state):
        batch_sizes = {leaf.shape[0] for leaf in jax.tree_util.tree_leaves(batch)}
        if len(batch_sizes) != 1:
            raise ValueError(f"Inconsistent batch sizes {batch_sizes}")

        # Special processing for the first chunk (gaussian noise, if needed) <- what?
        target_chunk = make_chunk_from_batch(
            channels=output_channels,
            batch=batch,
            params=params,
            net_aux=net_aux,
        )
        input_chunk = make_chunk_from_batch(
            channels=input_channels,
            batch=batch,
            params=params,
            net_aux=net_aux,
        )
        # Compute losses
        loss, grads = eqx.filter_value_and_grad(batch_loss)(state.net, input_chunk, target_chunk)# <--- is your net in state????
        # Update parameters
        out_state = state.apply_updates(grads)
        return out_state, loss

    return do_batch