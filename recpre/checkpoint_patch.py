"""Patch for Issue https://github.com/pytorch/pytorch/issues/133550 in Pytorch 2.5 and lower."""

# NB: this helper wraps fn before calling checkpoint_impl. kwargs and
#     saving/restoring of global state is handled here.

# mypy: allow-untyped-defs
import contextlib
from typing import Optional, Callable, Tuple, ContextManager


import torch
from torch.utils.checkpoint import (
    _get_debug_context_and_cb,
    _allowed_determinism_checks_to_fns,
    noop_context_fn,
    _infer_device_type,
    _is_compiling,
    _get_autocast_kwargs,
    _get_device_module,
    get_device_states,
    set_device_states,
    _enable_checkpoint_early_stop,
    _CheckpointFrame,
    _checkpoint_hook,
    _NoopSaveInputs,
)

from torch.utils._python_dispatch import TorchDispatchMode


_DEFAULT_DETERMINISM_MODE = "default"

_checkpoint_debug_enabled: Optional[bool] = None


def _checkpoint_without_reentrant_generator(
    fn,
    preserve_rng_state=True,
    context_fn: Callable[[], Tuple[ContextManager, ContextManager]] = noop_context_fn,
    determinism_check: str = _DEFAULT_DETERMINISM_MODE,
    debug: bool = False,
    *args,
    **kwargs,
):
    """Checkpointing without reentrant autograd.

    Args:
        function: describes what to run in the forward pass of the model or
            part of the model. It should also know how to handle the inputs
            passed as the tuple. For example, in LSTM, if user passes
            ``(activation, hidden)``, :attr:`function` should correctly use the
            first input as ``activation`` and the second input as ``hidden``
        preserve_rng_state(bool, optional):  Omit stashing and restoring
            the RNG state during each checkpoint.
            Default: ``True``
        context_fn(Callable, optional): A callable returning a tuple of two
            context managers. The function and its recomputation will be run
            under the first and second context managers respectively.
        determinism_check(str, optional): A string specifying the determinism
            check to perform. By default it is set to ``"default"`` which
            compares the shapes, dtypes, and devices of the recomputed tensors
            against those the saved tensors. To turn off this check, specify
            ``"none"``. Currently these are the only two supported values.
            Please open an issue if you would like to see more determinism
            checks.
        debug(bool, optional): If ``True``, error messages will also include
            a trace of the operators ran during the original forward computation
            as well as the recomputation.
        *args: Arguments to pass in to the given ``function``.
        **kwargs: Keyword arguments to pass into the given ``function``.
    """
    unpack_error_cb = None

    if _checkpoint_debug_enabled if _checkpoint_debug_enabled is not None else debug:
        if context_fn != noop_context_fn:
            raise ValueError("debug=True is incompatible with non-default context_fn")
        context_fn, unpack_error_cb = _get_debug_context_and_cb()

    if determinism_check in _allowed_determinism_checks_to_fns:
        metadata_fn = _allowed_determinism_checks_to_fns[determinism_check]
    else:
        raise ValueError(
            f"determinism_check should be one of {list(_allowed_determinism_checks_to_fns.keys())}, "
            f"but got {determinism_check}"
        )

    device_type = _infer_device_type(*args)

    forward_context, recompute_context = context_fn()
    if _is_compiling(fn, args, kwargs) and context_fn != noop_context_fn:
        assert isinstance(forward_context, TorchDispatchMode) and isinstance(recompute_context, TorchDispatchMode), (
            "In torch.compile mode, `context_fn` arg passed to `torch.utils.checkpoint` "
            + "must generate a tuple of two `TorchDispatchMode`s."
        )
    # Accommodates the (remote) possibility that autocast is enabled for cpu AND gpu.
    device_autocast_kwargs, cpu_autocast_kwargs = _get_autocast_kwargs(device_type=device_type)

    if preserve_rng_state:
        device_module = _get_device_module(device_type)
        fwd_cpu_state = torch.get_rng_state()
        # Don't eagerly initialize the cuda context by accident.
        # (If the user intends that the context is initialized later, within their
        # run_function, we SHOULD actually stash the cuda state here.  Unfortunately,
        # we have no way to anticipate this will happen before we run the function.
        # If they do so, we raise an error.)
        had_device_in_fwd = False
        if getattr(device_module, "_initialized", False):
            had_device_in_fwd = True
            fwd_devices, fwd_device_states = get_device_states(*args)

    def recompute_fn(*inputs):
        kwargs, *args = inputs
        # This will be called later during recomputation. This wrapping enables
        # the necessary global state to be captured.
        rng_devices = []
        if preserve_rng_state and had_device_in_fwd:
            rng_devices = fwd_devices
        with (
            torch.random.fork_rng(devices=rng_devices, enabled=preserve_rng_state, device_type=device_type)
            if preserve_rng_state
            else contextlib.nullcontext()
        ):
            if preserve_rng_state:
                torch.set_rng_state(fwd_cpu_state)
                if had_device_in_fwd:
                    set_device_states(fwd_devices, fwd_device_states, device_type=device_type)

            device_autocast_ctx = (
                torch.amp.autocast(device_type=device_type, **device_autocast_kwargs)  # type: ignore
                if torch.amp.is_autocast_available(device_type)
                else contextlib.nullcontext()
            )
            with device_autocast_ctx, torch.amp.autocast("cpu", **cpu_autocast_kwargs), recompute_context:  # type: ignore[attr-defined]
                fn(*args, **kwargs)

    new_frame = _CheckpointFrame(recompute_fn, _enable_checkpoint_early_stop, unpack_error_cb, metadata_fn)
    dummy = torch.empty((0,), requires_grad=True)
    new_frame.input_saver = _NoopSaveInputs.apply(dummy, kwargs, *args)

    # When ambient grad_mode is False
    if new_frame.input_saver.grad_fn is None:  # type: ignore[attr-defined]
        yield
        return

    with _checkpoint_hook(new_frame), forward_context:
        yield
    new_frame.forward_completed = True

    if preserve_rng_state and getattr(device_module, "_initialized", False) and not had_device_in_fwd:  # type: ignore[possibly-undefined]
        # Device was not initialized before running the forward, so we didn't
        # stash the device state.
        raise RuntimeError(
            "PyTorch's device state was initialized in the forward pass "
            "of a Checkpoint, which is not allowed. Please open an issue "
            "if you need this feature."
        )

    return
