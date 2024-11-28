from functools import partial
from typing import Optional, List, Any

import pandas as pd
import torch
import torch.nn as nn


def num_module_parameters(module: nn.Module, trainable: Optional[bool] = None) -> int:
    count = 0
    for p in module.parameters():
        if trainable is None or trainable == p.requires_grad:
            count += p.numel()
    return count


def clip_module_weights(module: nn.Module, max_magnitude: float):
    with torch.no_grad():
        for param in module.parameters():
            param[:] = param.clamp(-max_magnitude, max_magnitude)
            #print(param.max(), param.shape)


def dump_module_stacktrace(model: nn.Module, *input) -> Any:
    stack = []
    hooks = []

    def _hook(model, args, kwargs=None, name: str=""):
        stack.append({
            "module": name,
            "input": ", ".join([
                str(arg.shape) if isinstance(arg, torch.Tensor) else type(arg).__name__
                for arg in args
            ]),
            "params": model.extra_repr(),

        })

    def _register_hooks(model, path: List[str], idx: int):
        hooks.append(
            model.register_forward_pre_hook(partial(_hook, name=".".join(path)))
        )
        for name, child in model.named_children():
            _register_hooks(child, path + [name], idx + 1)

    _register_hooks(model, [type(model).__name__], 0)

    with torch.no_grad():
        result = model(*input)

    for hook in hooks:
        hook.remove()

    if stack:
        df = pd.DataFrame(stack)
        print(df.to_markdown(index=False))

    return result
