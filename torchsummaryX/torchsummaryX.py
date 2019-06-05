from collections import OrderedDict
import numpy as np
import pandas as pd
import torch

def summary(model, x, *args, **kwargs):
    """Summarize the given input model.
    Summarized information are 1) output shape, 2) kernel shape,
    3) number of the parameters and 4) operations (Mult-Adds)
    Args:
        model (Module): Model to summarize
        x (Tensor): Input tensor of the model with [N, C, H, W] shape
                    dtype and device have to match to the model
        args, kwargs: Other argument used in `model.forward` function
    """
    def register_hook(module):
        def hook(module, inputs, outputs):
            cls_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            # Lookup name in a dict that includes parents
            for name, item in module_names.items():
                if item == module:
                    key = "{}_{}".format(module_idx, name)

            info = OrderedDict()
            info["id"] = id(module)
            if isinstance(outputs, (list, tuple)):
                info["out"] = list(outputs[0].size())
            else:
                info["out"] = list(outputs.size())

            info["ksize"] = "-"
            info["inner"] = OrderedDict()
            info["params"], info["macs"] = 0, 0
            for name, param in module.named_parameters():
                info["params"] += param.nelement()

                if name == "weight":
                    ksize = list(param.size())
                    # to make [in_shape, out_shape, ksize, ksize]
                    if len(ksize) > 1:
                        ksize[0], ksize[1] = ksize[1], ksize[0]
                    info["ksize"] = ksize

                    # ignore N, C when calculate Mult-Adds in ConvNd
                    if "Conv" in cls_name:
                        info["macs"] += int(param.nelement() * np.prod(info["out"][2:]))
                    else:
                        info["macs"] += param.nelement()

                # RNN modules have inner weights such as weight_ih_l0
                elif "weight" in name:
                    info["inner"][name] = list(param.size())
                    info["macs"] += param.nelement()

            # if the current module is already-used, mark as "(recursive)"
            # check if this module has params
            if list(module.named_parameters()):
                for v in summary.values():
                    if info["id"] == v["id"]:
                        info["params"] = "(recursive)"

            if info["params"] == 0:
                info["params"], info["macs"] = "-", "-"

            summary[key] = info

        # ignore Sequential and ModuleList
        if not module._modules:
            hooks.append(module.register_forward_hook(hook))

    module_names = get_names_dict(model)

    hooks = []
    summary = OrderedDict()

    model.apply(register_hook)

    with torch.no_grad():
        model(x) if not (kwargs or args) else model(x, *args, **kwargs)

    for hook in hooks:
        hook.remove()

    # Use pandas to align the columns
    df = pd.DataFrame(summary).T
    df["Mult-Adds (M)"] = pd.to_numeric(df["macs"], errors="coerce")/1e6
    df["Params (K)"] = pd.to_numeric(df["params"], errors="coerce")/1e3
    df = df.rename(columns=dict(
        ksize="Kernel Shape",
        out="Output Shape",
    ))
    df.index.name = "Layer"
    df = df[["Kernel Shape", "Output Shape", "Params (K)", "Mult-Adds (M)"]]
    df_sum = df.sum()

    max_repr_width = max([len(row) for row in df.to_string().split("\n")])

    print("="*max_repr_width)
    print(df.replace(np.nan, "-"))
    print("-"*max_repr_width)
    print("Params (K): ", df_sum["Params (K)"])
    print("Mult-Adds (M): ", df_sum["Mult-Adds (M)"])
    print("="*max_repr_width)

    return df

def get_names_dict(model):
    """Recursive walk to get names including path."""
    names = {}

    def _get_names(module, parent_name=""):
        for key, m in module.named_children():
            cls_name = str(m.__class__).split(".")[-1].split("'")[0]
            num_named_children = len(list(m.named_children()))
            if num_named_children > 0:
                name = parent_name + "." + key if parent_name else key
            else:
                name = parent_name + "." + cls_name + "_"+ key if parent_name else key
            names[name] = m

            if isinstance(m, torch.nn.Module):
                _get_names(m, parent_name=name)

    _get_names(model)
    return names
