import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict

def summary(model, x, *args, **kwargs):
    """Summary the given model.
    Summarized information are output shape, kernel shape and 
    number of the parameters and operations (Mult-Adds)

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
            key = "{}_{}".format(module_idx, cls_name)

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

                if "weight" == name:
                    ksize = list(param.size())
                    # to make [in_shape, out_shape, ksize, ksize]
                    if len(ksize) > 1:
                        ksize[0], ksize[1] = ksize[1], ksize[0]
                    info["ksize"]  = ksize

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

    hooks = []
    summary = OrderedDict()

    model.apply(register_hook)
    model(x) if not (kwargs or args) else model(x, *args, **kwargs)

    for hook in hooks:
        hook.remove()

    print("-"*100)
    print("{:<15} {:>20} {:>20} {:>20} {:>20}"
        .format("Layer", "Kernel Shape", "Output Shape", 
                "# Params (K)", "# Mult-Adds (M)"))
    print("="*100)

    total_params, total_macs = 0, 0
    for layer, info in summary.items():
        repr_ksize  = str(info["ksize"])
        repr_out    = str(info["out"])
        repr_params = info["params"]
        repr_macs   = info["macs"]

        if isinstance(repr_params, (int, float)):
            total_params += repr_params
            repr_params   = "{0:,.2f}".format(repr_params/1000)
        if isinstance(repr_macs, (int, float)):
            total_macs += repr_macs
            repr_macs   = "{0:,.2f}".format(repr_macs/1000000)
            
        print("{:<15} {:>20} {:>20} {:>20} {:>20}"
            .format(layer, repr_ksize, repr_out, repr_params, repr_macs))

        # for RNN, describe inner weights (i.e. w_hh, w_ih)
        for inner_name, inner_shape in info["inner"].items():
            print("  {:<13} {:>20}".format(inner_name, str(inner_shape)))
    
    print("="*100)
    print("# Params:    {0:,.2f}K".format(total_params/1000))
    print("# Mult-Adds: {0:,.2f}M".format(total_macs/1000000))
    print("-"*100)
