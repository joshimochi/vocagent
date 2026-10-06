"""Activation boundaries for H3's branches outside accelerator module hooks."""


def align_h3_residual_outputs(model, device):
    """Return H3's scattered/gathered branch outputs to its input GPU.

    Accelerate aligns module inputs, but H3 also runs index_copy/index_select in
    its top-level forward, outside those hooks. The corresponding index tensors
    stay on the execution GPU. Returning these five branch outputs there keeps
    those operations local while the much larger transformer blocks stay sharded.
    Only activations move; parameters and their placement hooks are untouched.
    """
    names = ("proj_in", "audio_proj_in", "token_refiner", "proj_out", "audio_proj_out")
    modules = [getattr(model, name) for name in names]

    def return_to_execution_device(module, inputs, output):
        return output.to(device)

    # Native forward hooks run after Accelerate's wrapped forward. Replacing
    # module.forward here would bypass its device-alignment hooks.
    return tuple(module.register_forward_hook(return_to_execution_device) for module in modules)
