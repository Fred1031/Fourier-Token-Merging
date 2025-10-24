import torch
import math
from typing import Type, Dict, Any, Tuple, Callable
import copy

from . import merge
from .utils import isinstance_str, init_generator



def compute_merge(x: torch.Tensor, tome_info: Dict[str, Any]) -> Tuple[Callable, ...]:
    
    original_h, original_w = 64, 64
    original_tokens = original_h * original_w
    downsample = int(math.ceil(math.sqrt(original_tokens // x.shape[1])))

    args = tome_info["args"]
    use_dft     = args.get("use_dft", False)
    trunc_ratio = args.get("trunc_ratio", 1.0)
    high_weight = args.get("high_weight", 1.0)
    low_weight  = args.get("low_weight", 0.0)

    layer_idx = tome_info.get("layer_idx", 0)
    timestep = tome_info.get("timestep", None)
    # If missing, try shared view
    if timestep is None:
        timestep = tome_info.get("shared", {}).get("timestep", None)
    #print(timestep)

    

    adaptive = args.get("adaptive_trunc", True)
    basis = args.get("adaptive_basis", "timestep")  # "timestep" or "layer"
    direction = args.get("adaptive_direction", "increasing")
    timestep_bins = args.get("timestep_bins", 5)
    trunc_ratio_min = args.get("trunc_ratio_min", 0.3)
    trunc_ratio_max = args.get("trunc_ratio_max", 0.9)


    if adaptive:
        if basis == "timestep":
            t_val = timestep.item() if torch.is_tensor(timestep) else float(timestep)
            max_timestep = args.get("max_timestep", 999)

            

            # Step binning
            bin_idx = min(int((float(timestep) / max_timestep) * timestep_bins), timestep_bins - 1)
            t_norm = bin_idx / max(timestep_bins - 1, 1)

            
        elif basis == "layer":
            layer_idx = tome_info.get("layer_idx", 0)
            total_layers = args.get("total_layers", 16)
            t_norm = min(layer_idx / max(total_layers - 1, 1), 1.0)
        else:
            raise ValueError(f"Unknown adaptive_basis: {basis}")

        if direction == "increasing":
            trunc_ratio = trunc_ratio_min + (trunc_ratio_max - trunc_ratio_min) * t_norm
        elif direction == "decreasing":
            trunc_ratio = trunc_ratio_max - (trunc_ratio_max - trunc_ratio_min) * t_norm
        else:
            raise ValueError(f"Unknown adaptive_direction: {direction}")
    else:
        trunc_ratio = args.get("trunc_ratio", 1.0)


    


    if downsample <= args["max_downsample"]:
        w = int(math.ceil(original_w / downsample))
        h = int(math.ceil(original_h / downsample))
        r = int(x.shape[1] * args["ratio"])

        # Re-init the generator if it hasn't already been initialized or device has changed.
        if args["generator"] is None:
            args["generator"] = init_generator(x.device)
        elif args["generator"].device != x.device:
            args["generator"] = init_generator(x.device, fallback=args["generator"])
        
        # If the batch size is odd, then it's not possible for prompted and unprompted images to be in the same
        # batch, which causes artifacts with use_rand, so force it to be off.
        use_rand = False if x.shape[0] % 2 == 1 else args["use_rand"]
        m, u = merge.bipartite_soft_matching_random2d(x, w, h, args["sx"], args["sy"], r, ratio=args["ratio"], use_dft=use_dft, trunc_ratio=trunc_ratio, high_weight=high_weight, low_weight=low_weight,
                                                      no_rand=not use_rand, generator=args["generator"])
    else:
        m, u = (merge.do_nothing, merge.do_nothing)

    m_a, u_a = (m, u) if args["merge_attn"]      else (merge.do_nothing, merge.do_nothing)
    m_c, u_c = (m, u) if args["merge_crossattn"] else (merge.do_nothing, merge.do_nothing)
    m_m, u_m = (m, u) if args["merge_mlp"]       else (merge.do_nothing, merge.do_nothing)

    #print('merged ', m_a.shape)

    return m_a, m_c, m_m, u_a, u_c, u_m  # Okay this is probably not very good







def make_tome_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    """
    Make a patched class on the fly so we don't have to import any specific modules.
    This patch applies ToMe to the forward function of the block.
    """

    class ToMeBlock(block_class):
        # Save for unpatching later
        _parent = block_class

        def _forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
            m_a, m_c, m_m, u_a, u_c, u_m = compute_merge(x, self._tome_info)

            # This is where the meat of the computation happens
            x = u_a(self.attn1(m_a(self.norm1(x)), context=context if self.disable_self_attn else None)) + x
            x = u_c(self.attn2(m_c(self.norm2(x)), context=context)) + x
            x = u_m(self.ff(m_m(self.norm3(x)))) + x

            return x
    
    return ToMeBlock


def patch_unet_forward(unet):
    """Monkey-patch unet.forward to inject timestep into _tome_info."""
    original_forward = unet.forward

    def wrapped_forward(sample, timestep, encoder_hidden_states, *args, **kwargs):
        if hasattr(unet, "_tome_info"):
            unet._tome_info["timestep"] = timestep
        return original_forward(sample, timestep, encoder_hidden_states, *args, **kwargs)

    unet.forward = wrapped_forward




def make_diffusers_tome_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    """
    Make a patched class for a diffusers model.
    This patch applies ToMe to the forward function of the block.
    """
    class ToMeBlock(block_class):
        # Save for unpatching later
        _parent = block_class

        def forward(
            self,
            hidden_states,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            timestep=None,
            cross_attention_kwargs=None,
            class_labels=None,
        ) -> torch.Tensor:

            #print("timestep", timestep)
            if timestep is not None and self.use_ada_layer_norm_zero:
                # If using AdaLayerNormZero, we need to pass the timestep and class labels
                # to the layer norm for the first attention block.
                if self.use_ada_layer_norm:
                    raise ValueError("Cannot use both use_ada_layer_norm and use_ada_layer_norm_zero at the same time.")
                
                self._tome_info["timestep"] = timestep

            # (1) ToMe
            m_a, m_c, m_m, u_a, u_c, u_m = compute_merge(hidden_states, self._tome_info)

            if self.use_ada_layer_norm:
                norm_hidden_states = self.norm1(hidden_states, timestep)
            elif self.use_ada_layer_norm_zero:
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                    hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
                )
            else:
                norm_hidden_states = self.norm1(hidden_states)

            # (2) ToMe m_a
            norm_hidden_states = m_a(norm_hidden_states)

            # 1. Self-Attention
            cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
            attn_output = self.attn1(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )
            if self.use_ada_layer_norm_zero:
                attn_output = gate_msa.unsqueeze(1) * attn_output

            # (3) ToMe u_a
            hidden_states = u_a(attn_output) + hidden_states

            if self.attn2 is not None:
                norm_hidden_states = (
                    self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
                )
                # (4) ToMe m_c
                norm_hidden_states = m_c(norm_hidden_states)

                # 2. Cross-Attention
                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    **cross_attention_kwargs,
                )
                # (5) ToMe u_c
                hidden_states = u_c(attn_output) + hidden_states

            # 3. Feed-forward
            norm_hidden_states = self.norm3(hidden_states)
            
            if self.use_ada_layer_norm_zero:
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

            # (6) ToMe m_m
            norm_hidden_states = m_m(norm_hidden_states)

            ff_output = self.ff(norm_hidden_states)

            if self.use_ada_layer_norm_zero:
                ff_output = gate_mlp.unsqueeze(1) * ff_output

            # (7) ToMe u_m
            hidden_states = u_m(ff_output) + hidden_states

            return hidden_states

    return ToMeBlock






def hook_tome_model(model: torch.nn.Module):
    
    """ Adds a forward pre hook to get the image size. This hook can be removed with remove_patch. """
    def hook(module, args):
        module._tome_info["size"] = (args[0].shape[2], args[0].shape[3])
        #module._tome_info["layer_idx"] = 
        return None

    model._tome_info["hooks"].append(model.register_forward_pre_hook(hook))








def apply_patch(
        model: torch.nn.Module,
        ratio: float = 0.5,
        max_downsample: int = 1,
        sx: int = 2, sy: int = 2,
        use_rand: bool = True,
        merge_attn: bool = True,
        merge_crossattn: bool = False,
        merge_mlp: bool = False,
        use_dft: bool = False,
        trunc_ratio: float = 1.0,
        high_weight: float = 1.0,
        low_weight: float = 0.0,
        trunc_ratio_min: float = 0.5,
        trunc_ratio_max: float = 0.9,
        max_timestep: int = 999,
        total_layers: int = 16,
        adaptive_trunc: bool = True,
        adaptive_basis: str = "timestep",  # or "layer"
        adaptive_direction: str = "increasing",  # or "decreasing"
    ):

    """
    Patches a stable diffusion model with ToMe.
    Apply this to the highest level stable diffusion object (i.e., it should have a .model.diffusion_model).

    Important Args:
     - model: A top level Stable Diffusion module to patch in place. Should have a ".model.diffusion_model"
     - ratio: The ratio of tokens to merge. I.e., 0.4 would reduce the total number of tokens by 40%.
              The maximum value for this is 1-(1/(sx*sy)). By default, the max is 0.75 (I recommend <= 0.5 though).
              Higher values result in more speed-up, but with more visual quality loss.
    
    Args to tinker with if you want:
     - max_downsample [1, 2, 4, or 8]: Apply ToMe to layers with at most this amount of downsampling.
                                       E.g., 1 only applies to layers with no downsampling (4/15) while
                                       8 applies to all layers (15/15). I recommend a value of 1 or 2.
     - sx, sy: The stride for computing dst sets (see paper). A higher stride means you can merge more tokens,
               but the default of (2, 2) works well in most cases. Doesn't have to divide image size.
     - use_rand: Whether or not to allow random perturbations when computing dst sets (see paper). Usually
                 you'd want to leave this on, but if you're having weird artifacts try turning this off.
     - merge_attn: Whether or not to merge tokens for attention (recommended).
     - merge_crossattn: Whether or not to merge tokens for cross attention (not recommended).
     - merge_mlp: Whether or not to merge tokens for the mlp layers (very not recommended).
    """

    # Make sure the module is not currently patched
    remove_patch(model)

    is_diffusers = isinstance_str(model, "DiffusionPipeline") or isinstance_str(model, "ModelMixin")

    if not is_diffusers:
        if not hasattr(model, "model") or not hasattr(model.model, "diffusion_model"):
            # Provided model not supported
            raise RuntimeError("Provided model was not a Stable Diffusion / Latent Diffusion model, as expected.")
        diffusion_model = model.model.diffusion_model
    else:
        # Supports "pipe.unet" and "unet"
        diffusion_model = model.unet if hasattr(model, "unet") else model

    diffusion_model._tome_info = {
        "size": None,
        "hooks": [],
        "args": {
            "ratio": ratio,
            "max_downsample": max_downsample,
            "sx": sx, "sy": sy,
            "use_rand": use_rand,
            "generator": None,
            "merge_attn": merge_attn,
            "merge_crossattn": merge_crossattn,
            "merge_mlp": merge_mlp,
            "use_dft": use_dft,
            "trunc_ratio": trunc_ratio,
            "high_weight": high_weight,
            "low_weight": low_weight,
            "trunc_ratio_min": trunc_ratio_min,
            "trunc_ratio_max": trunc_ratio_max,
            "max_timestep": max_timestep,
            "total_layers": total_layers,
            "adaptive_trunc": adaptive_trunc,
            "adaptive_basis": adaptive_basis,
            "adaptive_direction": adaptive_direction,
        }
    }

    #print(model)
    '''
    # --- collect all transformer blocks ---
    blocks = []
    for module in diffusion_model.modules():
        if isinstance_str(module, "BasicTransformerBlock"):
            blocks.append(module)

    # --- assign layer_idx to each block ---
    for idx, module in enumerate(blocks):
        print(f"Layer {idx}: {module.__class__.__name__}")
        module._tome_info = diffusion_model._tome_info
        module._layer_idx = idx

        # swap in your ToMeBlock subclass
        make_fn = make_diffusers_tome_block if is_diffusers else make_tome_block
        module.__class__ = make_fn(module.__class__)
    '''

    hook_tome_model(diffusion_model)

    tome_block_cnt = 0
    tome_diffusers_block_cnt = 0
    layer_idx = 0

    total_layers = sum(
        1 for _ in diffusion_model.modules()
        if isinstance_str(_, "BasicTransformerBlock")
    )


    layer_idx = 0



    for _, module in diffusion_model.named_modules():
        # If for some reason this has a different name, create an issue and I'll fix it

        if isinstance_str(module, "BasicTransformerBlock"):

            make_tome_block_fn = make_diffusers_tome_block if is_diffusers else make_tome_block
            module.__class__ = make_tome_block_fn(module.__class__)
            
            module._tome_info = {
                "size": None,
                "hooks": diffusion_model._tome_info["hooks"],
                "args": diffusion_model._tome_info["args"].copy(),
                "shared": diffusion_model._tome_info  # shared view of timestep
            }
            module._tome_info["layer_idx"] = layer_idx
            module._tome_info["args"]["total_layers"] = total_layers



            layer_idx += 1

            

            # Something introduced in SD 2.0 (LDM only)
            if not hasattr(module, "disable_self_attn") and not is_diffusers:
                module.disable_self_attn = False

            # Something needed for older versions of diffusers
            if not hasattr(module, "use_ada_layer_norm_zero") and is_diffusers:
                module.use_ada_layer_norm = False
                module.use_ada_layer_norm_zero = False

    
    if is_diffusers:
        patch_unet_forward(diffusion_model)

    return model





def remove_patch(model: torch.nn.Module):
    """ Removes a patch from a ToMe Diffusion module if it was already patched. """
    # For diffusers
    model = model.unet if hasattr(model, "unet") else model

    for _, module in model.named_modules():
        if hasattr(module, "_tome_info"):
            for hook in module._tome_info["hooks"]:
                hook.remove()
            module._tome_info["hooks"].clear()

        if module.__class__.__name__ == "ToMeBlock":
            module.__class__ = module._parent
    
    return model
