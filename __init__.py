#shamelessly taken from forge

import logging
import comfy.model_patcher
import folder_paths

import torch
import collections
import bitsandbytes as bnb
import comfy
import copy

from bitsandbytes.nn.modules import Params4bit, QuantState
from bitsandbytes.functional import dequantize_4bit
from comfy.cli_args import args as cli_args
from comfy.model_patcher import ModelPatcher, string_to_seed

from .float_nf4 import stochastic_rounding_nf4

rounding_format_default = '2,1,7'
dtype_from_str = {"default": None, "float8_e4m3fn": torch.float8_e4m3fn, "float8_e5m2": torch.float8_e5m2}

def functional_linear_4bits(x, weight, bias):
    out = bnb.matmul_4bit(x, weight.t(), bias=bias, quant_state=weight.quant_state)
    out = out.to(x)
    return out

def functional_dequantize_4bit(weight):
    if not weight.bnb_quantized:
        return weight

    weight_original_device = weight.device

    if weight_original_device.type != 'cuda':
        weight = weight.cuda()

    weight = dequantize_4bit(weight, quant_state=weight.quant_state, blocksize=weight.blocksize, quant_type=weight.quant_type)

    if weight_original_device.type != 'cuda':
        weight = weight.to(device=weight_original_device)

    return weight

def copy_quant_state(state: QuantState, device: torch.device = None) -> QuantState:
    if state is None:
        return None

    device = device or state.absmax.device

    state2 = (
        QuantState(
            absmax=state.state2.absmax.to(device),
            shape=state.state2.shape,
            code=state.state2.code.to(device),
            blocksize=state.state2.blocksize,
            quant_type=state.state2.quant_type,
            dtype=state.state2.dtype,
        )
        if state.nested
        else None
    )

    return QuantState(
        absmax=state.absmax.to(device),
        shape=state.shape,
        code=state.code.to(device),
        blocksize=state.blocksize,
        quant_type=state.quant_type,
        dtype=state.dtype,
        offset=state.offset.to(device) if state.nested else None,
        state2=state2,
    )

class ForgeParams4bit(Params4bit):
    _torch_fn_depth=0

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if cls._torch_fn_depth > 0 or func != torch._C.TensorBase.detach:
            return super().__torch_function__(func, types, args, kwargs or {})
        cls._torch_fn_depth += 1
        try:
            slf = args[0]
            n = cls(
                    torch.nn.Parameter.detach(slf),
                    requires_grad=slf.requires_grad,
                    quant_state=copy_quant_state(slf.quant_state, slf.device),
                    blocksize=slf.blocksize,
                    compress_statistics=slf.compress_statistics,
                    quant_type=slf.quant_type,
                    quant_storage=slf.quant_storage,
                    bnb_quantized=slf.bnb_quantized,
                    module=slf.module
                )
            return n
        finally:
            cls._torch_fn_depth -= 1

    def to(self, *args, copy=False, **kwargs):
        if copy:
            return self.clone().to(*args, **kwargs)
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None and device.type == "cuda" and not self.bnb_quantized:
            if self.data.dtype != torch.bfloat16 and self.data.dtype != torch.float16:
                self.data = self.data.to(torch.device('cuda')).to(torch.bfloat16)
            
            # after called, model converted in to nf4 and enabled loras bad work!
            # when lora exist in WF, 'patch_weight_to_device' backup/restore fp8 weight and loras work good

            # import traceback
            # traceback.print_stack()
            return self._quantize(device)

            weight = ForgeParams4bit(
                self, #self.data, #.to(torch.bfloat16, copy=True),
                requires_grad=False,
                compress_statistics=self.compress_statistics,
                blocksize=self.blocksize,
                quant_type='nf4',
                quant_storage=torch.uint8,
                quant_state=copy_quant_state(self.quant_state, device),
                bnb_quantized=self.bnb_quantized,
                module=self.module
            )

            weight = weight._quantize(device)
            return weight
        else:
            n = self.__class__(
                torch.nn.Parameter.to(self, device=device, dtype=dtype, non_blocking=non_blocking),
                requires_grad=self.requires_grad,
                quant_state=copy_quant_state(self.quant_state, device),
                blocksize=self.blocksize,
                compress_statistics=self.compress_statistics,
                quant_type=self.quant_type,
                quant_storage=self.quant_storage,
                bnb_quantized=self.bnb_quantized,
                module=self.module
            )
            # self.module.quant_state = n.quant_state
            self.data = n.data
            self.quant_state = n.quant_state
            return n


class ForgeLoader4Bit(torch.nn.Module):
    def __init__(self, *, device, dtype, quant_type, **kwargs):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.empty(1, device=device, dtype=dtype))
        self.weight = None
        self.quant_state = None
        self.bias = None
        self.quant_type = quant_type

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        quant_state = getattr(self.weight, "quant_state", None)
        if quant_state is not None:
            for k, v in quant_state.as_dict(packed=True).items():
                destination[prefix + "weight." + k] = v if keep_vars else v.detach()
        return

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        quant_state_keys = {k[len(prefix + "weight."):] for k in state_dict.keys() if k.startswith(prefix + "weight.")}

        if any('bitsandbytes' in k for k in quant_state_keys):
            quant_state_dict = {k: state_dict[prefix + "weight." + k] for k in quant_state_keys}

            self.weight = ForgeParams4bit.from_prequantized(
                data=state_dict[prefix + 'weight'],
                quantized_stats=quant_state_dict,
                requires_grad=False,
                device=self.dummy.device,
                module=self
            )
            self.quant_state = self.weight.quant_state

            if prefix + 'bias' in state_dict:
                self.bias = torch.nn.Parameter(state_dict[prefix + 'bias'].to(self.dummy))

            del self.dummy
        elif hasattr(self, 'dummy'):
            if prefix + 'weight' in state_dict:
                self.weight = ForgeParams4bit(
                    state_dict[prefix + 'weight'].to(self.dummy),
                    requires_grad=False,
                    compress_statistics=True,
                    quant_type=self.quant_type,
                    quant_storage=torch.uint8,
                    module=self,
                )
                self.quant_state = self.weight.quant_state

            if prefix + 'bias' in state_dict:
                self.bias = torch.nn.Parameter(state_dict[prefix + 'bias'].to(self.dummy))

            del self.dummy
        else:
            super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


import comfy.ops

class NF4ModelPatcher(ModelPatcher):
    rounding_format = rounding_format_default

    def patch_weight_to_device(self, key, device_to=None, inplace_update=False):
        if key not in self.patches:
            return

        weight = comfy.utils.get_attr(self.model, key)

        inplace_update = self.weight_inplace_update or inplace_update

        if key not in self.backup:
            self.backup[key] = collections.namedtuple('Dimension', ['weight', 'inplace_update'])(weight.to(device=self.offload_device, copy=inplace_update), inplace_update)

        bnb_layer = hasattr(weight, 'bnb_quantized')
        bnb_kwargs = {}
        if bnb_layer:
            bnb_kwargs = {
                'compress_statistics': weight.compress_statistics,
                'blocksize': weight.blocksize,
                'quant_type': weight.quant_type,
                'quant_storage': torch.uint8,
                'quant_state': copy_quant_state(weight.quant_state, weight.device),
                'bnb_quantized': weight.bnb_quantized,
                'module': weight.module
            }
            weight = functional_dequantize_4bit(weight)
            
        # temp_weight = weight.to(torch.device('cuda'), copy=True, non_blocking=False).to(torch.bfloat16)
        if device_to is not None:
            temp_weight = comfy.model_management.cast_to_device(weight, device_to, torch.float32, copy=True)
        else:
            temp_weight = weight.to(torch.float32, copy=True)

        out_weight = comfy.lora.calculate_weight(self.patches[key], temp_weight, key)
        
        # To-do: Fix image burnout
        if self.rounding_format is not None:
            out_weight = stochastic_rounding_nf4(out_weight, self.rounding_format, seed=string_to_seed(key))
        else:
            out_weight = comfy.float.stochastic_rounding(out_weight, torch.float8_e4m3fn, seed=string_to_seed(key))

        out_weight = NF4ModelPatcher.reload_weight(out_weight.to(torch.bfloat16), **bnb_kwargs) # .float()
        # out_weight.to(torch.device('cpu'))

        if inplace_update:
            comfy.utils.copy_to_param(self.model, key, out_weight)
        else:
            comfy.utils.set_attr_param(self.model, key, out_weight)

    @staticmethod
    def reload_weight(weight, **kwargs):
        weight_original_device = weight.device
        weight = ForgeParams4bit(
            weight,
            requires_grad=False,
            **kwargs
        )

        weight = weight._quantize(weight_original_device)
        
        # if weight_original_device.type == 'cuda':
        #     weight = weight.to(weight_original_device)
        # else:
        #     weight = weight.cuda().to(weight_original_device)
        
        return weight
    
    def load(self, device_to=None, lowvram_model_memory=0, force_patch_weights=False, full_load=False):
        # revert broken changes https://github.com/comfyanonymous/ComfyUI/commit/bc6be6c11e48114889a368e8c3597df8aac64ae3

        mem_counter = 0
        patch_counter = 0
        lowvram_counter = 0
        loading = self._load_list()

        load_completely = []
        loading.sort(reverse=True)
        for x in loading:
            n = x[1]
            m = x[2]
            params = x[3]
            module_mem = x[0]

            lowvram_weight = False

            if not full_load and hasattr(m, "comfy_cast_weights"):
                if mem_counter + module_mem >= lowvram_model_memory:
                    lowvram_weight = True
                    lowvram_counter += 1
                    if hasattr(m, "prev_comfy_cast_weights"): #Already lowvramed
                        continue

            weight_key = "{}.weight".format(n)
            bias_key = "{}.bias".format(n)

            if lowvram_weight:
                if weight_key in self.patches:
                    if force_patch_weights:
                        self.patch_weight_to_device(weight_key)
                    else:
                        m.weight_function = comfy.model_patcher.LowVramPatch(weight_key, self.patches)
                        patch_counter += 1
                if bias_key in self.patches:
                    if force_patch_weights:
                        self.patch_weight_to_device(bias_key)
                    else:
                        m.bias_function = comfy.model_patcher.LowVramPatch(bias_key, self.patches)
                        patch_counter += 1

                m.prev_comfy_cast_weights = m.comfy_cast_weights
                m.comfy_cast_weights = True
            else:
                if hasattr(m, "comfy_cast_weights"):
                    if m.comfy_cast_weights:
                        comfy.model_patcher.wipe_lowvram_weight(m)

                mem_counter += module_mem
                load_completely.append((module_mem, n, m, params))
                # if full_load or mem_counter + module_mem < lowvram_model_memory:
                #     mem_counter += module_mem
                #     load_completely.append((module_mem, n, m, params))

        load_completely.sort(reverse=True)
        for x in load_completely:
            n = x[1]
            m = x[2]
            params = x[3]
            if hasattr(m, "comfy_patched_weights"):
                if m.comfy_patched_weights == True:
                    continue

            for param in params:
                self.patch_weight_to_device("{}.{}".format(n, param), device_to=device_to)

            logging.debug("lowvram: loaded module regularly {} {}".format(n, m))
            m.comfy_patched_weights = True

        for x in load_completely:
            x[2].to(device_to)

        if lowvram_counter > 0:
            logging.info("loaded partially {} {} {}".format(lowvram_model_memory / (1024 * 1024), mem_counter / (1024 * 1024), patch_counter))
            self.model.model_lowvram = True
        else:
            logging.info("loaded completely {} {} {}".format(lowvram_model_memory / (1024 * 1024), mem_counter / (1024 * 1024), full_load))
            self.model.model_lowvram = False
            if full_load:
                self.model.to(device_to)
                mem_counter = self.model_size()

        self.model.lowvram_patch_counter += patch_counter
        self.model.device = device_to
        self.model.model_loaded_weight_memory = mem_counter

    # def partially_unload(self, device_to, memory_to_free=0):
    #     for n, m in self.model.named_modules():
    #         if not hasattr(m, "comfy_cast_weights"): # and 'make_ops.<locals>.OPS.Linear' in str(m.__class__):
    #             # logging.info(f'n: {n}')
    #             # append attr from comfy.ops.CastWeightBiasOp for partially unloading weights
    #             # class Linear(loader_class, comfy.ops.CastWeightBiasOp) raise Exception 'quant_state is not None'
    #             m.comfy_cast_weights = False

    #     mod_size = self.model_size()
    #     result = super().partially_unload(device_to, memory_to_free)
    #     logging.info(f'[{self.model.__class__.__name__} ({mod_size/1024**3:.1f}gb)] partially_unload: {device_to}, memory_to_free={memory_to_free/1024**3:.1f}gb / result={result/1024**3:.1f}gb')
    #     return result
    
    def partially_unload(self, device_to, memory_to_free=0):
        memory_freed = super().partially_unload(device_to, memory_to_free)

        for n, m in self.model.named_modules():
            if memory_to_free < memory_freed:
                break
            if 'make_ops.<locals>.OPS.Linear' not in str(m.__class__):
                continue

            msize = comfy.model_management.module_size(m)
            m.to(device_to)
            memory_freed += msize

        self.model.model_loaded_weight_memory -= memory_freed
        mod_size = self.model_size()
        logging.info(f'[{self.model.__class__.__name__} ({mod_size/1024**3:.1f}gb)] partially_unload: {device_to}, memory_to_free={memory_to_free/1024**3:.1f}gb / result={memory_freed/1024**3:.1f}gb')
        return memory_freed
    
    def partially_load(self, device_to, extra_memory=0):
        for n, m in self.model.named_modules():
            if 'make_ops.<locals>.OPS.Linear' in str(m.__class__):
                if m.weight.device != device_to:
                    m.to(device_to)
        
        logging.info(f'[{self.model.__class__.__name__}] partially_load: {device_to}')

        return super().partially_load(device_to, extra_memory)
    
    def clone(self, *args, **kwargs):
        n = NF4ModelPatcher(self.model, self.load_device, self.offload_device, self.size, weight_inplace_update=self.weight_inplace_update)
        n.patches = {}
        for k in self.patches:
            n.patches[k] = self.patches[k][:]
        n.patches_uuid = self.patches_uuid

        n.object_patches = self.object_patches.copy()
        n.model_options = copy.deepcopy(self.model_options)
        n.backup = self.backup
        n.object_patches_backup = self.object_patches_backup
        n.rounding_format = getattr(self, "rounding_format", rounding_format_default)
        return n


def make_ops(loader_class, current_device = None, current_dtype = None, current_manual_cast_enabled = False, current_bnb_dtype = None):

    class OPS(comfy.ops.manual_cast):
        class Linear(loader_class):
        # class Linear(loader_class, comfy.ops.CastWeightBiasOp):
            def __init__(self, *args, device=None, dtype=None, **kwargs):
                super().__init__(device=device, dtype=dtype, quant_type=current_bnb_dtype)
                self.parameters_manual_cast = current_manual_cast_enabled

            def forward(self, x):
                self.weight.quant_state = self.quant_state

                if self.bias is not None and self.bias.dtype != x.dtype:
                    # Maybe this can also be set to all non-bnb ops since the cost is very low.
                    # And it only invokes one time, and most linear does not have bias
                    self.bias.data = self.bias.data.to(x.dtype)

                if not self.parameters_manual_cast:
                    return functional_linear_4bits(x, self.weight, self.bias)
                elif not self.weight.bnb_quantized:
                    assert x.device.type == 'cuda', 'BNB Must Use CUDA as Computation Device!'
                    layer_original_device = self.weight.device
                    self.weight = self.weight._quantize(x.device)
                    bias = self.bias.to(x.device) if self.bias is not None else None
                    out = functional_linear_4bits(x, self.weight, bias)
                    self.weight = self.weight.to(layer_original_device)
                    return out
                else:
                    raise RuntimeError("Unexpected state in forward")

    return OPS

class SP_CheckpointLoaderBNB:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
            "load_dtype": (("default", "float8_e4m3fn", "float8_e5m2"), {"default": "float8_e4m3fn"}),
         }}

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"

    CATEGORY = "loaders"

    def load_checkpoint(self, ckpt_name, load_clip='True', load_vae='True', load_dtype='default', bnb_dtype='nf4', rounding_format='default', custom_rounding_format=rounding_format_default):
        if bnb_dtype == "default":
            bnb_dtype = None
        ops = make_ops(ForgeLoader4Bit, current_bnb_dtype = bnb_dtype)
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        model, clip, vae, _ = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=load_vae=="True", output_clip=load_clip=="True", embedding_directory=folder_paths.get_folder_paths("embeddings"), model_options={"custom_operations": ops, "dtype": dtype_from_str[load_dtype]})

        model = NF4ModelPatcher.clone(model)

        if rounding_format=='default':
            rounding_format = None
        elif rounding_format=='custom':
            rounding_format = custom_rounding_format
        model.rounding_format = rounding_format

        return model, clip, vae

class SP_CheckpointLoaderBNB_Advanced(SP_CheckpointLoaderBNB):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
            "load_clip": (["True", "False"], ),
            "load_vae": (["True", "False"], ),
            "load_dtype": (("default", "float8_e4m3fn", "float8_e5m2"), {"default": "float8_e4m3fn"}),
            "bnb_dtype": (("default", "nf4", "fp4"), {"default": "nf4"}),
            "rounding_format": (("default", "2,1,7", "custom"), ),
            "custom_rounding_format": ("STRING", {"default": rounding_format_default}),
         }}

class SP_UnetLoaderBNB:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "unet_name": (folder_paths.get_filename_list("unet"), ),
            "load_dtype": (("default", "float8_e4m3fn", "float8_e5m2"), {"default": "float8_e4m3fn"}),
         }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_checkpoint"

    CATEGORY = "loaders"

    def load_checkpoint(self, unet_name, load_dtype='default', bnb_dtype='nf4', rounding_format='default', custom_rounding_format=rounding_format_default):
        if bnb_dtype == "default":
            bnb_dtype = None
        ops = make_ops(ForgeLoader4Bit, current_bnb_dtype = bnb_dtype)
        unet_path = folder_paths.get_full_path("unet", unet_name)
        model = comfy.sd.load_diffusion_model(unet_path, model_options={"custom_operations": ops, "dtype": dtype_from_str[load_dtype]})

        model = NF4ModelPatcher.clone(model)

        if rounding_format=='default':
            rounding_format = None
        elif rounding_format=='custom':
            rounding_format = custom_rounding_format
        model.rounding_format = rounding_format

        return model, 
    
class SP_UnetLoaderBNB_Advanced(SP_UnetLoaderBNB):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "unet_name": (folder_paths.get_filename_list("unet"), ),
            "load_dtype": (("default", "float8_e4m3fn", "float8_e5m2"), {"default": "float8_e4m3fn"}),
            "bnb_dtype": (("default", "nf4", "fp4"), {"default": "nf4"}),
            "rounding_format": (("default", "2,1,7", "custom"), ),
            "custom_rounding_format": ("STRING", {"default": rounding_format_default}),
         }}

NODE_CLASS_MAPPINGS = {
    "SP_UnetLoaderBNB": SP_UnetLoaderBNB,
    "SP_CheckpointLoaderBNB": SP_CheckpointLoaderBNB,
    "SP_UnetLoaderBNB_Advanced": SP_UnetLoaderBNB_Advanced,
    "SP_CheckpointLoaderBNB_Advanced": SP_CheckpointLoaderBNB_Advanced,
}

