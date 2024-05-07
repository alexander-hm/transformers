# Quantizer for Hierarchical Residual Learning
#
#
import importlib.metadata
from typing import TYPE_CHECKING

from packaging import version

from .base import HfQuantizer

if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel

from ..utils import is_accelerate_available, is_torch_available, logging
from ..utils.quantization_config import QuantizationConfigMixin

if is_torch_available():
    import torch

logger = logging.get_logger(__name__)

class HRQuantizer(HfQuantizer):
    """
    Quantizer of HR-VQLoRA method. Enables quantizations of residuals.
    """
    requires_calibration = False  
    required_packages = []
    # requires_parameters_quantization = False by default
    # Most of the recent quantization method packs int2/int4 weights inside torch.uint8 weights



    def __init__(self, quantization_config: QuantizationConfigMixin, **kwargs):
        super().__init__(quantization_config, **kwargs)
        self.quantization_config = quantization_config

    # TODO: What do we need to validate our environment
    def validate_environment(self, *args, **kwargs):
        # TODO: DO we need accelerate to quantize
        if not (is_accelerate_available()):
            raise ImportError(
                "Using `bitsandbytes` 8-bit quantization requires Accelerate: `pip install accelerate` "
                "and the latest version of bitsandbytes: `pip install -i https://pypi.org/simple/ bitsandbytes`"
            )

        if kwargs.get("from_tf", False) or kwargs.get("from_flax", False):
            raise ValueError(
                "Converting into 4-bit or 8-bit weights from tf/flax weights is currently not supported, please make"
                " sure the weights are in PyTorch format."
            )

        if not torch.cuda.is_available():
            raise RuntimeError("No GPU found. A GPU is needed for quantization.")

    # TODO: How to update torch dtype
    # Copied from transformers.quantizers.quantizer_bnb_8bit.Bnb8BitHfQuantizer.update_torch_dtype
    def update_torch_dtype(self, torch_dtype: "torch.dtype") -> "torch.dtype":
        if torch_dtype is None:
            # We force the `dtype` to be float16, this is a requirement from `bitsandbytes`
            logger.info(
                "Overriding torch_dtype=%s with `torch_dtype=torch.float16` due to "
                "requirements of `bitsandbytes` to enable model loading in 8-bit or 4-bit. "
                "Pass your own torch_dtype to specify the dtype of the remaining non-linear layers or pass"
                " torch_dtype=torch.float16 to remove this warning.",
                torch_dtype,
            )
            torch_dtype = torch.float16
        return torch_dtype
    
    # Copied from transformers.quantizers.quantizer_awq.AwqQuantizer.update_torch_dtype
    def update_torch_dtype(self, torch_dtype):
        if torch_dtype is None:
            torch_dtype = torch.float16
        elif torch_dtype != torch.float16:
            logger.warning("We suggest you to set `torch_dtype=torch.float16` for better efficiency with AWQ.")
        return torch_dtype

    # TODO: How do we need to preprocess the model before loading weights
    # TODO: Likely have to replace nn.Linear with our custom residual layer
    # Copied from transformers.quantizers.quantizer_bnb_8bit.Bnb8BitHfQuantizer._process_model_before_weight_loading
    def _process_model_before_weight_loading(self, model: "PreTrainedModel", **kwargs):
        from ..integrations import get_keys_to_not_convert, replace_with_hr_linear

        load_in_8bit_fp32_cpu_offload = self.quantization_config.llm_int8_enable_fp32_cpu_offload

        # We keep some modules such as the lm_head in their original dtype for numerical stability reasons
        if self.quantization_config.llm_int8_skip_modules is None:
            self.modules_to_not_convert = get_keys_to_not_convert(model)
        else:
            self.modules_to_not_convert = self.quantization_config.llm_int8_skip_modules

        if not isinstance(self.modules_to_not_convert, list):
            self.modules_to_not_convert = [self.modules_to_not_convert]

        self.modules_to_not_convert.extend(keep_in_fp32_modules)

        # Extend `self.modules_to_not_convert` to keys that are supposed to be offloaded to `cpu` or `disk`
        if isinstance(device_map, dict) and len(device_map.keys()) > 1:
            keys_on_cpu = [key for key, value in device_map.items() if value in ["disk", "cpu"]]

            if len(keys_on_cpu) > 0 and not load_in_8bit_fp32_cpu_offload:
                raise ValueError(
                    "If you want to offload some keys to `cpu` or `disk`, you need to set "
                    "`llm_int8_enable_fp32_cpu_offload=True`. Note that these modules will not be "
                    " converted to 8-bit but kept in 32-bit."
                )
            self.modules_to_not_convert.extend(keys_on_cpu)

        model = replace_with_bnb_linear(
            model, modules_to_not_convert=self.modules_to_not_convert, quantization_config=self.quantization_config
        )
        # TODO: consider bringing replace_with_bnb_linear() code from ..integrations/bitsandbyter.py to here

        model.config.quantization_config = self.quantization_config

    def _process_model_after_weight_loading(self, model):
        return

    @property
    def is_serializable(self):
        return True
    
    def is_trainable(self):
        return True