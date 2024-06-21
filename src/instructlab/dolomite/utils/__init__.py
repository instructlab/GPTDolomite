# Local
from .hf_hub import download_repo
from .safetensors import SafeTensorsWeightsManager
from .wrapper import get_module_class_from_name

try:
    # Third Party
    import flash_attn

    _IS_FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    _IS_FLASH_ATTENTION_AVAILABLE = False


def is_flash_attention_available() -> bool:
    return _IS_FLASH_ATTENTION_AVAILABLE
