
try:
    import apex

    _IS_APEX_AVAILABLE = True
except ImportError:
    _IS_APEX_AVAILABLE = False


def is_apex_available() -> bool:
    return _IS_APEX_AVAILABLE


try:
    import deepspeed

    _IS_DEEPSPEED_AVAILABLE = True
except ImportError:
    _IS_DEEPSPEED_AVAILABLE = False


def is_deepspeed_available() -> bool:
    return _IS_DEEPSPEED_AVAILABLE


try:
    import flash_attn

    _IS_FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    _IS_FLASH_ATTENTION_AVAILABLE = False


def is_flash_attention_available() -> bool:
    return _IS_FLASH_ATTENTION_AVAILABLE


try:
    import aim

    _IS_AIM_AVAILABLE = True
except ImportError:
    _IS_AIM_AVAILABLE = False


def is_aim_available() -> bool:
    return _IS_AIM_AVAILABLE


try:
    import wandb

    _IS_WANDB_AVAILABLE = True
except ImportError:
    _IS_WANDB_AVAILABLE = False


def is_wandb_available() -> bool:
    return _IS_WANDB_AVAILABLE


try:
    import colorlog

    _IS_COLORLOG_AVAILABLE = True
except ImportError:
    _IS_COLORLOG_AVAILABLE = False


def is_colorlog_available() -> bool:
    return _IS_COLORLOG_AVAILABLE


try:
    import transformer_engine

    _IS_TRANSFORMER_ENGINE_AVAILABLE = True
except ImportError:
    _IS_TRANSFORMER_ENGINE_AVAILABLE = False


def is_transformer_engine_available() -> bool:
    return _IS_TRANSFORMER_ENGINE_AVAILABLE


try:
    import msamp

    _IS_MS_AMP_AVAILABLE = True
except ImportError:
    _IS_MS_AMP_AVAILABLE = False


def is_ms_amp_available() -> bool:
    return _IS_MS_AMP_AVAILABLE


try:
    import triton

    _IS_TRITON_AVAILABLE = True
except ImportError:
    _IS_TRITON_AVAILABLE = False


def is_triton_available() -> bool:
    return _IS_TRITON_AVAILABLE
