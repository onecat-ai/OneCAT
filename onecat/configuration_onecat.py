import copy
import os
from typing import Union

from transformers import LlamaConfig, Qwen2Config
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class PatchEmbeddingConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`PatchEmbeddingLayer`]. It is used to
    instantiate a vision encoder according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        patch_size (`int`, *optional*, defaults to 14):
            The size (resolution) of each patch.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        hidden_size (`int`, *optional*, defaults to 3200):
            Dimensionality of the encoder layers and the pooler layer.
   
    """

    model_type = 'patch_layer' 

    def __init__(
            self,
            patch_size=14,
            image_size=224,
            hidden_size=3200,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.image_size = image_size

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> 'PretrainedConfig':
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        if 'model_type' in config_dict and hasattr(cls, 'model_type') and config_dict['model_type'] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f'{cls.model_type}. This is not supported for all configurations of models and can yield errors.'
            ) 

        
        return cls.from_dict(config_dict, **kwargs)
    
    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = dict(
            hidden_size=self.hidden_size,
            patch_size=self.patch_size,
            image_size=self.image_size,
        )
        return output


class Qwen2Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Qwen2Model`]. It is used to instantiate a
    Qwen2 model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of
    Qwen2-7B-beta [Qwen/Qwen2-7B-beta](https://huggingface.co/Qwen/Qwen2-7B-beta).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 151936):
            Vocabulary size of the Qwen2 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Qwen2Model`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 22016):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 32):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1 the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to `32`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 32768):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        use_sliding_window (`bool`, *optional*, defaults to `False`):
            Whether to use sliding window attention.
        sliding_window (`int`, *optional*, defaults to 4096):
            Sliding window attention (SWA) window size. If not specified, will default to `4096`.
        max_window_layers (`int`, *optional*, defaults to 28):
            The number of layers that use SWA (Sliding Window Attention). The bottom layers use SWA while the top use full attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.

    ```python
    >>> from transformers import Qwen2Model, Qwen2Config

    >>> # Initializing a Qwen2 style configuration
    >>> configuration = Qwen2Config()

    >>> # Initializing a model from the Qwen2-7B style configuration
    >>> model = Qwen2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "qwen2"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=22016,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=28,
        attention_dropout=0.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.max_window_layers = max_window_layers

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class OneCatVLChatConfig(PretrainedConfig):
    r"""
    Configuration for the OneCat vision-language chat model.

    This configuration composes two sub-configurations:
    - ``patch_vision_config``: A PatchEmbedding configuration describing how images are split into patches
      and projected to the vision hidden size.
    - ``llm_config``: A language model configuration (Llama or Qwen2) describing the causal LM backbone.

    The language model configuration to instantiate is inferred from ``llm_config['architectures'][0]``:
    - ``'LlamaForCausalLM'`` -> uses ``transformers.LlamaConfig``
    - ``'Qwen2ForCausalLM'`` -> uses the Qwen2-style config defined in this file
    - Any other value falls back to the Qwen2-style config

    Args:
        patch_vision_config (dict or PatchEmbeddingConfig, optional):
            The vision patch embedding config or its dictionary form. If ``None``, a default
            ``PatchEmbeddingConfig`` is created.
        llm_config (dict or PretrainedConfig, optional):
            The LLM config (its dictionary form is expected). Must contain an ``architectures`` field
            with the first element indicating the target model class name as described above. If ``None``,
            a default Qwen2-style configuration is used.
        downsample_ratio (float, optional, defaults to 0.5):
            Spatial downsample ratio applied to visual features before projecting to the LLM hidden size.
        template (str, optional):
            Name of the chat template/prompt preset used by downstream conversation formatting.
        **kwargs: Additional keyword arguments forwarded to ``PretrainedConfig``.

    Attributes:
        model_type (str): Always ``'onecat'``.
        is_composition (bool): Always ``True`` to indicate this config nests other configs.
        patch_vision_config (PatchEmbeddingConfig): Vision patch embedding sub-config.
        llm_config (PretrainedConfig): Backbone LLM sub-config.
        hidden_size (int): Mirrors ``llm_config.hidden_size``.
        tie_word_embeddings (bool): Always set to ``False`` and applied to ``llm_config`` as well.
    
    """

    model_type = 'onecat'
    is_composition = True

    def __init__(
            self,
            patch_vision_config=None,
            llm_config=None,
            downsample_ratio=0.5,
            template=None,
            **kwargs):
        super().__init__(**kwargs)

        if patch_vision_config is None:
            patch_vision_config = {'architectures': ['PatchEmbeddingLayer']}
            logger.info('patch_vision_config is None. Initializing the PatchEmbeddingConfig with default values.')

        if llm_config is None:
            llm_config = {'architectures': ['']}
            logger.info('llm_config is None. Initializing the LLM config with default values (fallback to Qwen2Config).')

        self.patch_vision_config = PatchEmbeddingConfig(**patch_vision_config)
        if llm_config['architectures'][0] == 'LlamaForCausalLM':
            self.llm_config = LlamaConfig(**llm_config)
        elif llm_config['architectures'][0] == 'Qwen2ForCausalLM':
            self.llm_config = Qwen2Config(**llm_config)
        else:
            self.llm_config = Qwen2Config()

        self.downsample_ratio = downsample_ratio
        self.template = template
        self.hidden_size = self.llm_config.hidden_size
        # By default, we use tie_word_embeddings=False for models of all sizes.
        self.tie_word_embeddings = False
        self.llm_config.tie_word_embeddings = self.tie_word_embeddings

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output['patch_vision_config'] = self.patch_vision_config.to_dict()
        output['llm_config'] = self.llm_config.to_dict()
        output['model_type'] = self.__class__.model_type
        output['downsample_ratio'] = self.downsample_ratio
        output['template'] = self.template
        return output
