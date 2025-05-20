"""A HuggingFace-style model configuration."""

from transformers import PretrainedConfig
from math import sqrt


class RavenConfig(PretrainedConfig):
    model_type = "huginn_raven"
    keys_to_ignore_at_inference = [""]
    attribute_map = {"num_attention_heads": "n_heads", "hidden_size": "n_embd", "num_hidden_layers": "n_layers"}

    def __init__(
        self,
        n_embd: int = 5280,
        n_heads: int = 55,
        n_layers: int = 8,  # total of prelude + recurrent + coda
        block_size: int = 4096,
        vocab_size: int = 65536,
        padding_multiple: int = 4096,
        tie_embeddings: bool = True,
        intermediate_size: int = 17920,
        bias: bool = False,
        architecture_class_name: str = "RecurrentGPT",
        block_class_name: str = "SandwichBlock",
        norm_class_name: str = "RMSNorm_llama",
        norm_eps: float = 0.000001,
        mlp_class_name: str = "GatedMLP",
        nonlin_name: str = "SiLU",
        init_strategy: str = "takase",
        init_orthogonal: bool = False,
        state_init: str = "like-init",
        injection_type: str = "linear",
        n_layers_in_recurrent_block: int = 4,
        mean_recurrence: int = 32,
        sampling_scheme: str = "poisson-lognormal-filling",
        mean_backprop_depth: int = 8,
        n_layers_in_prelude: int = 2,
        n_layers_in_coda: int = 2,
        qk_bias: bool = True,
        activation_checkpoint_impl: str = "per-iteration",
        rope_base: float = 50_000,
        torch_dtype: str = "bfloat16",
        transformers_version: str = "4.47.1",
        **kwargs,
    ):
        self.n_embd = n_embd
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.block_size = block_size
        self.vocab_size = self.padded_vocab_size = vocab_size
        self.padding_multiple = padding_multiple
        self.tie_embeddings = tie_embeddings
        self.intermediate_size = intermediate_size
        self.bias = bias
        self.architecture_class_name = architecture_class_name
        self.block_class_name = block_class_name
        self.norm_class_name = norm_class_name
        self.norm_eps = norm_eps
        self.mlp_class_name = mlp_class_name
        self.nonlin_name = nonlin_name
        self.init_strategy = init_strategy
        self.init_orthogonal = init_orthogonal
        self.state_init = state_init
        self.injection_type = injection_type
        self.n_layers_in_recurrent_block = n_layers_in_recurrent_block
        self.mean_recurrence = mean_recurrence
        self.sampling_scheme = sampling_scheme
        self.mean_backprop_depth = mean_backprop_depth
        self.n_layers_in_prelude = n_layers_in_prelude
        self.n_layers_in_coda = n_layers_in_coda
        self.qk_bias = qk_bias
        self.activation_checkpoint_impl = activation_checkpoint_impl
        self.rope_base = rope_base
        self.torch_dtype = torch_dtype  # Added from JSON
        self.transformers_version = transformers_version  # Added from JSON
        # inference
        self.test_time_noise = 0
        self.test_time_noise_type = "fixed"
        # Derived
        self.num_key_value_heads = n_heads
        self.num_attention_heads = n_heads
        self.head_dim = n_embd // n_heads
        self.effective_expected_depth = (
            self.n_layers_in_prelude + self.n_layers_in_coda + self.n_layers_in_recurrent_block * self.mean_recurrence
        )
        self.init_values = {
            "std": sqrt(2 / (5 * self.n_embd)),
            "out_proj": sqrt(2 / (5 * self.n_embd)) / sqrt(2 * self.effective_expected_depth),
            "embedding": sqrt(2 / (5 * self.n_embd)),
            "embed_scale": sqrt(self.n_embd),
        }

        super().__init__(
            # pad_token_id=65509,
            # bos_token_id=65504,
            # eos_token_id=65505,
            tie_word_embeddings=tie_embeddings,
            **kwargs,
        )
