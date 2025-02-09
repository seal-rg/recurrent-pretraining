# type: ignore
from copy import deepcopy

"""This config file contains only working / in-progress architecture definitions for model_dynamic.py
"""

configs = []

###############
# Baselines
###############

baselines = [
    dict(
        name="baby-llama-200m",
        hf_config=dict(org="tomg-group-umd", name="baby-llama-200m"),
        architecture_class_name="GPT",
        block_size=4096,
        vocab_size=32000,
        padding_multiple=32768,
        n_layer=12,
        num_attention_heads=32,
        num_key_value_heads=8,
        n_embd=1024,
        bias=False,
        norm_class_name="RMSNorm_llama",
        norm_eps=1e-5,
        mlp_class_name="GatedMLP",
        nonlin_name="SiLU",
        intermediate_size=2816,
        init_strategy="scaled",
    ),
    dict(
        name="baby-llama-long-200m",
        hf_config=dict(org="tomg-group-umd", name="baby-llama-long-200m"),
        architecture_class_name="GPT",
        block_size=4096,
        vocab_size=32000,
        padding_multiple=4096,
        n_layer=24,
        num_attention_heads=32,
        num_key_value_heads=8,
        n_embd=768,
        bias=False,
        norm_class_name="RMSNorm_llama",
        norm_eps=1e-5,
        mlp_class_name="GatedMLP",
        nonlin_name="SiLU",
        intermediate_size=2048,
        init_strategy="scaled",
    ),
    dict(
        name="baby-snake-200m",
        hf_config=dict(org="tomg-group-umd", name="baby-llama-long-200m"),
        architecture_class_name="GPT",
        block_size=4096,
        vocab_size=32000,
        padding_multiple=4096,
        n_layer=48,
        num_attention_heads=32,
        num_key_value_heads=8,
        n_embd=512,
        bias=False,
        norm_class_name="RMSNorm_llama",
        norm_eps=1e-5,
        mlp_class_name="GatedMLP",
        nonlin_name="SiLU",
        intermediate_size=1536,
        init_strategy="scaled",
    ),
    dict(
        name="gpt2-124m",
        hf_config=dict(org="tomg-group-umd", name="gpt2-124m"),
        architecture_class_name="GPT",
        block_size=1024,
        vocab_size=32000,  # actually 50k but we aren't retokenizing for that
        padding_multiple=4096,
        n_layer=12,
        num_attention_heads=12,
        num_key_value_heads=12,
        n_embd=768,
        bias=False,
        norm_class_name="LayerNorm",
        norm_eps=1e-5,
        mlp_class_name="BaseMLP",
        nonlin_name="GELU",
        intermediate_size=3072,
        init_strategy="scaled",
    ),
]
configs.extend(baselines)

###############
# Sanity Checks
###############

sanity = [
    dict(
        name="brick-200m",
        hf_config=dict(org="tomg-group-umd", name="brick-200m"),
        architecture_class_name="GPT",
        block_size=4096,
        vocab_size=32000,
        padding_multiple=4096,
        n_layer=4,
        n_embd=2048,
        bias=False,
        norm_class_name="Identity",
        norm_eps=1e-5,
        mlp_class_name="BaseMLP",
        nonlin_name="ReLU",
        intermediate_size=8192,
        init_strategy="normal",
        attn_impl="debug-skip",
    ),
    dict(
        name="big-brick-200m",
        hf_config=dict(org="tomg-group-umd", name="big-brick-200m"),
        architecture_class_name="GPT",
        block_size=4096,
        vocab_size=32000,
        padding_multiple=4096,
        n_layer=24,
        n_embd=4096,
        bias=False,
        block_class_name="Brick",
        mlp_class_name="BrickLP",
        nonlin_name="ReLU",
        intermediate_size=4096,
        init_strategy="normal",
        attn_impl="debug-skip",
    ),
    #     Iteration   1024 | Loss: 63.4014 | 3426761036166133342386782208.00 PPL      | Update      8|
    #  (optimizer.step)| MFU : 78.13%  | tok/sec: 144291.1 | steps/sec: 4.40 |
]
configs.extend(sanity)

# ###############
# # Dense
# ###############
weaver = [
    dict(
        name="baby-weaver-200m",
        hf_config=dict(org="tomg-group-umd", name="baby-weaver-200m"),
        block_size=4096,
        vocab_size=32000,
        padding_multiple=4096,
        n_layer=12,
        num_attention_heads=32,
        num_key_value_heads=32,
        n_embd=1024,
        intermediate_size=2816,
        bias=False,
        architecture_class_name="DenseGPT",
        block_class_name="WeaverBlock",
        norm_class_name="RMSNorm_llama",
        norm_eps=1e-5,
        mlp_class_name="GatedMLP",
        nonlin_name="SiLU",
        init_strategy="scaled",
        injection_type="add",
    ),
]
configs.extend(weaver)

# ###############
# # recurrent
# ###############
rec = [
    dict(
        name="baby-recurrent-200m",  # to be used with llama2 tokenizer
        hf_config=dict(org="tomg-group-umd", name="baby-recurrent-200m"),
        block_size=4096,
        vocab_size=32000,
        padding_multiple=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
        n_embd=1024,
        intermediate_size=2816,
        bias=False,
        architecture_class_name="RecurrentGPT",
        block_class_name="TransformerPostNormBlock",
        norm_class_name="RMSNorm_llama",
        norm_eps=1e-5,
        mlp_class_name="GatedMLP",
        nonlin_name="SiLU",
        init_strategy="scaled",
        init_orthogonal=True,
        # pick defaults from config
        injection_type="add",
        n_layers_in_recurrent_block=4,
        mean_recurrence=20,
        mean_backprop_depth=4,
        n_layers_in_prelude=1,
        n_layers_in_coda=1,
    ),
    dict(
        name="debug-recurrent",
        hf_config=dict(org="tomg-group-umd", name="baby-recurrent-200m"),
        block_size=384,
        vocab_size=32000,
        padding_multiple=4096,
        num_attention_heads=16,
        num_key_value_heads=8,
        n_embd=256,
        intermediate_size=256,
        bias=False,
        architecture_class_name="RecurrentGPT",
        block_class_name="TransformerPostNormBlock",
        norm_class_name="RMSNorm_llama",
        norm_eps=1e-5,
        mlp_class_name="GatedMLP",
        nonlin_name="SiLU",
        init_strategy="scaled",
        #
        injection_type="add",
        n_layers_in_recurrent_block=2,
        mean_recurrence=2,
        mean_backprop_depth=1,
        n_layers_in_prelude=1,
        n_layers_in_coda=1,
    ),
    dict(
        name="magpie-150m",
        hf_config=dict(org="tomg-group-umd", name="magpie-150m"),
        block_size=4096,
        vocab_size=65536,
        padding_multiple=4096,
        tie_embeddings=True,
        num_attention_heads=32,
        num_key_value_heads=32,
        n_embd=1024,
        intermediate_size=3520,
        bias=False,
        architecture_class_name="RecurrentGPT",
        block_class_name="TransformerPostNormBlock",
        norm_class_name="RMSNorm_llama",
        norm_eps=1e-5,
        mlp_class_name="GatedMLP",
        nonlin_name="SiLU",
        init_strategy="scaled",
        init_orthogonal=True,
        state_init="like-init",
        # spell out the recurrent settings
        injection_type="add",
        n_layers_in_recurrent_block=4,
        mean_recurrence=20,
        sampling_scheme="poisson-fill",
        mean_backprop_depth=4,
        n_layers_in_prelude=1,
        n_layers_in_coda=1,
        # Model initialized with 60m parameters in recurrent block.
        # Can unfold to 1,946m parameters at test time.
    ),
    dict(
        name="corax-raven-800m",
        hf_config=dict(org="tomg-group-umd", name="corax-raven-800m"),
        block_size=4096,
        vocab_size=65536,
        padding_multiple=4096,
        tie_embeddings=True,
        num_attention_heads=32,
        num_key_value_heads=32,
        n_embd=4096,
        intermediate_size=8192,
        bias=False,
        architecture_class_name="RecurrentGPT",
        block_class_name="TransformerPostNormBlock",
        norm_class_name="RMSNorm_llama",
        norm_eps=1e-5,
        mlp_class_name="GatedMLP",
        nonlin_name="SiLU",
        init_strategy="scaled",
        init_orthogonal=True,
        state_init="like-init",
        # spell out the recurrent settings
        injection_type="add",
        n_layers_in_recurrent_block=2,
        mean_recurrence=20,
        sampling_scheme="poisson-fill",
        mean_backprop_depth=4,  #
        n_layers_in_prelude=1,
        n_layers_in_coda=1,
        # Model initialized with 285m parameters in recurrent block.
        # Can unfold to 9,127m parameters at test time.
    ),
    dict(
        name="kolk-raven-1.1b",
        hf_config=dict(org="tomg-group-umd", name="kolk-raven-1.1b"),
        block_size=4096,
        vocab_size=65536,
        padding_multiple=4096,
        tie_embeddings=True,
        num_attention_heads=64,
        num_key_value_heads=64,
        n_embd=4096,
        intermediate_size=8192,
        bias=False,
        architecture_class_name="RecurrentGPT",
        block_class_name="TransformerPostNormBlock",
        norm_class_name="RMSNorm_llama",
        norm_eps=1e-5,
        mlp_class_name="GatedMLP",
        nonlin_name="SiLU",
        init_strategy="scaled",
        init_orthogonal=True,
        state_init="like-init",
        # spell out the recurrent settings
        injection_type="add",
        n_layers_in_recurrent_block=4,
        mean_recurrence=24,
        sampling_scheme="poisson-fill",
        mean_backprop_depth=8,
        n_layers_in_prelude=1,
        n_layers_in_coda=1,
    ),
    dict(
        name="tower-raven-3b",
        hf_config=dict(org="tomg-group-umd", name="tower-raven-3b"),
        block_size=4096,
        vocab_size=65536,
        padding_multiple=4096,
        tie_embeddings=True,
        num_attention_heads=64,
        num_key_value_heads=64,
        n_embd=4096,
        intermediate_size=8192,
        bias=False,
        architecture_class_name="RecurrentGPT",
        block_class_name="TransformerPostNormBlock",
        norm_class_name="RMSNorm_llama",
        norm_eps=1e-5,
        mlp_class_name="GatedMLP",
        nonlin_name="SiLU",
        init_strategy="scaled",
        init_orthogonal=True,
        state_init="like-init",
        # spell out the recurrent settings
        injection_type="add",
        n_layers_in_recurrent_block=8,
        mean_recurrence=32,
        sampling_scheme="poisson-fill",
        mean_backprop_depth=8,
        n_layers_in_prelude=2,
        n_layers_in_coda=2,
    ),
    dict(
        name="deluge-raven-4b",
        hf_config=dict(org="tomg-group-umd", name="merseburg-raven-4b"),
        block_size=4096,
        vocab_size=65536,
        padding_multiple=4096,
        tie_embeddings=True,
        num_attention_heads=64,
        num_key_value_heads=64,
        n_embd=7040,
        intermediate_size=14080,
        bias=False,
        architecture_class_name="RecurrentGPT",
        block_class_name="TransformerPostNormBlock",
        norm_class_name="RMSNorm_llama",
        norm_eps=1e-5,
        mlp_class_name="GatedMLP",
        nonlin_name="SiLU",
        init_strategy="scaled",
        init_orthogonal=True,
        state_init="like-init",
        # spell out the recurrent settings
        injection_type="add",
        n_layers_in_recurrent_block=4,
        mean_recurrence=32,
        sampling_scheme="poisson-fill",
        mean_backprop_depth=8,
        n_layers_in_prelude=2,
        n_layers_in_coda=2,
    ),
    # Notes:
    # 0: Total parameters: 4,426,421,120
    # 0: Model initialized with 1,982m parameters in recurrent block.
    # 0: Will unfold to 65,884m mean parameters at test time.
    # 0: Can unfold to 129,325m max parameters at test time.
    dict(
        name="merseburg-raven-3.5b",
        hf_config=dict(org="tomg-group-umd", name="merseburg-raven-3.5b"),
        block_size=4096,
        vocab_size=65536,
        padding_multiple=4096,
        tie_embeddings=True,
        num_attention_heads=55,
        num_key_value_heads=55,
        n_embd=5280,
        intermediate_size=17920,
        bias=False,
        architecture_class_name="RecurrentGPT",
        block_class_name="SandwichBlock",
        norm_class_name="UnparametrizedRMSNorm",
        norm_eps=1e-6,
        mlp_class_name="GatedMLP",
        nonlin_name="SiLU",
        init_strategy="scaled",
        init_orthogonal=True,
        state_init="like-init",
        # spell out the recurrent settings
        injection_type="add",
        n_layers_in_recurrent_block=4,
        mean_recurrence=32,
        sampling_scheme="poisson-lognormal-filling",
        mean_backprop_depth=8,
        n_layers_in_prelude=2,
        n_layers_in_coda=2,
        qk_bias=False,
        # 0: Total parameters: 3,508,961,280
        # 0: Model initialized with 1,581m parameters in recurrent block.
        # 0: Will unfold to 52,534m mean parameters at test time (32 rec).
        # 0: Could unfold to 103,141m parameters at test time (64 rec).
    ),
    dict(
        name="corone-raven-3.5b",  # not capable of flight :(
        hf_config=dict(org="tomg-group-umd", name="merseburg-raven-3.5b"),
        block_size=4096,
        vocab_size=65536,
        padding_multiple=4096,
        tie_embeddings=True,
        num_attention_heads=55,
        num_key_value_heads=55,
        n_embd=5280,
        intermediate_size=17920,
        bias=False,
        architecture_class_name="RecurrentGPT",
        block_class_name="TransformerPreNormBlock",
        norm_class_name="RMSNorm_llama",
        norm_eps=1e-6,
        mlp_class_name="GatedMLP",
        nonlin_name="SiLU",
        init_strategy="takase",
        init_orthogonal=False,  # we're buying for now that all of these vectors are probably orthogonal
        state_init="like-init",
        # spell out the recurrent settings
        injection_type="linear",
        n_layers_in_recurrent_block=4,
        mean_recurrence=32,
        sampling_scheme="poisson-lognormal-filling",
        mean_backprop_depth=8,
        n_layers_in_prelude=2,
        n_layers_in_coda=2,
        qk_bias=True,
        activation_checkpoint_impl="per-iteration",  # SAC broken
        #  Total parameters: 3,509,051,040
        #  Model initialized with 1,581m parameters in recurrent block.
        #  Will unfold to 52,535m mean parameters at test time (32 rec).
        #  Could unfold to 103,144m parameters at test time (64 rec).
    ),
    dict(
        name="nebel-raven-3.5b",
        hf_config=dict(org="tomg-group-umd", name="nebel-raven-3.5b"),
        block_size=4096,
        vocab_size=65536,
        padding_multiple=4096,
        tie_embeddings=True,
        num_attention_heads=55,
        num_key_value_heads=55,
        n_embd=5280,
        intermediate_size=17920,
        bias=False,
        architecture_class_name="RecurrentGPT",
        block_class_name="SandwichBlock",
        norm_class_name="RMSNorm_llama",
        norm_eps=0.000001,
        mlp_class_name="GatedMLP",
        nonlin_name="SiLU",
        init_strategy="takase",
        init_orthogonal=False,  # we're buying for now that all of these vectors are probably orthogonal
        state_init="like-init",
        # spell out the recurrent settings
        injection_type="linear",
        n_layers_in_recurrent_block=4,
        mean_recurrence=32,
        sampling_scheme="poisson-lognormal-filling",
        mean_backprop_depth=8,
        n_layers_in_prelude=2,
        n_layers_in_coda=2,
        qk_bias=True,
        activation_checkpoint_impl="per-iteration",  # SAC broken
        #  Total parameters: 3,509,051,040
        #  Model initialized with 1,581m parameters in recurrent block.
        #  Will unfold to 52,535m mean parameters at test time (32 rec).
        #  Could unfold to 103,144m parameters at test time (64 rec).
    ),
    # Shape notes:
    # The best outcome was 115.9 TFLOPS @ 4096x7040x14080 (MxNxK) (tried 1 shapes)
    # The best outcome was 114.7 TFLOPS @ 4096x6144x12288 (MxNxK) (tried 1 shapes)
    # best:                115.3 TFLOPS @ 4096x3520x12320 (MxNxK)
    # pow2 search:         120.5 TFLOPS @ 4096x7936x12288 (MxNxK) (tried 512 shapes)
    # 110 search:          119.1 TFLOPS @ 4096x8800x7920 (MxNxK))
    # The best outcome was 119.1 TFLOPS @ 4096x8800x7920 (MxNxK) (tried 5754 shapes)
    # refs: llama 3.1 70b is
    # 8192 x 28672 x 80 layers (with 64 x 8 heads)
    #       llama 3.2 3B is
    # 3072 x 8192 x 28 layers (with 24 x 8 heads)
    #                      118.8 TFLOPS @ 4096x5280x12320
    #                      115.5 TFLOPS @ 4096x3520x12320
    #                      119.1 TFLOPS @ 4096x5280x12320 (MxNxK)
    # The best outcome was 117.0 TFLOPS @ 4096x5120x17920 (MxNxK) (tried 1 shapes)
    # The best outcome was 115.3 TFLOPS @ 4096x6016x12304 (MxNxK) (tried 1176 shapes)
    #               best:  116.3 TFLOPS @ 4096x5376x18432 (MxNxK)
    #                      116.4 TFLOPS @ 4096x5472x18432
    # candidates:
    # - 5280 x 17920 -> mamf = 117.7 TFLOPS @ 4096x5280x35840 (MxNxK) (tried 1 shapes)
    # - 5120 x 17920 -> mamf = 117.9 TFLOPS @ 4096x5120x35840 (MxNxK) (tried 1 shapes)
    # - 5120 x 16384 -> mamf = 116.2 TFLOPS @ 4096x5120x32768 (MxNxK) (tried 1 shapes)
    # - 5120 x 17920 -> mamf = 118.0 TFLOPS @ 4096x5120x35840 (MxNxK) (tried 1 shapes) [rerun]
    # - 5280 x 18480 -> mamf = 115.7 TFLOPS @ 4096x5280x36960 (MxNxK) (tried 1 shapes)
    # - 7040 x 14080 -> mamf = 119.0 TFLOPS @ 4096x7040x28160 (MxNxK) (tried 1 shapes)
    # - 7168 x 14336 -> mamf = 118.3 TFLOPS @ 4096x7168x28672 (MxNxK) (tried 1 shapes)
    # - 7936 x 12288 -> mamf = 124.8 TFLOPS @ 4096x7936x24576 (MxNxK) (tried 1 shapes)
    # final candidates
    # 5120 x 17920 -> kinda slow overall
    # 7040 x 14080
]
configs.extend(rec)


###############
# Meta LLaMA 2
###############
llama_2 = [
    dict(
        name="Llama-debug",
        hf_config=dict(org="tomg-group-umd", name="Llama-debug"),
        architecture_class_name="GPT",
        vocab_size=32000,
        block_size=384,  # to find it again
        num_attention_heads=8,
        num_key_value_heads=2,
        n_embd=512,
        padding_multiple=4096,
        n_layer=4,
        bias=False,
        norm_class_name="RMSNorm",
        mlp_class_name="GatedMLP",
        nonlin_name="SiLU",
        intermediate_size=256,
    ),
]
for c in llama_2:
    for kind in ("", "-chat"):
        copy = deepcopy(c)
        copy["name"] = c["name"].format(kind)
        copy["hf_config"]["name"] = c["hf_config"]["name"].format(kind)
        configs.append(copy)


# ###############
# # Meta LLaMA 3
# ###############
# llama_3 = [
#     # https://huggingface.co/meta-llama/Meta-Llama-3-8B/blob/main/config.json
#     dict(
#         name="Llama-3-8B{}",
#         hf_config=dict(org="meta-llama", name="Meta-Llama-3-8B{}"),
#         block_size=8192,
#         vocab_size=128256,
#         padding_multiple=4096,
#         n_layer=32,
#         num_attention_heads=32,
#         num_key_value_heads=8,
#         rotary_percentage=1.0,
#         parallel_residual=False,
#         bias=False,
#         norm_class_name="RMSNorm",
#         mlp_class_name="LLaMAMLP",
#         intermediate_size=14336,
#         rope_base=500000,
#     ),
#     # https://huggingface.co/meta-llama/Meta-Llama-3-70B/blob/main/config.json
#     dict(
#         name="Llama-3-70B{}",
#         hf_config=dict(org="meta-llama", name="Meta-Llama-3-70B{}"),
#         block_size=8192,
#         vocab_size=128256,
#         padding_multiple=4096,
#         n_layer=80,
#         num_attention_heads=64,
#         n_embd=8192,
#         num_key_value_heads=8,
#         rotary_percentage=1.0,
#         parallel_residual=False,
#         bias=False,
#         norm_class_name="RMSNorm",
#         mlp_class_name="LLaMAMLP",
#         intermediate_size=28672,
#         rope_base=500000,
#     ),
# ]
# for c in llama_3:
#     for kind in ("", "-Instruct"):
#         copy = deepcopy(c)
#         copy["name"] = c["name"].format(kind)
#         copy["hf_config"]["name"] = c["hf_config"]["name"].format(kind)
#         configs.append(copy)


# ##################################
# # togethercomputer LLaMA-2-7B-32K
# ##################################
# together_llama2_32k = [
#     # https://huggingface.co/togethercomputer/LLaMA-2-7B-32K/blob/main/config.json
#     dict(
#         name="LLaMA-2-7B-32K",
#         hf_config=dict(org="togethercomputer", name="LLaMA-2-7B-32K"),
#         vocab_size=32000,
#         padding_multiple=4096,
#         n_layer=32,
#         rotary_percentage=1.0,
#         parallel_residual=False,
#         bias=False,
#         norm_class_name="RMSNorm",
#         mlp_class_name="LLaMAMLP",
#         intermediate_size=11008,
#         rope_condense_ratio=8,
#     )
# ]
# configs.extend(together_llama2_32k)


# ################
# # Microsoft Phi
# ################
# phi = [
#     # https://huggingface.co/microsoft/phi-1_5/blob/main/config.json
#     dict(
#         name="phi-1_5",
#         hf_config=dict(org="microsoft", name="phi-1_5"),
#         vocab_size=50257,
#         padded_vocab_size=51200,
#         block_size=2048,
#         n_embd=2048,
#         n_layer=24,
#         rotary_percentage=0.5,  # 32 / (n_embd / n_head) = 32 / 64
#         shared_attention_norm=True,
#         lm_head_bias=True,
#         gelu_approximate="tanh",
#     ),
#     # https://huggingface.co/microsoft/phi-2/blob/main/config.json
#     dict(
#         name="phi-2",
#         hf_config=dict(org="microsoft", name="phi-2"),
#         vocab_size=50257,
#         padded_vocab_size=51200,
#         block_size=2048,
#         n_embd=2560,
#         n_layer=32,
#         rotary_percentage=0.4,  # 32 / (n_embd / n_head) = 32 / 80
#         shared_attention_norm=True,
#         lm_head_bias=True,
#         gelu_approximate="tanh",
#     ),
# ]
# configs.extend(phi)


# ############
# # TinyLlama
# ############
tiny_llama = [
    dict(
        name="tiny-llama-190m",
        hf_config=dict(org="tomg-group-umd", name="tiny-llama-190m"),
        architecture_class_name="GPT",
        block_size=2048,
        vocab_size=32000,
        padding_multiple=4096,
        n_layer=12,
        num_attention_heads=32,
        num_key_value_heads=4,
        n_embd=1024,
        bias=False,
        norm_class_name="RMSNorm",  # original TinyLlama uses FusedRMSNorm
        norm_eps=1e-5,
        mlp_class_name="GatedMLP",
        nonlin_name="SiLU",
        intermediate_size=2816,
    ),
    dict(
        name="tiny-llama-1.1b{}",
        hf_config=dict(org="TinyLlama", name="TinyLlama-1.1B{}"),
        architecture_class_name="GPT",
        block_size=2048,
        vocab_size=32000,
        padding_multiple=4096,
        n_layer=22,
        num_attention_heads=32,
        num_key_value_heads=4,
        n_embd=2048,
        bias=False,
        norm_class_name="RMSNorm",  # original TinyLlama uses FusedRMSNorm
        norm_eps=1e-5,
        mlp_class_name="GatedMLP",
        nonlin_name="SiLU",
        intermediate_size=5632,
    ),
]
for c in tiny_llama:
    for kind, hf_postfix in (("", "-intermediate-step-1431k-3T"), ("-chat", "-Chat-v1.0")):
        copy = deepcopy(c)
        copy["name"] = c["name"].format(kind)
        copy["hf_config"]["name"] = c["hf_config"]["name"].format(hf_postfix)
        configs.append(copy)


######################
# Scaling Laws "Gemma"
######################
scaling_laws_gemma = [
    dict(
        name="PyGemma-2-2b",
        hf_config=dict(org="google", name="gemma-2-2b"),
        architecture_class_name="GPT",
        scale_embeddings=True,
        vocab_size=50304,
        block_size=8192,
        intermediate_size=2048 * 4,
        n_embd=2048,
        n_layer=26,
        num_attention_heads=16,
        num_key_value_heads=8,
        bias=False,
        norm_class_name="RMSNorm",
        mlp_class_name="GatedMLP",
        nonlin_name="GELU",
    ),
    dict(
        name="PyGemma-2-1b",
        hf_config=dict(org="google", name="gemma-2-2b"),
        architecture_class_name="GPT",
        scale_embeddings=True,
        vocab_size=50304,
        block_size=8192,
        intermediate_size=2048 * 4,
        n_embd=2048,
        n_layer=12,
        num_attention_heads=16,
        num_key_value_heads=8,
        bias=False,
        norm_class_name="RMSNorm",
        mlp_class_name="GatedMLP",
        nonlin_name="GELU",
    ),
    dict(
        name="PyGemma-2-500m",
        hf_config=dict(org="google", name="gemma-2-2b"),
        architecture_class_name="GPT",
        scale_embeddings=True,
        vocab_size=50304,
        block_size=8192,
        intermediate_size=1536 * 4,
        n_embd=1536,
        n_layer=12,
        num_attention_heads=12,
        num_key_value_heads=6,
        bias=False,
        norm_class_name="RMSNorm",
        mlp_class_name="GatedMLP",
        nonlin_name="GELU",
    ),
    dict(
        name="PyGemma-2-100m",
        hf_config=dict(org="google", name="gemma-2-2b"),
        architecture_class_name="GPT",
        scale_embeddings=True,
        vocab_size=50304,
        block_size=8192,
        intermediate_size=768 * 4,
        n_embd=768,
        n_layer=3,
        num_attention_heads=6,
        num_key_value_heads=3,
        bias=False,
        norm_class_name="RMSNorm",
        mlp_class_name="GatedMLP",
        nonlin_name="GELU",
    ),
]
configs.extend(scaling_laws_gemma)


name_to_config = {config["name"]: config for config in configs}
