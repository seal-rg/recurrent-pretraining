# vLLM plugin - experimental!

Install the plugin by moving into this folder and calling
``` uv pip install -e .```
(or whatever the equivalent build command is in your python setup)
You will then be able to serve the model with something along the lines of

```
vllm serve tomg-group-umd/huginn-0125   --trust-remote-code   --dtype bfloat16   --tensor-parallel-size 1   --max-model-len 4096   --gpu-memory-utilization 0.7
```

If you don't want to wait for the graph capture and compilation, you can also enforce eager.


## lm-eval Integration

The main usecase for this to to enable faster evals. A more involved `lm-eval` example is
```
lm_eval --model vllm   --model_args 'pretrained=tomg-group-umd/huginn-0125,trust_remote_code=True,dtype=bfloat16,tensor_parallel_size=1,dtype=bfloat16,gpu_memory_utilization=0.8,data_parallel_size=1,hf_overrides={"mean_recurrence" : 32}' \
  --tasks gsm8k_cot   --batch_size auto   --num_fewshot 8   --output_path outputs/vllm   --apply_chat_template   \
  --system_instruction "You are a helpful assistant that can assist users with mathematical reasoning." \
  --fewshot_as_multiturn
```
This will run GSM8k CoT via lm-eval through the vllm backend. You can easily increase the `data_parallel_size` to run on more workers, if available. Arguments given to the model like `mean_recurrence` can be modified as shown.


## Rough edges
* Need to make it easier to pass gen_kwargs
* Everything token-level adaptive is unimplemented

# Dev

To debug the vllm implementation, run, which will run in eager by default.
```python raven_vllm.py```
But, make sure to test both implementation with `init_scale=0.0` and with the same number of recurrent steps.
