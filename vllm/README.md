# vLLM plugin - experimental!

Install the plugin by moving into this folder and calling
``` uv pip install -e .```
(or whatever the equivalent build command is in your python setup)
You will then be able to serve the model with something along the lines of

```
vllm serve tomg-group-umd/huginn-0125   --trust-remote-code   --dtype bfloat16   --tensor-parallel-size 1   --max-model-len 4096   --gpu-memory-utilization 0.7
```

If you don't want to wait for the graph capture and compilation, you can also enforce eager.



Rough edges:
* Need to make it easier to pass gen_kwargs
* Everything token-level adaptive is unimplemented
* Does `--hf-overrides` work for `mean_recurrence`?

# Dev

To debug the vllm implementation, run, which will run in eager by default.
```python raven_vllm.py```
But, make sure to test both implementation with `init_scale=0.0` and with the same number of recurrent steps.