# Pretraining a Depth-Recurrent Model

This repo contains the code we used to train a recurrent-depth model at scale on 4096 AMD GPUs on Frontier. All details on this model can be found in the tech report: "Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach." The final model is `huginn-0125`, which can be found here: https://huggingface.co/tomg-group-umd/huginn-0125

This repo also contains all code to prepare the tokenizer and data, mostly in `scripts/`. 

I (Jonas) do not necessarily think that you should pretrain your own model with this implementation, but I hope it serves as a useful reference for the exact choices we took to run this model (at all), and how we ran this model given the limitations of AMD systems. If you are working with either of these, feel free to always raise an Issue asking for more details. 


## Code Setup:
*  The actual model definition is in `repre/model_dynamic.py`.
*  The training is orchestrated from `train.py`.
*  Model shapes can be found in `recpre/model_registry.py`. The final model is the shape `nebel-raven-3.5b`
*  The configurations for our two large-scale runs are in `launch_configs/`. 
*  The environment flags can be read out of `launch_frontier.py`.
* The parallelism implementation is deep down in `recpre/utils.py`, in a class called `SimpleFabric`. `_allreduce_chunk_stream` was used for inter-node communication, which was the only solution to remedy RCCL hangs at scale when using the OFI plugin, at the time of writing.

The code to run the model at inference is probablier easier to look at, if you just want to see the model architecture.
It can be found on all Huggingface repos of this model, and at `recpre/raven_modeling_minimal.py`.

