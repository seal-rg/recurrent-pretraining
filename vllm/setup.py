from setuptools import setup

setup(
    name="vllm_raven_plugin",
    version="0.1",
    py_modules=["raven_vllm"],
    entry_points={"vllm.general_plugins": ["register_raven = raven_vllm:register"]},
)
