import os
import subprocess
import sys

from llamafactory.train.tuner import run_exp

from llamafactory.extras.misc import get_device_count
from llamafactory import launcher
import yaml

master_addr = "127.0.0.1"
master_port = "20202"
args = "llama3_lora_sft_ds3.yaml"


file = "/data/home/frx/cultural_llm/LLaMA-Factory/src/llamafactory/launcher.py"
force_torchrun = os.environ.get("FORCE_TORCHRUN", "0").lower() in ["true", "1"]
if force_torchrun or get_device_count() > 1:
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_port = os.environ.get("MASTER_PORT", str(20202))
    # print(os.environ.get("NNODES", "1"))
    # print(os.environ.get("RANK", "0"))
    # print(os.environ.get("NPROC_PER_NODE", str(get_device_count())))
    # print(master_addr)
    # print(master_port)
    # print(launcher.__file__)
    # print(" ".join(sys.argv[1:]))
    process = subprocess.run(
        (
            "torchrun --nnodes {nnodes} --node_rank {node_rank} --nproc_per_node {nproc_per_node} "
            "--master_addr {master_addr} --master_port {master_port} {file_name} {args}"
        ).format(
            nnodes=os.environ.get("NNODES", "1"),
            node_rank=os.environ.get("RANK", "0"),
            nproc_per_node=os.environ.get("NPROC_PER_NODE", str(get_device_count())),
            master_addr=master_addr,
            master_port=master_port,
            file_name=launcher.__file__,
            args=" ".join(sys.argv[1:]),
        ),
        shell=True,
    )
    sys.exit(process.returncode)
else:
    with open("qwen2vl_lora_dpo.yaml", "r") as file:
        args = yaml.safe_load(file)
    run_exp(args=args)