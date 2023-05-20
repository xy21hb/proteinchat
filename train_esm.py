"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import argparse
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

# import esm
import minigpt4.tasks as tasks
from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank, init_distributed_mode
from minigpt4.common.logger import setup_logger
from minigpt4.common.registry import registry
from minigpt4.common.utils import now

# imports modules for registration
# from pipeline.datasets.builders import *
from minigpt4.datasets.pdb_dataset import ESMDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--pdb-path", default="/home/h5guo/data/esm_subset/pt", help="path to protein embedding file.")
    parser.add_argument("--ann-path", default="/home/h5guo/data/esm_subset/ann.json", help="path to annotation file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    args = parser.parse_args()
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def get_runner_class(cfg):
    """
    Get runner class from config. Default to epoch-based runner.
    """
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base"))

    return runner_cls


def main():
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"

    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    job_id = now()
    args = parse_args()
    cfg = Config(args)
    init_distributed_mode(cfg.run_cfg)
    setup_seeds(cfg)
    # set after init_distributed_mode() to only log on master.
    setup_logger()
    cfg.pretty_print()
    task = tasks.setup_task(cfg)

    print('Loading protein_encoder Done')

    datasets_raw = ESMDataset(pdb_root=args.pdb_path,
                              ann_paths=args.ann_path,
                              chain="A")
    datasets = {'esm': {'train': datasets_raw}}


    model = task.build_model(cfg)


    runner = get_runner_class(cfg)(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    )
    runner.train()


if __name__ == "__main__":
    main()
