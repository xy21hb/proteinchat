"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

# import esm
import minigpt4.tasks as tasks
from minigpt4.esm.esm_config import Config
from minigpt4.common.dist_utils import get_rank, init_distributed_mode
from minigpt4.common.logger import setup_logger
from minigpt4.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from minigpt4.common.registry import registry
from minigpt4.common.utils import now

# imports modules for registration
from minigpt4.datasets.builders import *
from pdb_dataset_copy import ESMDataset
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
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
    print("1*******************")
    cfg = Config(parse_args())
    print("2*******************")
    init_distributed_mode(cfg.run_cfg)
    print("3*******************")

    setup_seeds(cfg)
    print("4*******************")
    # set after init_distributed_mode() to only log on master.
    setup_logger()
    print("5*******************")
    cfg.pretty_print()
    print("6*******************")
    task = tasks.setup_task(cfg)
    print("7*******************")

    # protein_encoder, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    # protein_encoder = protein_encoder.eval()
    # freeze_protein_encoder = True
    # if freeze_protein_encoder:
    #     for name, param in protein_encoder.named_parameters():
    #         param.requires_grad = False
    print('Loading protein_encoder Done')

    datasets_raw = ESMDataset(pdb_root="/home/h5guo/data/esm_subset/pt",
                              ann_paths="/home/h5guo/data/esm_subset/ann.json",
                              chain="A")
    # print(datasets_raw.__getitem__(0)["pdb_coords"].shape)
    # exit()
    datasets = {'esm': {'train': datasets_raw}}


    print("8*******************")
    model = task.build_model(cfg)

    print("9*******************")

    runner = get_runner_class(cfg)(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    )
    print("10*******************")
    runner.train()


if __name__ == "__main__":
    main()
