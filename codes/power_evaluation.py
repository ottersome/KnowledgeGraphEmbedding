import argparse
import glob
import json
import os
import re
from typing import Dict, Tuple

import numpy as np

from model import KGEModel, test_step_explicitArgs
from mods.logging import setup_logger

import debugpy
from data_utils import get_triplets
import matplotlib.pyplot as plt
import wandb
import torch


def argsies() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_path_regex", default="../models/RotatE_FB15k_[0-9][0-9]*")
    ap.add_argument("--data_path", default="../data/FB15k")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--metrics_path", default="../metrics")

    ap.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Whether or not to use debugpy attachment.",
    )
    ap.add_argument(
        "--dport", "-p", default=42020, type=int, help="The port to attach debugpy to."
    )

    args = ap.parse_args()

    return args


def load_model(trained_model_path: str, device: str) -> Tuple[KGEModel, Dict]:
    logger.info(f"Loading model from {trained_model_path}")
    config_path = os.path.join(trained_model_path, "config.json")
    config = json.load(open(config_path))

    kge_model = KGEModel(
        model_name=config["model"],
        nentity=config["nentity"],
        nrelation=config["nrelation"],
        hidden_dim=config["hidden_dim"],
        gamma=config["gamma"],
        double_entity_embedding=config["double_entity_embedding"],
        double_relation_embedding=config["double_relation_embedding"],
    )

    # Now we load the checkpointn
    print("Checking : " + trained_model_path)
    checkpoint = torch.load(os.path.join(trained_model_path , "checkpoint"))

    entity_embeddings = np.load(
        os.path.join(trained_model_path, "entity_embedding.npy")
    )
    relation_embeddings = np.load(
        os.path.join(trained_model_path, "relation_embedding.npy")
    )
    kge_model.load_embeddings(entity_embeddings, relation_embeddings)

    # Load the state dict
    kge_model.load_state_dict(checkpoint["model_state_dict"])
    kge_model.to(device)

    # Restore other saved variables (for reference I guess)
    save_variables = {k: v for k,v in checkpoint.items() if k not in ["model_state_dict", "optimizer_state_dict"]}

    logger.info(f"Model loaded to {device}")

    return kge_model, config


def main(args: argparse.Namespace):
    global logger

    # Check if our working directory is repo/codes, if not warn and exit
    current_dir = os.getcwd()
    parent_dir = os.path.basename(current_dir)
    if parent_dir != "codes":
        logger.warning(
            f"You are not in the codes directory, but rather in {os.getcwd()} with parent dir {parent_dir}. Exiting..."
        )
        exit(1)

    logger.info("Starting the evaluation of these things")

    # Get all directories matching the regex pattern
    directories = glob.glob(args.models_path_regex)
    sorted_paths = sorted(directories, key=lambda x: int(re.search(r"\d+$", x).group()))  # type: ignore

    train_triples = get_triplets(args.data_path, "train.txt")
    valid_triples = get_triplets(args.data_path, "valid.txt")
    test_triples = get_triplets(args.data_path, "test.txt")

    all_true_triples = train_triples + valid_triples + test_triples

    ########################################
    # Ensure we can dump metrics to disk
    ########################################
    os.makedirs(args.metrics_path, exist_ok=True)

    overall_metrics: Dict[str, Dict[str, float]] = {}

    sorted_paths_str = "\n\t-".join(sorted_paths)
    logger.info(f"Will evaluate the follwing paths: \n\t-{sorted_paths_str}")

    for path in sorted_paths:
        logger.info(f"Evaluating {path}")
        kge_model, model_config = load_model(path, args.device)

        logger.info(f"About to test model {path}")
        metrics = test_step_explicitArgs(
            kge_model,
            test_triples,
            all_true_triples,
            model_config["countries"],
            model_config["regions"],
            model_config["cuda"],
            model_config["cpu_num"],
            model_config["test_batch_size"],
            model_config["nentity"],
            model_config["nrelation"],
            model_config["test_log_steps"],
            logger,
        )

        path_basename = os.path.basename(path)
        overall_metrics[path_basename] = metrics

    # Dump metrics to disk
    with open(os.path.join(args.metrics_path, "raw_metrics.json"), "w") as fout:
        json.dump(overall_metrics, fout)

    # Now we can compute call the graphing
    graph_metrics(overall_metrics, args.metrics_path)


def graph_metrics(metrics: Dict[str, Dict[str, float]], figure_save_path: str):
    """
    Will graph bar plots for each of the metrics:
        MRR, MR, HITS@1, HITS@3, HITS@10
        Which are the first key in the metrics dictionary
    """

    # Invert order of keys, first will be metric, second will be model
    metric_keys = list(metrics.values())[0].keys()
    new_metrics: Dict[str, Dict[str, float]] = {}

    for metric_key in metric_keys:
        new_metrics[metric_key] = {}
        for model, value in metrics.items():
            new_metrics[metric_key][model] = value[metric_key]

    # Create bar figure for each metric_key, each bar will be a model
    for i, metric_key in enumerate(metric_keys):
        plt.figure(i)
        plt.bar(
            list(new_metrics[metric_key].keys()), list(new_metrics[metric_key].values())
        )
        plt.title(metric_key)
        plt.xlabel("Model")
        plt.ylabel(metric_key)
        plt.xticks(rotation=90)
        save_path = os.path.join(figure_save_path, f"{metric_key}.png")
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Saved graph to {figure_save_path}")


if __name__ == "__main__":
    args = argsies()
    logger = setup_logger("__MAIN__")

    # Add debugpy
    if args.debug:
        logger.info("Attaching debugpy to port %d" % args.dport)
        debugpy.listen(args.dport)
        debugpy.wait_for_client()
        logger.info("Debugpy attached")

    main(args)
