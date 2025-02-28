import argparse
import json
import os
from typing import Dict, Tuple

import debugpy
import numpy as np
from matplotlib import pyplot as plt
import torch
from data_utils import get_triplets
from model import KGEModel
from mods.logging import setup_logger

import wandb


def argsies() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="../models/RotatE_FB15k_1000")
    ap.add_argument("--data_path", default="../data/FB15k")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--figures_path", default="../figures/histograms")

    ap.add_argument(
        "--normalize",
        action="store_true",
        help="Whether to normalize it before showing the histogram",
    )

    ap.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Whether or not to use debugpy attachment.",
    )

    return ap.parse_args()


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
    checkpoint = torch.load(os.path.join(trained_model_path, "checkpoint"))

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
    save_variables = {
        k: v
        for k, v in checkpoint.items()
        if k not in ["model_state_dict", "optimizer_state_dict"]
    }

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

    # Get all directories matching the regex pattern
    path = args.model_path
    os.makedirs(args.figures_path, exist_ok=True)

    kge_model, model_config = load_model(path, args.device)

    ########################################
    # Actual evaluation of histograms
    ########################################
    embeddings_to_eval = {
        "entity": kge_model.entity_embedding.detach().cpu().numpy(),
        # "relation": kge_model.relation_embedding.detach().cpu().numpy(),
        # Code below does not account for relation, you woudl have to do that yourself
    }
    
    for emb_name, emb in embeddings_to_eval.items():
        if args.normalize:
            emb = np.linalg.norm(emb, axis=1)
        
        embeddings_num_dim = emb.shape[1] // 2
        assert emb.shape[1] % 2 == 0, "Embedding shape is not divisible by 2"
        print(f"Embedding shape is {emb.shape}")

        complex_num = emb[:, :1000] + 1j * emb[:, 1000:]
        magnitudes = np.abs(complex_num)

        for i in range(embeddings_num_dim):
            # Dump each dimension to a separate file

            plt.figure(i, figsize=(10, 10))
            plt.tight_layout()
            plt.hist(magnitudes[:, i], bins=100)
            plt.title(f"{emb_name} dimension {i}")
            plt.xlabel(f"{emb_name} dimension {i}")
            plt.ylabel("Count")
            plt.xticks(rotation=90)
            save_path = os.path.join(args.figures_path, f"{emb_name}_{i}.png")
            plt.savefig(save_path)
            plt.close()
            logger.info(f"Saved graph to {save_path}")

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
