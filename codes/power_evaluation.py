import argparse
import glob
import json
import os
import re
from typing import Dict, Tuple

import numpy as np

from model import KGEModel, test_step_explicitArgs
from mods.logging import setup_logger

from data_utils import get_triplets, read_triple


def argsies() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_path_regex", default="../models/RotatE_FB15k_[0-9][0-9]*")
    ap.add_argument("--data_path", default="../data/FB15k")
    ap.add_argument("--device", default="cuda:0")

    args = ap.parse_args()

    return args




def load_model(trained_model_path: str, device: str) -> Tuple[KGEModel, Dict] :
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
        double_relation_embedding=config["double_relation_embedding"]
    )


    entity_embeddings = np.load(os.path.join(trained_model_path, "entity_embedding.npy"))
    relation_embeddings = np.load(os.path.join(trained_model_path, "relation_embedding.npy"))
    kge_model.load_embeddings(entity_embeddings, relation_embeddings)

    kge_model.to(device)

    logger.info(f"Model loaded to {device}")

    return kge_model, config

def main(args: argparse.Namespace):
    global logger

    logger.info("Starting the evaluation of these things")

    # Get all directories matching the regex pattern
    directories = glob.glob(args.models_path_regex)
    sorted_paths = sorted(directories, key=lambda x: int(re.search(r'\d+$', x).group())) # type: ignore

    train_triples = get_triplets(args.data_path, "train.txt")
    valid_triples = get_triplets(args.data_path, "valid.txt")
    test_triples = get_triplets(args.data_path, "test.txt")

    all_true_triples = train_triples + valid_triples + test_triples

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
            logger
        )





if __name__ == "__main__":
    args = argsies()
    logger = setup_logger("__MAIN__")
    main(args)
