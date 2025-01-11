import os
from typing import List, Tuple

def compute_entities_and_rel_dict(data_path):
    """
    Compute the dictionary of entities and relations.
    Will compute from triplet files rather than from .dict files
    Honestly I am not so sure about this function anymore
    :param data_path: the path of the data file.
    :return: two dictionaries, one for entities and one for relations.
    """
    entity2id = {}
    relation2id = {}
    all_splits = ["train", "dev", "test"]
    all_files = [data_path + f"/{split}.txt" for split in all_splits]
    for f in all_files:
        with open(f) as fin:
            for line in fin:
                head, relation, tail = line.strip().split('\t')
                if head not in entity2id:
                    entity2id[head] = len(entity2id)
                if relation not in relation2id:
                    relation2id[relation] = len(relation2id)
                if tail not in entity2id:
                    entity2id[tail] = len(entity2id)
                    
    return entity2id, relation2id

def read_triple(file_path, entity2id, relation2id) -> List[Tuple[int, int, int]]:
    '''
    Read triples and map them into ids.
    '''
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples

def get_triplets(data_path: str, triple_filename: str = "test.txt"):
    # Tokens Path
    entities_path = os.path.join(data_path, 'entities.dict')
    relations_path = os.path.join(data_path, 'relations.dict')
    # We create our own entities.dict and relations.dict if they don't exist
    if not os.path.exists(entities_path) or not os.path.exists(relations_path):
        # Use Yellow Escape Codes
        print("\033[1;33m entities.dict and relations.dict not found in data_path. Will continue to generate myself\033[0m")
        entity2id, relation2id = compute_entities_and_rel_dict(data_path)
        with open(entities_path, "w") as fout:
            for entity, eid in entity2id.items():
                fout.write(f"{eid}\t{entity}\n")
        with open(relations_path, "w") as fout:
            for relation, rid in relation2id.items():
                fout.write(f"{rid}\t{relation}\n")
        print(
            f"Got {len(entity2id)} entities and {len(relation2id)} relations"
            f"Will proceed to save them to {entities_path} and {relations_path}"
        )

    with open(entities_path) as fin:
        entity2id = dict()
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)

    with open(relations_path) as fin:
        relation2id = dict()
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)

    test_triples = read_triple(os.path.join(data_path, triple_filename), entity2id, relation2id)

    return test_triples
