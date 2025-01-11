import pdb

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
