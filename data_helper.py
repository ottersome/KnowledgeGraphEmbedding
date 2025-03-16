import os
from typing import Optional

def split_file(input_file, split_character: str, train_ratio=0.8, dev_ratio=0.1, test_ratio=0.1):
    try:
        # Read all lines
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        total_lines = len(lines)
        
        # Calculate split points
        train_end = int(total_lines * train_ratio)
        dev_end = train_end + int(total_lines * dev_ratio)
        
        # Split data
        train_data = lines[:train_end]
        dev_data = lines[train_end:dev_end]
        test_data = lines[dev_end:]
        
        # Write splits to files
        base_name = input_file.rsplit(split_character, 1)[0]
        
        with open(f"{base_name}_train.txt", 'w', encoding='utf-8') as f:
            f.writelines(train_data)
            
        with open(f"{base_name}_train.txt", 'w', encoding='utf-8') as f:
            f.writelines(dev_data)
            
        with open(f"{base_name}_test.txt", 'w', encoding='utf-8') as f:
            f.writelines(test_data)
            
        print(f"Split complete: Train={len(train_data)}, Dev={len(dev_data)}, Test={len(test_data)} lines")
        
    except Exception as e:
        print(f"Error: {str(e)}")

def fb_wiki_v2(
    triplet_file: str,
    save_location: str,
    randomize: bool,
    train_perc: float,
    valid_perc: float,
):

    with open(triplet_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    total_lines = len(lines)

    # Permute them lines
    if randomize:
        import random
        random.shuffle(lines)

    # Calculate split points
    train_amnt = int(total_lines * train_perc)
    valid_amnt = int(total_lines * valid_perc)

    # Split data
    train_data = lines[:train_amnt]
    valid_data = lines[train_amnt: valid_amnt + train_amnt]
    test_data = lines[train_amnt + valid_amnt:]

    # Save into the location
    train_save_location = os.path.join(save_location, "train.txt")
    valid_save_location = os.path.join(save_location, "train.txt")
    test_save_location = os.path.join(save_location, "test.txt")
    if not os.path.exists(save_location):
        os.makedirs(save_location)

    with open(train_save_location, "w", encoding="utf-8") as f:
        f.writelines(train_data)

    with open(valid_save_location, "w", encoding="utf-8") as f:
        f.writelines(valid_data)

    with open(test_save_location, "w", encoding="utf-8") as f:
        f.writelines(test_data)

    print(f"Split complete: Train={len(train_data)}, valid={len(valid_data)}, Test={len(test_data)} lines")

    return train_data, valid_data, test_data

def test_counting_index(entities_path: str, relations_path: str) -> None:
    entities_lines = open(entities_path, "r").readlines()
    relations_lines = open(relations_path, "r").readlines()

    entity_count = len(entities_lines)
    relation_count = len(relations_lines)

    from tqdm import tqdm

    max_line = max(entity_count, relation_count)
    for i in tqdm(range(max_line)):
        entity_line: Optional[str] = entities_lines[i] if i < entity_count else ""
        relation_line: Optional[str] = relations_lines[i] if i < relation_count else ""

        try:
            ent_weird_idx, ent_mid = entity_line.strip().split(None) if entity_line else ("", "")
        except ValueError:
            print(f"Weird entity line (line_n: {i}):\n\033[1;31m{entity_line}\033[0m")
            exit()
        rel_weird_idx, rel_mid = relation_line.strip().split(None) if relation_line else ("", "")

        if ent_weird_idx != "" and int(ent_weird_idx) != i:
            print(f"Weird entity index {ent_weird_idx} at line {i}")
            exit()
        if rel_weird_idx != "" and int(rel_weird_idx) != i:
            print(f"Weird relation index {rel_weird_idx} at line {i}")
            exit()

    return None

# Example usage
if __name__ == "__main__":

    # raw.kb
    # split_file("./raw.kb", ".")

    # FBWikiV2
    # fb_wiki_v2(
    #     "./data/FBWikiV2/triplet_filt_fb_wiki_alt.txt",
    #     save_location="./data/FBWikiV2",
    #     randomize=True,
    #     train_perc=0.8,
    #     valid_perc=0.1,
    # )

    # Check for weird indexes
    test_counting_index(
        "./data/FB15k/entities.dict",
        "./data/FB15k/relations.dict"
    )
