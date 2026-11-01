import numpy as np
from typing import Set

def load_filtered_ids(filtered_id_path: str) -> Set[int]:
    with np.load(filtered_id_path, allow_pickle=False) as data:
        filtered_ids_set = set(data["filtered_ids"].tolist())
    return filtered_ids_set
