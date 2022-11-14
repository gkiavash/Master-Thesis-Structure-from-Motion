import sys
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import torch

import pycolmap

from pixsfm.refine_colmap import PixSfM
from pixsfm.configs import parse_config_path, default_configs
from pixsfm.util.database import COLMAPDatabase, pair_id_to_image_ids


torch.cuda.empty_cache()


def pairs_from_db(pairs_path: Path, database_path: Path):
    db = COLMAPDatabase.connect(str(database_path))
    pair_ids = db.execute("SELECT pair_id FROM matches").fetchall()
    pairs = [pair_id_to_image_ids(pids[0]) for pids in pair_ids]
    image_id_to_name = db.image_id_to_name()
    pairs = [(image_id_to_name[id1], image_id_to_name[id2])
             for id1, id2 in pairs]
    with open(pairs_path, "w") as doc:
        [doc.write(" ".join(pair) + "\n") for pair in pairs]
    db.close()


# BASE_PATH = '/home/gkiavash/Downloads/sfm_street_1'
# tag = 't1'

BASE_PATH = sys.argv[1]
tag = sys.argv[2]
config = sys.argv[3]

dataset = Path(BASE_PATH)
outputs = dataset / 'refined/'
outputs.mkdir(parents=True, exist_ok=True)

# Setup the paths
images = dataset / 'images/'
sift_sfm_dir = dataset / 'sparse/0/'
sift_database_path = dataset / "database.db"

sfm_dir = outputs / f'sfm_{tag}'
database_path = sfm_dir / f'database_refined_{tag}.db'
pairs_path = sfm_dir / "pairs.txt"
cache = sfm_dir / f'dense_features_{tag}.h5'
sfm_dir_colmap = sfm_dir / "colmap"

sfm_dir.mkdir(parents=True, exist_ok=True)
sfm_dir_colmap.mkdir(parents=True, exist_ok=True)

if config == "low":
    conf = OmegaConf.load(parse_config_path("low_memory"))
    print(type(conf), conf)
    conf['dense_features']['model'] = {"name": 'dsift'}
    # conf['dense_features']['device'] = "cpu"
    print(type(conf), conf)
else:
    conf = OmegaConf.load(parse_config_path("default"))

refiner = PixSfM(conf=conf)
# refiner = PixSfM(config)

# Refine keypoints in database
keypoints, ka_data, feature_manager = refiner.refine_keypoints_from_db(
    database_path,
    sift_database_path,
    images,
    cache_path=cache
)

pairs_from_db(pairs_path, database_path)
pycolmap.verify_matches(database_path, pairs_path)

# triangulate new points with poses from original model
reference_model = pycolmap.Reconstruction(sift_sfm_dir)
reconstruction = pycolmap.triangulate_points(reference_model, database_path, images, sfm_dir_colmap)

# Refine the resulting reconstruction
refiner.run_ba(
    reconstruction,
    images,
    cache_path=cache,
    # feature_manager=feature_manager
)

reconstruction.write(sfm_dir)
print('done')
