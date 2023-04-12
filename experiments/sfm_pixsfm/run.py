import sys
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

from hloc import (
    extract_features,
    match_features,
    pairs_from_exhaustive
)
from pixsfm.refine_hloc import PixSfM
from pixsfm.configs import parse_config_path, default_configs


BASE_DIR = sys.argv[1]
BASE_DIR = Path(BASE_DIR)

path_images = BASE_DIR / 'images'
path_keypoints = BASE_DIR / 'keypoints.h5'
path_pairs = BASE_DIR / 'pairs.txt'
path_matches = BASE_DIR / 'matches.h5'
path_sfm = BASE_DIR / 'sfm/'
path_cache = BASE_DIR / f'dense_features_cache.h5'

keypoints_conf = extract_features.confs["r2d2"]
matcher_conf = match_features.confs["NN-superpoint"]

keypoints_conf["model"]["max_keypoints"] = 30000

extract_features.main(
    keypoints_conf,
    path_images,
    as_half=False,
    feature_path=path_keypoints
)


def pairs_from_sequential(
        output: Path,
        image_list: Path = None,
        num_of_pairs_per_frame: int = 3
):
    pairs = []
    path_images_list = list(image_list.iterdir())
    for ind, frame in enumerate(path_images_list):
        for i in range(1, num_of_pairs_per_frame + 1):
            try:
                pairs.append((
                    frame.name,
                    path_images_list[ind + i].name
                ))
            except IndexError as e:
                continue
    with open(output, 'w') as f:
        f.write('\n'.join(' '.join([i, j]) for i, j in pairs))


pairs_from_sequential(
    output=path_pairs,
    image_list=path_images,
)
# pairs_from_exhaustive.main(
#     output=path_pairs,
#     features=path_keypoints,
# )

match_features.main(
    conf=matcher_conf,
    pairs=path_pairs,
    features=path_keypoints,
    matches=path_matches
)


conf = OmegaConf.load(parse_config_path("default"))
conf["mapping"]["KA"]["apply"] = False

refiner = PixSfM(conf)
reconstruction, sfm_outputs = refiner.reconstruction(
    output_dir=path_sfm,
    image_dir=path_images,
    pairs_path=path_pairs,
    features_path=path_keypoints,
    matches_path=path_matches,
    cache_path=path_cache,
)
