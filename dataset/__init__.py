

from .reconf_group import ReconFusionGroupDataset
from .reconf_gen import ReconFusionDirectGenDataset
from .custom_cam import StaticCamSingleDataset, StaticCamMultiDataset
from .gso import GSOSingleDataset, GSOMultiDataset
# from .c4d import C4DSingleDataset


dataset_func = {
    # reconfusion format dataset
    "reconfgroup": ReconFusionGroupDataset,
    "reconfdirectgen": ReconFusionDirectGenDataset,
    
    # obj centric generation
    "static_cam_single": StaticCamSingleDataset,
    "static_cam_multi": StaticCamMultiDataset,
    
    ## gso dataset
    "gso-single": GSOSingleDataset,
    "gso-multi": GSOMultiDataset,
}

def create_dataset(dataset_name: str, **kwargs):
    try:
        return dataset_func[dataset_name](**kwargs)
    except KeyError:
        raise ValueError(f"Unknown dataset {dataset_name}")

    