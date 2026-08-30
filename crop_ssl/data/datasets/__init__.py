"""
CropSSL Datasets.

All supported plant disease detection datasets for
cross-domain robustness evaluation.
"""

from crop_ssl.data.datasets.plantvillage import PlantVillageDataset
from crop_ssl.data.datasets.plantdoc import PlantDocDataset
from crop_ssl.data.datasets.rice_leaf import RiceLeafDataset
from crop_ssl.data.datasets.coffee_leaf import CoffeeLeafDataset
from crop_ssl.data.datasets.domainnet_plant import DomainNetPlant
from crop_ssl.data.datasets.new_plant_diseases import NewPlantDiseasesDataset
from crop_ssl.data.datasets.cassava_leaf import CassavaLeafDataset
from crop_ssl.data.datasets.plant_pathology import PlantPathologyDataset
from crop_ssl.data.datasets.icassava_2019 import ICassava2019Dataset
from crop_ssl.data.datasets.plant_seg import PlantSegDataset
from crop_ssl.data.datasets.field_plant import FieldPlantDataset
from crop_ssl.data.datasets.diamos_plant import DiaMOSPlantDataset
from crop_ssl.data.datasets.bracol import BRACOLDataset
from crop_ssl.data.datasets.few_shot_sampler import (
    FewShotSampler,
    BalancedClassSampler,
    DomainStratifiedSampler,
)
from crop_ssl.data.datasets.cross_domain_dataset import (
    CrossDomainDataset,
    DATASET_REGISTRY,
)

__all__ = [
    "PlantVillageDataset",
    "PlantDocDataset",
    "RiceLeafDataset",
    "CoffeeLeafDataset",
    "DomainNetPlant",
    "NewPlantDiseasesDataset",
    "CassavaLeafDataset",
    "PlantPathologyDataset",
    "ICassava2019Dataset",
    "PlantSegDataset",
    "FieldPlantDataset",
    "DiaMOSPlantDataset",
    "BRACOLDataset",
    "FewShotSampler",
    "BalancedClassSampler",
    "DomainStratifiedSampler",
    "CrossDomainDataset",
    "DATASET_REGISTRY",
]
