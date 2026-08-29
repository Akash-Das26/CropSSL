"""Data loading and preprocessing pipeline for crop disease datasets."""

from crop_ssl.data.datasets.plantvillage import PlantVillageDataset
from crop_ssl.data.datasets.plantdoc import PlantDocDataset
from crop_ssl.data.datasets.rice_leaf import RiceLeafDataset
from crop_ssl.data.datasets.coffee_leaf import CoffeeLeafDataset
from crop_ssl.data.datasets.domainnet_plant import DomainNetPlant
from crop_ssl.data.datasets.new_plant_diseases import NewPlantDiseasesDataset
from crop_ssl.data.datasets.cassava_leaf import CassavaLeafDataset
from crop_ssl.data.datasets.plant_pathology import PlantPathologyDataset
from crop_ssl.data.datasets.icassava_2019 import ICassava2019Dataset
from crop_ssl.data.datasets.few_shot_sampler import FewShotSampler
from crop_ssl.data.datasets.cross_domain_dataset import CrossDomainDataset

DATASET_REGISTRY = {
    "plantvillage": PlantVillageDataset,
    "plantdoc": PlantDocDataset,
    "rice_leaf": RiceLeafDataset,
    "coffee_leaf": CoffeeLeafDataset,
    "domainnet_plant": DomainNetPlant,
    "new_plant_diseases": NewPlantDiseasesDataset,
    "cassava_leaf": CassavaLeafDataset,
    "plant_pathology": PlantPathologyDataset,
    "icassava_2019": ICassava2019Dataset,
}

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
    "FewShotSampler",
    "CrossDomainDataset",
    "DATASET_REGISTRY",
]
