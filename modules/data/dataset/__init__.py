# from modules.data.dataset.registration_dataset import BasicRegistrationDataset
# from modules.data.dataset.LMA_dataset import LMADataset
# from modules.data.dataset.strainmat_dataset import StrainMatDataset
# from modules.data.dataset.joint_dataset import JointDataset
from modules.data.dataset.slice_dataset import SliceDataset
from modules.data.dataset.reg_pair_dataset import RegPairDataset
from modules.data.dataset.reg_vol_pair_dataset import RegVolPairDataset
from modules.data.dataset.video_volume_dataset import VideoVolDataset
def build_datasets(datasets_configs, data_splits, all_config=None):
    datasets = {}
    for dataset_name, dataset_config in datasets_configs.items():
        data_split = data_splits[dataset_name]        
        if dataset_config['type'] == 'SliceDataset':
            datasets[dataset_name] = SliceDataset(
                data_split['data'], 
                config=dataset_config,
                full_config=all_config, 
                dataset_name=dataset_name)
        elif dataset_config['type'] == 'RegPairDataset':
            datasets[dataset_name] = RegPairDataset(
                data_split['data'], 
                config=dataset_config,
                full_config=all_config, 
                dataset_name=dataset_name)
        elif dataset_config['type'] == 'RegVolPairDataset':
            datasets[dataset_name] = RegVolPairDataset(
                data_split['data'], 
                config=dataset_config,
                full_config=all_config, 
                dataset_name=dataset_name)
        elif dataset_config['type'] == 'VideoVolDataset':
            datasets[dataset_name] = VideoVolDataset(
                data_split['data'], 
                config=dataset_config,
                full_config=all_config, 
                dataset_name=dataset_name)
        else:
            raise ValueError(f'Unknown dataset type: {dataset_config["type"]}')
    return datasets