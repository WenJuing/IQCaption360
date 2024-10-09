import torch


class Config(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def IQCaption360_config():
    config = Config({
        # model setting
        'num_vps': 8,                       # number of viewports in a sequence.
        'img_channels': 3,
        'img_size': 224,
        'dim': 64,                          # dimension after Stem module.
        'depths': (2, 2, 5, 3),             # number of maxvit block in each stage.
        'channels': (128, 256, 512, 512),     # channels in each stage.
        'num_heads': (2, 4, 8, 16),          # number of head in each stage.
        
        'mlp_ratio': 3,
        'drop_rate': 0.,
        'pos_drop_rate': 0.,
        'attn_drop_rate': 0.,
        'drop_path_rate': 0.,              # droppath rate in encoder block.
        'kernel_size': 7,
        'layer_scale': None,
        'dilations': None,
        'qkv_bias': True,
        'qk_scale': None,
        'select_rate': 0.5,                 # the rate of select feature from all viewport features.
        'num_classes': 4,
        'hidden_dim': 1152,                   
        
        
        # resource setting
        'vp_path': '/media/xxxy/Elements/datasets/OIQ-10K/viewports_8',
        'train_info_csv_path': '/home/xxxy/tzw/databases/OIQ-10K/OIQ-10K_train_info.csv',
        'test_info_csv_path': '/home/xxxy/tzw/databases/OIQ-10K/OIQ-10K_test_info.csv',
        'save_ckpt_path': '',
        'load_ckpt_path': '',
        'tensorboard_path': '',
        # train setting
        'seed': 42,
        'model_name': 'IQCaption360-AFA-MSFS-VPFS-DRPN-QSPN-lr1e-4-bs32-epoch50',
        'dataset_name': 'OIQ-10K',
        'epochs': 50,
        'batch_size': 32,
        'num_workers': 8,
        'lr': 1e-4,
        'lrf': 0.01,
        'weight_decay': 5e-4,
        'momentum': 0.9,
        'p': 1,
        'q': 2,
        'use_tqdm': True,
        'use_tensorboard': False,
        'batch_print': False,
        'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    })  
        
    return config