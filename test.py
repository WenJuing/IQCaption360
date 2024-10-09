import torch
from IQCaption360 import create_model
from utils import mean_squared_error, logistic_func, fit_function, set_seed
from torch.utils.data import DataLoader
from my_dataset import MyDataset
from config import IQCaption360_config
from torchvision import transforms as transforms
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm


def main(cfg):
    set_seed(cfg)
    # create model
    print("*****begin test*******************************************************")
    model = create_model(pretrained=False, cfg=cfg).to(cfg.device)
    checkpoint = torch.load("/home/xxxy/tzw/IQCaption360/ckpt/IQCaption360/ablation/IQCaption360-OIQ10K-AFA-MSFS-VPFS_1d2-DRPN-QSPN-lr1e-4-bs32-epoch50/best_epoch_35.pth", map_location=cfg.device)
    print(model.load_state_dict(checkpoint['model_state_dict'], strict=False))
    print("weights had been load!\n")

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), 
    ])

    test_dataset = MyDataset(cfg=cfg, info_csv_path=cfg.test_info_csv_path, transform=test_transform)
    print(len(test_dataset), "test data has been load!")
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=False,
    )

    # test
    model.eval()
    pred_all = []
    mos_all = []
    with torch.no_grad():
        accu_acc = torch.zeros(1).to(cfg.device)
        sample_num = 0
        p_pred_all = []
        d_all = []
        pred_all = []
        mos_all = []
        test_loader = tqdm(test_loader)
        for i, (img, d, mos) in enumerate(test_loader):
            sample_num += img.shape[0]
            p_pred, score = model(img.to(cfg.device))

            pred_label = torch.argmax(p_pred, 1)
            acc_num = (pred_label == d.to(cfg.device)).sum().item()
            accu_acc += acc_num
            acc = accu_acc.item() / sample_num

            p_pred_all = np.append(p_pred_all, pred_label.cpu().data.numpy())
            d_all = np.append(d_all, d.data.numpy())
            pred_all = np.append(pred_all, score.cpu().data.numpy())
            mos_all = np.append(mos_all, mos.data.numpy())
        
        logistic_pred_all = fit_function(mos_all, pred_all)
        
        plcc = pearsonr(logistic_pred_all, mos_all)[0]
        srcc = spearmanr(logistic_pred_all, mos_all)[0]
        rmse = mean_squared_error(logistic_pred_all, mos_all, squared=False)
        print("acc: %.4f, plcc: %.4f, srcc: %.4f, rmse: %.4f" % (acc, plcc, srcc, rmse))

        
if __name__ == '__main__':
    cfg = IQCaption360_config()
    main(cfg)