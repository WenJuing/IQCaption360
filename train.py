import os
import math
import torch
import torch.nn as nn
from IQCaption360 import create_model
from utils import train_one_epoch_IQA, test_IQA, compute_IQCaption360, norm_loss_with_normalization, set_seed
from torch.utils.data import DataLoader
from my_dataset import MyDataset
from config import IQCaption360_config
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import transforms as transforms
import warnings
from scipy.optimize import OptimizeWarning
import time
import sys
from weight_methods import WeightMethods


def main(cfg):
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print(cfg)
    set_seed(cfg)
    warnings.filterwarnings("ignore", category=OptimizeWarning)
    if cfg.use_tensorboard is True:
        ts_path = cfg.tensorboard_path + "/" + cfg.model_name
        if os.path.exists(ts_path) is False:
            os.makedirs(ts_path)
        sw = SummaryWriter(log_dir=ts_path)
    
    # create model
    model = create_model(pretrained=True, cfg=cfg).to(cfg.device)
    compute_IQCaption360(cfg)

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / cfg.epochs)) / 2) * (1 - cfg.lrf) + cfg.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    
    loss_func1 = nn.CrossEntropyLoss()
    loss_func2 = norm_loss_with_normalization       # norm-in-norm loss
    
    # load pre-train weight
    if cfg.load_ckpt_path != "":
        assert os.path.exists(cfg.load_ckpt_path), "weights file: '{}' not exist.".format(cfg.load_ckpt_path)
        checkpoint = torch.load(cfg.load_ckpt_path, map_location=cfg.device)
        print(model.load_state_dict(checkpoint['model_state_dict'], strict=False))
        if cfg.continue_training:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print("weights had been load!\n")
    
    train_transform = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_dataset = MyDataset(cfg=cfg, info_csv_path=cfg.train_info_csv_path, transform=train_transform)
    print(len(train_dataset), "train data has been load!")
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.batch_size, 
        num_workers=cfg.num_workers, 
        shuffle=True,
        drop_last=True,
    )
    test_dataset = MyDataset(cfg=cfg, info_csv_path=cfg.test_info_csv_path, transform=test_transform)
    print(len(test_dataset), "test data has been load!")
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=False,
    )

    best_plcc = 0
    best_srcc = 0
    best_rmse = 0
    best_epoch = 0
    begin_epoch = 0
    total_time = 0

    # weighted loss
    weighting_method = WeightMethods(
            method='dwa',
            n_tasks=2,
            alpha=1.5,
            temp=2.0,
            n_train_batch=cfg.batch_size,
            n_epochs=cfg.epochs,
            main_task=0,
            device=cfg.device
        )
    print("model:", cfg.model_name, "| dataset:", cfg.dataset_name, "| device:", cfg.device)
    for epoch in range(begin_epoch, cfg.epochs):
        # train
        start_time = time.time()
        train_loss, train_acc, train_plcc, train_srcc, train_rmse, w = train_one_epoch_IQA(model, train_loader, loss_func1, loss_func2, optimizer, epoch, weighting_method, cfg)
        end_time = time.time()
        spend_time = end_time-start_time
        total_time += spend_time
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "[train epoch %d/%d] loss: %.6f, w1: %.4f, w2: %.4f, acc: %.4f, plcc: %.4f, srcc: %.4f, rmse: %.4f, lr: %.6f, time: %.2f min, total time: %.2f h" % \
                (epoch+1, cfg.epochs, train_loss, w[0], w[1], train_acc, train_plcc, train_srcc, train_rmse, optimizer.param_groups[0]["lr"], spend_time/60, total_time/3600))
        sys.stdout.flush()
        
        scheduler.step()

        # test
        start_time = time.time()
        test_acc, test_plcc, test_srcc, test_rmse = test_IQA(model, test_loader, epoch, cfg)
        end_time = time.time()
        spend_time = end_time-start_time
        total_time += spend_time
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "[test  epoch %d/%d] ACC: %.4f, PLCC: %.4f, SRCC: %.4f, RMSE: %.4f, LR: %.6f, TIME: %.2f MIN, total time: %.2f h" % \
                (epoch+1, cfg.epochs, test_acc, test_plcc, test_srcc, test_rmse, optimizer.param_groups[0]["lr"], spend_time/60, total_time/3600))
        sys.stdout.flush()

        if cfg.use_tensorboard is True:
            sw.add_scalars(cfg.model_name+"/"+cfg.dataset_name+" Loss", {'train': train_loss}, epoch)
            sw.add_scalars(cfg.model_name+"/"+cfg.dataset_name+" acc", {'train': train_acc, 'test': test_acc}, epoch)
            sw.add_scalars(cfg.model_name+"/"+cfg.dataset_name+" plcc", {'train': train_plcc, 'test': test_plcc}, epoch)
            sw.add_scalars(cfg.model_name+"/"+cfg.dataset_name+" srcc", {'train': train_srcc, 'test': test_srcc}, epoch)
            sw.add_scalars(cfg.model_name+"/"+cfg.dataset_name+" rmse", {'train': train_rmse, 'test': test_rmse}, epoch)
            sw.add_scalar(cfg.model_name+"/"+cfg.dataset_name+" learning_rate", optimizer.param_groups[0]["lr"], epoch)
        if test_plcc + test_srcc > best_plcc + best_srcc:
            best_plcc = test_plcc
            best_srcc = test_srcc
            best_rmse = test_rmse
            best_epoch = epoch+1
            w_phat = cfg.save_ckpt_path + "/" + cfg.model_name
            if os.path.exists(w_phat) is False:
                os.makedirs(w_phat)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_plcc': best_plcc,
                'best_srcc': best_srcc,
                }, w_phat + "/best_epoch_"+str(epoch+1)+".pth")
        
        if (epoch % 10 == 0 or epoch == (cfg.epochs-1)) and epoch > 0:
            print("="*80)
            print("[test epoch %d/%d] best_PLCC: %.4f, best_SRCC: %.4f, best_RMSE: %.4f" % (best_epoch, cfg.epochs, best_plcc, best_srcc, best_rmse))
            print("="*80)


if __name__ == '__main__':
    cfg = IQCaption360_config()
    main(cfg)