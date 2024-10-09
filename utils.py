import torch
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr
import numpy as np
from scipy.optimize import curve_fit
from thop import profile
import sys
import torch.nn.functional as F
from IQCaption360 import create_model


def mean_squared_error(actual, predicted, squared=True):
    """MSE or RMSE (squared=False)"""
    actual = np.array(actual)
    predicted = np.array(predicted)
    error = predicted - actual
    res = np.mean(error**2)
    
    if squared==False:
        res = np.sqrt(res)
    
    return res


def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    return yhat

def fit_function(y_label, y_output):
    beta = [np.max(y_label), np.min(y_label), np.mean(y_output), 0.5]
    popt, _ = curve_fit(logistic_func, y_output, y_label, p0=beta, maxfev=100000000)
    y_output_logistic = logistic_func(y_output, *popt)
    
    return y_output_logistic


def train_one_epoch_IQA(model, data_loader, loss_func1, loss_func2, optimizer, epoch, weighting_method, cfg):
    model.train()
    accu_loss = torch.zeros(1).to(cfg.device)    # cumulative loss
    accu_acc = torch.zeros(1).to(cfg.device)    # cumulative acc
    optimizer.zero_grad()
    
    sample_num = 0
    pred_all = []
    mos_all = []
    if cfg.use_tqdm is True:
        data_loader = tqdm(data_loader)
    for i, (img, d, mos) in enumerate(data_loader):
        sample_num += img.shape[0]
        p_pred, score = model(img.to(cfg.device))
        loss1 = loss_func1(p_pred, d.to(cfg.device).long())
        loss2 = loss_func2(score, mos.to(cfg.device), p=cfg.p, q=cfg.q)
        loss = loss1 + loss2
        total_loss = loss1.mean() + loss2.mean()
        all_loss = [loss1.mean(), loss2.mean()]
        shared_parameters = None
        last_shared_layer = None
        if not torch.isnan(total_loss):
            # weight losses and backward
            total_loss = weighting_method.backwards(
                all_loss,
                epoch=epoch,
                logsigmas=None,
                shared_parameters=shared_parameters,
                last_shared_params=last_shared_layer,
                returns=True
            )
        else:
            total_loss.backward()
            continue
        
        accu_loss += total_loss.detach()
        pred_label = torch.argmax(p_pred, 1)
        acc_num = (pred_label == d.to(cfg.device)).sum().item()
        accu_acc += acc_num
        
        loss = accu_loss.item() / sample_num
        acc = accu_acc.item() / sample_num
        optimizer.step()
        optimizer.zero_grad()

        pred_all = np.append(pred_all, score.cpu().data.numpy())
        mos_all = np.append(mos_all, mos.data.numpy())

        if cfg.use_tqdm is True:
            data_loader.set_description("[train epoch %d] loss: %.6f, acc: %.4f" % (epoch+1, loss, acc))
        elif cfg.batch_print is True:
            print("[train epoch %d/%d, batch %d/%d]  loss:%.4f, loss1: %.4f, loss2: %.4f, acc: %.4f" % (epoch+1, cfg.epochs, i+1, len(data_loader), loss, loss1, loss2, acc))
            sys.stdout.flush()

    # adaptive weighted loss
    w = weighting_method.method.lambda_weight[:, epoch]

    logistic_pred_all = fit_function(mos_all, pred_all)
    plcc = pearsonr(logistic_pred_all, mos_all)[0]
    srcc = spearmanr(logistic_pred_all, mos_all)[0]
    rmse = mean_squared_error(logistic_pred_all, mos_all, squared=False)
    
    return loss, acc, plcc, srcc, rmse, w

@torch.no_grad()
def test_IQA(model, data_loader, epoch, cfg):
    model.eval()
    accu_acc = torch.zeros(1).to(cfg.device)
    
    sample_num = 0
    pred_all = []
    mos_all = []
    if cfg.use_tqdm is True:
        data_loader = tqdm(data_loader)
    for i, (img, d, mos) in enumerate(data_loader):
        sample_num += img.shape[0]
        p_pred, score = model(img.to(cfg.device))
        pred_label = torch.argmax(p_pred, 1)
        acc_num = (pred_label == d.to(cfg.device)).sum().item()
        accu_acc += acc_num
        
        acc = accu_acc.item() / sample_num
        
        pred_all = np.append(pred_all, score.cpu().data.numpy())
        mos_all = np.append(mos_all, mos.data.numpy())
        
        if cfg.use_tqdm is True:
            data_loader.set_description("[test epoch %d] acc: %.2f" % (epoch+1, acc))
        elif cfg.batch_print is True:   
            print("[test epoch %d/%d, batch %d/%d]  acc:%.4f" % (epoch+1, cfg.epochs, i+1, len(data_loader), acc))
            sys.stdout.flush()
        
    logistic_pred_all = fit_function(mos_all, pred_all)
    plcc = pearsonr(logistic_pred_all, mos_all)[0]
    srcc = spearmanr(logistic_pred_all, mos_all)[0]
    rmse = mean_squared_error(logistic_pred_all, mos_all, squared=False)
    
    return acc, plcc, srcc, rmse


def compute_IQCaption360(cfg):
    model = create_model(pretrained=False, cfg=cfg).to(cfg.device)
    x = torch.randn(1, cfg.num_vps, 3, cfg.img_size, cfg.img_size).to(cfg.device)
    flops, params = profile(model, (x,), verbose=False)
    print("FLOPs: %.1f G" % (flops / 1E9))
    print("Params: %.1f M" % (params / 1E6))


def norm_loss_with_normalization(y_pred, y, alpha=[1, 1], p=2, q=2, eps=1e-8, detach=False, exponent=True):
    """norm_loss_with_normalization: norm-in-norm"""
    N = y_pred.size(0)
    if N > 1:  
        m_hat = torch.mean(y_pred.detach()) if detach else torch.mean(y_pred)
        y_pred = y_pred - m_hat
        normalization = torch.norm(y_pred.detach(), p=q) if detach else torch.norm(y_pred, p=q)
        y_pred = y_pred / (eps + normalization)
        y = y - torch.mean(y)
        y = y / (eps + torch.norm(y, p=q))
        scale = np.power(2, max(1, 1./q)) * np.power(N, max(0, 1./p-1./q))
        loss0, loss1 = 0, 0
        if alpha[0] > 0:
            err = y_pred - y
            if p < 1:
                err += eps
            loss0 = torch.norm(err, p=p) / scale
            loss0 = torch.pow(loss0, p) if exponent else loss0
        if alpha[1] > 0:
            rho = torch.cosine_similarity(y_pred.unsqueeze(0), y.unsqueeze(0))
            err = rho * y_pred - y
            if p < 1:
                err += eps
            loss1 = torch.norm(err, p=p) / scale
            loss1 = torch.pow(loss1, p) if exponent else loss1
        return (alpha[0] * loss0 + alpha[1] * loss1) / (alpha[0] + alpha[1])
    else:
        return F.l1_loss(y_pred, y_pred.detach())
    
    
def set_seed(cfg):
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False