import os
import torch
import argparse
import numpy as np
from PIL import Image
from IQCaption360 import create_model
from config import IQCaption360_config
from torchvision import transforms as transforms


def numerical_to_textual(pred_label, score):
    range_dict = {3: 'no distorted region', 2: 'one distorted region', 
                  1: 'two distorted region', 0: 'full distorted region'}
    
    D = range_dict[pred_label.item()]

    std_score = (score-0.001728)/(0.006276-0.001728)
    if std_score <= 0.3333:
        Q = 'pool'
    elif std_score <= 0.6667:
        Q = 'fair'
    else:
        Q = 'good'

    if Q == 'pool' and pred_label < 3:
        R = 'should be discarded'
    elif Q == 'good' and pred_label > 0:
        R = 'should be saved'
    elif Q == 'pool' and pred_label == 3 or Q == 'fair' and pred_label < 2:
        R = 'is recommended to be discarded'
    else:
        R = 'is recommended to be saved'

    return np.around(std_score.item(), 4), D, Q, R


def main(cfg, args):
    model = create_model(pretrained=False, cfg=cfg).to(cfg.device)
    checkpoint = torch.load(args.load_ckpt_path, map_location=cfg.device)
    print(model.load_state_dict(checkpoint['model_state_dict'], strict=False))
    print("weights had been load!\n")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), 
    ])

    vp_list = []
    for vp in os.listdir(args.test_img_path):
        vp = Image.open(os.path.join(args.test_img_path, vp))
        vp = transform(vp)
        vp = vp.float().unsqueeze(0)
        vp_list.append(vp)
    img = torch.cat(vp_list).unsqueeze(0)

    model.eval()
    p_pred, score = model(img.to(cfg.device))
    pred_label = torch.argmax(p_pred, 1)
    score, D, Q, R = numerical_to_textual(pred_label, score)
    print('Pred Score: {}'.format(score))
    print('pred Caption: A {}-quality omnidirectional image with {}. It {}.'.format(Q, D, R))
        
if __name__ == '__main__':
    cfg = IQCaption360_config()
    parse = argparse.ArgumentParser()

    parse.add_argument('--load_ckpt_path', type=str, default='/home/xxxy/tzw/IQCaption360/ckpt/IQCaption360/ablation/IQCaption360-OIQ10K-AFA-MSFS-VPFS_1d2-DRPN-QSPN-lr1e-4-bs32-epoch50/best_epoch_35.pth')
    parse.add_argument('--test_img_path', type=str, default='/home/xxxy/tzw/databases/viewports_8/dis45.png')
    parse.add_argument('--device', type=str, default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    args = parse.parse_args()

    main(cfg, args)