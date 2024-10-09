"""
360 Image Quality Caption (IQCaption360)
"""
import torch
import torch.nn as nn
from config import IQCaption360_config
import torch.nn.functional as F
import numpy as np
from nat import ConvTokenizer, NATBlock


class IQCaption360(nn.Module):
    def __init__(
        self, depths, num_heads, embed_dim, kernel_size=7, norm_layer=nn.LayerNorm, cfg=None, **kwargs):
        super(IQCaption360, self).__init__()
        self.num_vps = cfg.num_vps
        self.num_levels = len(depths)
        self.num_features = int(embed_dim * 2 ** (len(depths) - 1))
        self.patch_embed = ConvTokenizer(
            in_chans=cfg.img_channels, embed_dim=embed_dim, norm_layer=norm_layer
        )
        self.pos_drop = nn.Dropout(p=cfg.pos_drop_rate)

        dpr = [x.item() for x in torch.linspace(0, cfg.drop_path_rate, sum(depths))]
        self.levels = nn.ModuleList()
        for i in range(self.num_levels):
            level = NATBlock(
                dim=int(embed_dim * 2**i),
                depth=depths[i],
                num_heads=num_heads[i],
                kernel_size=kernel_size,
                dilations=None if cfg.dilations is None else cfg.dilations[i],
                mlp_ratio=cfg.mlp_ratio,
                qkv_bias=cfg.qkv_bias,
                qk_scale=cfg.qk_scale,
                drop=cfg.drop_rate,
                attn_drop=cfg.attn_drop_rate,
                drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                norm_layer=norm_layer,
                downsample=(i < self.num_levels - 1),
                layer_scale=cfg.layer_scale,
            )
            self.levels.append(level)

        self.afa = AFA(norm_layer=norm_layer, cfg=cfg)

        self.vpfs = VPFS(num_vps=cfg.num_vps, select_rate=cfg.select_rate, chs=cfg.channels[-1])
        self.drpn = DRPN(num_classes=cfg.num_classes, cfg=cfg)
        self.qspn = QSPN(in_chs=cfg.channels[-1], hidden_dim=cfg.hidden_dim)
        
        self.apply(self._init_weights)

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        
        multi_scale_feats = []
        for level in self.levels:
            x = level(x)
            multi_scale_feats.append(x.permute(0, 3, 1, 2))

        p, q = self.afa(multi_scale_feats)

        return p, q

    def forward(self, x):
        ps = [] 
        qs = []
        scores = []
        for i in range(self.num_vps):
            p, q = self.forward_features(x[:, i, ...])
            ps.append(p)
            qs.append(q)

        p_pred = self.drpn(ps)
        
        qs = self.vpfs(qs)
        for i in range(qs.shape[1]):
            scores.append(self.qspn(qs[:, i, :].unsqueeze(dim=1)))
        q_score = torch.cat(scores, dim=1)
        score = q_score.mean(dim=-1)

        return p_pred, score

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class AFA(nn.Module):
    """Adaptive Feature Aggregation Module"""
    def __init__(self, norm_layer=nn.LayerNorm, cfg=None):
        super(AFA, self).__init__()
        self.norm1 = norm_layer(cfg.channels[0])
        self.norm2 = norm_layer(cfg.channels[1])
        self.norm3 = norm_layer(cfg.channels[2])
        self.norm4 = norm_layer(cfg.channels[0])
        self.norm5 = norm_layer(cfg.channels[1])
        self.norm6 = norm_layer(cfg.channels[2])
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        self.msfs1 = MSFS(chs_list=cfg.channels, chs=cfg.channels[0], stage_idx=0)
        self.msfs2 = MSFS(chs_list=cfg.channels, chs=cfg.channels[1], stage_idx=1)
        self.msfs3 = MSFS(chs_list=cfg.channels, chs=cfg.channels[2], stage_idx=2)
        
        self.reduce1 = nn.Linear(np.sum(cfg.channels[:3]), cfg.channels[-1])
        self.reduce2 = nn.Linear(np.sum(cfg.channels[:3]), cfg.channels[-1])

    def forward(self, multi_scale_feats):
        msfs1 = self.msfs1(multi_scale_feats)
        msfs2 = self.msfs2(multi_scale_feats)
        msfs3 = self.msfs3(multi_scale_feats)
        
        msfs1 = msfs1.permute(0, 1, 3, 4, 2)
        msfs2 = msfs2.permute(0, 1, 3, 4, 2)
        msfs3 = msfs3.permute(0, 1, 3, 4, 2)
        
        p1, p2, p3 = msfs1[:, 0, ...], msfs2[:, 0, ...], msfs3[:, 0, ...]
        q1, q2, q3 = msfs1[:, 1, ...], msfs2[:, 1, ...], msfs3[:, 1, ...]
            
        p1 = self.norm1(p1).flatten(1, 2)
        p1 = self.avgpool(p1.transpose(1, 2)).flatten(1)
        p2 = self.norm2(p2).flatten(1, 2)
        p2 = self.avgpool(p2.transpose(1, 2)).flatten(1)
        p3 = self.norm3(p3).flatten(1, 2)
        p3 = self.avgpool(p3.transpose(1, 2)).flatten(1)
        q1 = self.norm4(q1).flatten(1, 2)
        q1 = self.avgpool(q1.transpose(1, 2)).flatten(1)
        q2 = self.norm5(q2).flatten(1, 2)
        q2 = self.avgpool(q2.transpose(1, 2)).flatten(1)
        q3 = self.norm6(q3).flatten(1, 2)
        q3 = self.avgpool(q3.transpose(1, 2)).flatten(1)
        p = self.reduce1(torch.cat((p1, p2, p3), dim=-1))
        q = self.reduce2(torch.cat((q1, q2, q3), dim=-1))
        
        return p, q
    
def gn(planes, channel_per_group=4, max_groups=32):
    groups = planes // channel_per_group
    return nn.GroupNorm(min(groups, max_groups), planes)

class MSFS(nn.Module): 
    """Multi-scale Feature Selector"""
    def __init__(self, chs_list, chs, stage_idx):
        super(MSFS, self).__init__()
        self.len = len(chs_list)
        self.stage_idx = stage_idx
        up = []
        for i in range(len(chs_list)):
            up.append(nn.Sequential(nn.Conv2d(chs_list[i], chs, 1, 1, bias=False), gn(chs)))
        self.merge = nn.ModuleList(up)
        merge_convs, fcs, convs = [], [], []
        for m in range(2):   # 2
            merge_convs.append(nn.Sequential(
                        nn.Conv2d(chs, chs//4, 1, 1, bias=False),
                        gn(chs//4),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(chs//4, chs, 1, 1, bias=False),
                        gn(chs),
                    ))
            fcs.append(nn.Sequential(
                    nn.Linear(chs, chs//4, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Linear(chs//4, self.len, bias=False),    # self.len = 4
                ))
            convs.append(nn.Sequential(nn.Conv2d(chs, chs, 3, 1, 1, bias=False), gn(chs), nn.ReLU(inplace=True)))
        self.merge_convs = nn.ModuleList(merge_convs)
        self.fcs = nn.ModuleList(fcs)
        self.convs = nn.ModuleList(convs)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim=1)
        self.relu =nn.ReLU(inplace=True)

    def forward(self, multi_feats):
        x_size = multi_feats[self.stage_idx].size()
        feas = []
        for i in range(len(multi_feats)):         
            feas.append(self.merge[i](F.interpolate(multi_feats[i], x_size[2:], mode='bilinear', align_corners=True)).unsqueeze(dim=1))
        feas = torch.cat(feas, dim=1)
        fea_sum = torch.sum(feas, dim=1)
        
        outs = []
        for mode_ in range(2):
            fea_u = self.merge_convs[mode_](fea_sum)
            fea_s = self.gap(fea_u).squeeze(-1).squeeze(-1)
            fea_z = self.fcs[mode_](fea_s)
            selects = self.softmax(fea_z)
            feas_f = selects.reshape(x_size[0], self.len, 1, 1, 1).expand_as(feas) * feas
            _, index = torch.topk(selects, 2, dim=1)
            selected = []
            for i in range(x_size[0]):
                selected.append(torch.index_select(feas_f[i], dim=0, index=index[i]).unsqueeze(dim=0))
            selected = torch.cat(selected, dim=0)
            fea_v = selected.sum(dim=1)
            outs.append(self.convs[mode_](self.relu(fea_v)).unsqueeze(dim=1))

        return torch.cat(outs, dim=1)


class VPFS(nn.Module):
    """Viewport Feature Selector"""
    def __init__(self, num_vps=8, select_rate=0.5, chs=512):
        super().__init__()
        self.len = num_vps
        self.select_rate = select_rate
        self.merge_convs = nn.Sequential(
                    nn.Conv1d(chs, chs//4, 1, 1, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(chs//4, chs, 1, 1, bias=False),
                )
        self.fcs = nn.Sequential(
                nn.Linear(chs, chs//4, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(chs//4, self.len, bias=False),
            )
        self.convs = nn.Sequential(nn.Conv1d(chs, chs, 1, 1, bias=False), nn.ReLU(inplace=True))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim=1)
        self.relu =nn.ReLU(inplace=True)

    def forward(self, qs):
        feas = torch.stack(qs, dim=1)
        fea_sum = torch.sum(feas, dim=1)
        
        fea_u = self.merge_convs(fea_sum.unsqueeze(dim=-1)).squeeze(dim=-1)
        fea_z = self.fcs(fea_u)
        selects = self.softmax(fea_z)
        feas_f = selects.reshape(feas.shape[0], self.len, 1).expand_as(feas) * feas
        _, index = torch.topk(selects, int(self.len*self.select_rate), dim=1)
        selected = []
        for i in range(feas.shape[0]):
            selected.append(torch.index_select(feas_f[i], dim=0, index=index[i]).unsqueeze(dim=0))
        out = torch.cat(selected, dim=0)

        return out
    
class DRPN(nn.Module):
    """Distorted Range Prediction Network"""
    def __init__(self, num_classes=4, cfg=None) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features=cfg.num_vps*cfg.channels[3], out_features=cfg.hidden_dim, bias=False)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(in_features=cfg.hidden_dim, out_features=num_classes, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, qs):
        q_img = torch.cat(qs, dim=-1)
        q_img = self.fc1(q_img)
        q_img = self.gelu(q_img)
        q_img = self.fc2(q_img)
        q_pred = self.softmax(q_img)
        
        return q_pred.squeeze(dim=1)
    
class QSPN(nn.Module):
    """Quality Score Prediction Network"""
    def __init__(self, in_chs=512, hidden_dim=1152) -> None:
        super().__init__()
        self.mlp1 = MlpConv(in_channels=in_chs, hidden_dim=hidden_dim)
        self.gelu = nn.GELU()
        self.mlp2 = MlpConv(in_channels=in_chs, hidden_dim=hidden_dim)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, q):
        q_s = self.gelu(self.mlp1(q))
        q_w = self.softmax(self.mlp2(q))
        score = (q_s * q_w).sum((1, 2))
        
        return score.unsqueeze(dim=-1)

class MlpConv(nn.Module):
    def __init__(self, in_channels, hidden_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_dim, 1, bias=False)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv1d(hidden_dim, in_channels, 1, bias=False)
        
    def forward(self, x):
        x = self.conv1(x.transpose(1, 2))
        x = self.gelu(x)
        x = self.conv2(x).transpose(1, 2)
        
        return x
    
def create_model(pretrained=False, cfg=None, **kwargs):
    model = IQCaption360(
        depths=cfg.depths,
        num_heads=cfg.num_heads,
        embed_dim=cfg.dim,
        mlp_ratio=cfg.mlp_ratio,
        drop_path_rate=cfg.drop_path_rate,
        kernel_size=cfg.kernel_size,
        cfg=cfg,
        **kwargs
    )
    if pretrained:
        url = "https://shi-labs.com/projects/nat/checkpoints/CLS/nat_mini.pth"
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        print(model.load_state_dict(checkpoint, strict=False))
    return model


if __name__ == "__main__":
    cfg = IQCaption360_config()
    model = create_model(pretrained=False, cfg=cfg).to("cuda:0")
    x = torch.randn(4, 8, 3, 224, 224).to("cuda:0")
    x, y = model(x)
    print(x.shape)
    print(y.shape)