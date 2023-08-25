import torch, torch.nn
import torch.nn.functional as F
import timm
import numpy as np
import cv2

from torch.utils.checkpoint import checkpoint
from typing import List, Optional

def prepare_input(img:np.ndarray, shape=(2048,2048)):
    dst = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    rate = min(shape[0]/img.shape[0], shape[1]/img.shape[1])
    resized = cv2.resize(img, None, fx=rate, fy=rate)
    dst[:resized.shape[0],:resized.shape[1]] = resized
    dst = dst.transpose(2,0,1) / 255.

    return torch.from_numpy(dst).float(), rate
    
class FPN(torch.nn.Module):
    def __init__(self, in_channels=[96, 192, 384, 768], out_channel=128) -> None:
        super().__init__()
        in_convs = [torch.nn.Conv2d(inch, out_channel, 1, bias=False) for inch in in_channels]
        self.in_convs = torch.nn.ModuleList(in_convs)
        p_convs = [torch.nn.Conv2d(out_channel, out_channel//len(in_channels), 3, 1, 1, bias=False) for _ in range(len(in_channels))]
        self.p_convs = torch.nn.ModuleList(p_convs)
        self.grad_checkpoint = False
        self._out_channel=out_channel

    @property
    def out_channel(self):
        return self._out_channel

    @torch.jit.ignore
    def no_weight_decay(self):
        return []
    
    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpoint = enable

    def forward_no_gc(self, xx:List[torch.Tensor]):
        ins = [self.in_convs[ii](xi) for ii, xi in enumerate(xx)]
        ups = []
        for ii in range(1,len(ins)):
            hh, ww = ins[ii-1].shape[-2:]
            up_i = ins[ii-1] + F.upsample(ins[ii], (hh, ww))
            ups.append(up_i)
        ups.append(ins[-1])
        ups = [self.p_convs[ii](xi) for ii, xi in enumerate(ups)]
        hh, ww = ups[0].shape[-2:]
        ups = [F.upsample(xi, (hh,ww)) for xi in ups]

        return torch.cat(ups, dim=1)
    
    def forward_gc(self, xx:List[torch.Tensor]):
        ins = [checkpoint(self.in_convs[ii], xi, use_reentrant=False) for ii, xi in enumerate(xx)]
        ups = []
        for ii in range(1,len(ins)):
            hh, ww = ins[ii-1].shape[-2:]
            up_i = ins[ii-1] + F.upsample(ins[ii], (hh, ww))
            ups.append(up_i)
        ups.append(ins[-1])
        ups = [checkpoint(self.p_convs[ii], xi, use_reentrant=False) for ii, xi in enumerate(ups)]
        hh, ww = ups[0].shape[-2:]
        ups = [F.upsample(xi, (hh,ww)) for xi in ups]

        return torch.cat(ups, dim=1)

    def forward(self, xx:List[torch.Tensor]):
        if self.grad_checkpoint:
            return self.forward_gc(xx)
        else:
            return self.forward_no_gc(xx)

class MaxxVitBase256(torch.nn.Module):
    STEM_WIDTH = 64
    def __init__(self, use_fpn=False, fpn_dim=128) -> None:
        super().__init__()
        # self.backbone:timm.models.maxxvit.MaxxVit = timm.create_model('maxvit_base_224', img_size=256, rel_pos_type='mlp')
        # timm 0.9.xでmaxvit_base_224がなくなったので代わりに直接作成
        self.backbone = timm.models.MaxxVit(
            timm.models.MaxxVitCfg(
                embed_dim=(96, 192, 384, 768),
                depths=(2, 6, 14, 2),
                block_type=('M',) * 4,
                stem_width=64,
                ),
            img_size=256
        )
        
        # 使わないweightがあると怒られることが有るのでなかったコトにする
        self.backbone.head = None
        self.mask_token = torch.nn.Parameter(
            torch.zeros(
                3 # RGB
            )
        )
        self.grad_checkpoint = False

        self.fpn = None
        if use_fpn:
            self.fpn = FPN(in_channels=self.feature_channels, out_channel=fpn_dim)

    def freeze(self, enable=True):
        for param in self.parameters():
            param.requires_grad = not enable

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.backbone.set_grad_checkpointing(enable)
        self.grad_checkpoint = enable
        if self.fpn:
            self.fpn.set_grad_checkpointing(enable)

    @property
    def feature_channels(self):
        if self.fpn:
            return [96, 192, 384, 768, self.fpn.out_channel]
        else:
            return [96, 192, 384, 768]

    @torch.jit.ignore
    def no_weight_decay(self):
        _names = self.backbone.no_weight_decay()
        _names = ['backbone.'+nn for nn in _names]
        _names.append('mask_token')

        if self.fpn:
            _fpn = self.fpn.no_weight_decay()
            _fpn = ['fpn'+nn for nn in _fpn]
            _names.extend(_fpn)

        return _names

    def _apply_mask_token(self, xx:torch.Tensor, mask:torch.Tensor):        
        if mask is None:
            return xx
        replace = torch.where(mask==1)
        xx = xx.permute(0,2,3,1).clone()
        xx[replace] = self.mask_token.type_as(xx)
        xx = xx.permute(0,3,1,2)

        return xx

    def forward(self, img, mask=None):
        img = self._apply_mask_token(img, mask)
        xx = self.backbone.stem(img)

        interm = [xx,]

        for ii, stage in enumerate(self.backbone.stages):
            xx = stage(xx)
            if ii == len(self.backbone.stages)-1:
                xx = self.backbone.norm(xx)
            interm.append(xx)

        if self.fpn:
            xx = self.fpn(interm[1:])
            interm.append(xx)

        return interm