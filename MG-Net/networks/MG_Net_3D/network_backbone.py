#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks.dynunet_block import UnetOutBlock
from networks.MG_Net_3D.CGCN_UpBlock import UnetrBasicBlock, UnetrUpBlock
from monai.utils import ensure_tuple_rep
from lib.utils.tools.logger import Logger as Log
from lib.models.tools.module_helper import ModuleHelper
from networks.MG_Net_3D.uxnet_encoder import uxnet_conv

from networks.MG_Net_3D.decoders import SPA, GCB
from networks.MG_Net_3D.skipconnect3d import MSDeformAttnPixelDecoder2D

class ProjectionHead(nn.Module):
    def __init__(self, dim_in, proj_dim=256, proj='convmlp', bn_type='torchbn'):
        super(ProjectionHead, self).__init__()

        Log.info('proj_dim: {}'.format(proj_dim))

        if proj == 'linear':
            self.proj = nn.Conv2d(dim_in, proj_dim, kernel_size=1)
        elif proj == 'convmlp':
            self.proj = nn.Sequential(
                nn.Conv2d(dim_in, dim_in, kernel_size=1),
                ModuleHelper.BNReLU(dim_in, bn_type=bn_type),
                nn.Conv2d(dim_in, proj_dim, kernel_size=1)
            )

    def forward(self, x):
        return F.normalize(self.proj(x), p=2, dim=1)


class MG_Net_2D(nn.Module):

    def __init__(
        self,
        in_chans=1,
        out_chans=13,
        depths=[2, 2, 2, 2],
        feat_size=[48, 96, 192, 384],
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        hidden_size: int = 768,
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        spatial_dims=2,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.depths = depths
        self.drop_path_rate = drop_path_rate
        self.feat_size = feat_size
        self.layer_scale_init_value = layer_scale_init_value
        self.out_indice = []
        for i in range(len(self.feat_size)):
            self.out_indice.append(i)

        self.spatial_dims = spatial_dims
        self.normalize = True
        img_size = (96, 96)
        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_size = ensure_tuple_rep(2, spatial_dims)
        window_size = ensure_tuple_rep(7, spatial_dims)


        self.uxnet_2d = uxnet_conv(
            in_chans=self.in_chans,
            depths=self.depths,
            dims=self.feat_size,
            drop_path_rate=self.drop_path_rate,
            layer_scale_init_value=1e-6,
            out_indices=self.out_indice
        )
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.in_chans,
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[1],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[2],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[3],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.hidden_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.hidden_size,
            out_channels=self.feat_size[3],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.feat_size[2],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[1],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=48, out_channels=self.out_chans)
        self.spa = SPA(spatial_dims=spatial_dims)
        self.gcb2D_5 = GCB(feat_size[3]*2, 11, 1, 'mr', 'gelu', 'batch', True, False, 0.2, 1, n=36, drop_path=0.0, relative_pos=True, padding=5)
        self.gcb2D_4 = GCB(feat_size[2]*2, 11, 1, 'mr', 'gelu', 'batch', True, False, 0.2, 1, n=36, drop_path=0.0, relative_pos=True, padding=5)
        self.gcb2D_3 = GCB(feat_size[1]*2, 11, 1, 'mr', 'gelu', 'batch', True, False, 0.2, 1, n=36, drop_path=0.0, relative_pos=True, padding=5)
        self.out_head3 = nn.Conv2d(feat_size[3], self.out_chans, 1)
        self.out_head2 = nn.Conv2d(feat_size[2], self.out_chans, 1)
        self.out_head1 = nn.Conv2d(feat_size[1], self.out_chans, 1)
        
        backbone_feature_shape = {'res3': {'channel': 96, 'stride': 8}, 'res4': {'channel': 192, 'stride': 16}, 'res5': {'channel': 384, 'stride': 32}}
        self.detr_decoder = MSDeformAttnPixelDecoder2D(input_shape=backbone_feature_shape)
        self.detr_output_enc2 = nn.Conv2d(120, self.feat_size[1], 1)
        self.detr_output_enc3 = nn.Conv2d(120, self.feat_size[2], 1)
        self.detr_output_enc4 = nn.Conv2d(120, self.feat_size[3], 1)

    
    def forward(self, x_in):
        outs = self.uxnet_2d(x_in)
        enc1 = self.encoder1(x_in)
        x2 = outs[0]
        enc2 = self.encoder2(x2)
        x3 = outs[1]
        enc3 = self.encoder3(x3)
        x4 = outs[2]
        enc4 = self.encoder4(x4)
        x5 = outs[3]
        
        input_tensor = {"res3": enc2, "res4": enc3, "res5": enc4}
        detr_output = self.detr_decoder(input_tensor)
        detr_output_enc2 = self.detr_output_enc2(detr_output[0])
        detr_output_enc3 = self.detr_output_enc3(detr_output[1])
        detr_output_enc4 = self.detr_output_enc4(detr_output[2])
        enc4 = enc4 + detr_output_enc4
        enc3 = enc3 + detr_output_enc3
        enc2 = enc2 + detr_output_enc2

        enc_hidden = self.encoder5(x5)
        enc_hidden = self.gcb2D_5(enc_hidden)
        enc_hidden = self.spa(enc_hidden) * enc_hidden
        dec3 = self.decoder5(enc_hidden, enc4)
        dec3 = self.gcb2D_4(dec3)
        dec3 = self.spa(dec3) * dec3
        dec2 = self.decoder4(dec3, enc3)
        dec2 = self.gcb2D_3(dec2)
        dec2 = self.spa(dec2) * dec2
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0)
        out = self.out(out)
        
        p13 = self.out_head3(dec3)
        p12 = self.out_head2(dec2)
        p11 = self.out_head1(dec1)

        p13 = F.interpolate(p13, size=(96, 96), mode='bilinear', align_corners=False)
        p12 = F.interpolate(p12, size=(96, 96), mode='bilinear', align_corners=False)
        p11 = F.interpolate(p11, size=(96, 96), mode='bilinear', align_corners=False)

        out = (out + p13 + p12 + p11) / 4
        
        return out