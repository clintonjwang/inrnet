import numpy as np
from inrnet.inn.inr import INRBatch
import torch
import math

from inrnet.inn.support import BoundingBox, Mask
Tensor = torch.Tensor
nn = torch.nn
F = nn.functional

from collections import namedtuple

class BoundingBoxToken:
    def __init__(self, bbox: BoundingBox):
        """
        Token represented as a bounding box
        """
        super().__init__()
        self.bbox = bbox

    def add_positional_encoding(self):
        """
        Integrated positional encoding.
        """
        freq_scales = [math.log2(d) for d in self.bbox.shape]
        raise NotImplementedError

class MaskToken:
    def __init__(self, mask: Mask):
        """
        Token represented as a bounding box
        """
        super().__init__()
        self.mask = mask

    def add_positional_encoding(self):
        """
        Integrated positional encoding.
        """
        raise NotImplementedError

class Sampler(nn.Module):
    def __init__(self):
        """Cross-attention layer between previous tokens (produces K) and new tokens (produces Q and V) to produce new points to sample and 
        """
        super().__init__()

    def refine_tokens(token_bboxes: BoundingBoxToken, token_informativeness, token_points_to_sample):
        token_bboxes = sample(inr)
        return

class NerfTokenizer(nn.Module):
    def __init__(self, coarse_levels: int=4):
        """
        Tokenizes a NeRF by partitioning the volume into regions corresponding to tokens.
        The partition is learned by a few cross-attention layers.
        """
        super().__init__()
        self.coarse_levels = coarse_levels

    def initialize_tokens(self):
        tokens = []
        dx = dy = dz = 1/self.coarse_levels
        for x1 in np.arange(0,.999,dx):
            for y1 in np.arange(0,.999,dy):
                for z1 in np.arange(0,.999,dz):
                    bbox = BoundingBox((x1, x1+dx), (y1, y1+dy), (z1, z1+dz))
                    tokens.append(BoundingBoxToken(bbox))
        return tokens

    def initialize_sampling_coords(self):
        raise NotImplementedError

    def forward(self, nerf: INRBatch):
        """
        1. Initializes tokens to a coarse 4*4*4 grid, and samples 64 points from each.
        2. Create a sequence of 64 tokens: an embedding of the 64 points + positional encoding
        3. Each token passes through self-attention layer to produce a set of non-overlapping masks, importance weights.
            Conditioned on the masks, regress 128 coordinates to sample, and extract a linear embedding K from the sampled RGBA values.
        4. Each token is formed from the 128 coordinates in its mask.
        5. Process the masks to produce a new token sequence of size T (the number of masks allocated at least 4 sampling points), sampling the coordinates specified up to the budget.

        Args:
            nerf (INRBatch): NeRFs to process
        """
        tokens = self.initialize_tokens()
        bboxes = nerf.initialize_token_bboxes()
        bboxes.add_positional_encoding()
        return tokens

class NerfTransformer(nn.Module):
    def __init__(self, core_layers: nn.Module, prefix_layers: nn.Module|None=None, suffix_layers: nn.Module|None=None,
            head_only: bool=False):
        """Transforms a NeRF into a 3D INR (NeRF, SDF, 3D seg, MDL, etc.)

        Args:
            core_layers (Module): task-independent layers
            prefix_layers (Module): input-dependent layers
            suffix_layers (Module): output-dependent layers
            head_only (bool): whether to only use the CLS token
        """
        super().__init__()
        self.core_layers = core_layers
        self.prefix_layers = prefix_layers
        self.suffix_layers = suffix_layers


    def forward(self, in_tokens: Tensor):
        """
        6. Embed the given tokens with positional encoding. Create Q and V from these tokens.
        7. Compute cross-attention with QKV to produce a new set of masks, importance weights, and coordinates to sample, as well as Q and V
        8. Repeat 4-6, but the new tokens must be of size T (select most important masks), and produce K.
        9. Cross-attention layer, and repeat 7-8.

        Args:
            in_tokens (Tensor): _description_

        Returns:
            _type_: _description_
        """        
        in_token_embeddings = self.prefix_layers(in_tokens)
        out_token_embeddings = self.core_layers(in_token_embeddings)
        if self.head_only:
            return self.suffix_layers(out_token_embeddings[0])
        else:
            return self.suffix_layers(out_token_embeddings)


# class AttnNet(nn.Module):
#     def __init__(self, out_ch, in_ch=3, spatial_dim=2, C=512):
#         super().__init__()
#         self.layer1 = SelfAttnINRLayer(in_ch, out_ch, spatial_dim=spatial_dim)
#         self.layer2 = SelfAttnINRLayer(C, out_ch, spatial_dim=spatial_dim)

#     def forward(self, inr):
#         x = self.layer1(inr)
#         x = F.silu(self.layer2(x), inplace=True) #torch.cat([x,t], dim=1)
#         x = F.silu(self.layer3(x), inplace=True)
#         return self.layer4(x)

def translate_vit_model(input_shape):
    # ViT = VisionTransformer(num_heads=6, embed_dims=384, drop_path_rate=0.1)
    # dict(num_classes=7, in_channels=[384, 384, 384, 384])

    pass
    # sd = torch.load('/data/vision/polina/users/clintonw/code/diffcoord/temp/upernet_convnext.pth')['state_dict']
    return



# class ViTSegmenter(nn.Module):
#     def __init__(self, ViT, decoder):
#         super().__init__()
#         self.num_classes = decoder.num_classes
#         self.ViT = ViT
#         self.decoder = decoder
#     def forward(self, inr):
#         # self = InrNet
#         x = self.ViT(inr)
#         new_inr = self.decoder(*x)
#         return new_inr
#     def __len__(self):
#         return 2
#     def __getitem__(self, ix):
#         if ix == 0:
#             return self.ViT
#         elif ix == 1:
#             return self.decoder
#     def __iter__(self):
#         yield self.ViT
#         yield self.decoder



# import torch.utils.checkpoint as cp
# from mmcv.cnn import build_norm_layer
# from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
# from mmcv.cnn.utils.weight_init import (constant_init, kaiming_init,
#                                         trunc_normal_)
# from mmcv.runner import (BaseModule, CheckpointLoader, ModuleList,
#                          load_state_dict)
# from torch.nn.modules.batchnorm import _BatchNorm
# from torch.nn.modules.utils import _pair as to_2tuple

# from mmseg.ops import resize
# from mmseg.utils import get_root_logger

# class VisionTransformer(nn.Module):
#     """Vision Transformer.
#     This backbone is the implementation of `An Image is Worth 16x16 Words:
#     Transformers for Image Recognition at
#     Scale <https://arxiv.org/abs/2010.11929>`_.
#     Args:
#         img_size (int | tuple): Input image size. Default: 224.
#         patch_size (int): The patch size. Default: 16.
#         in_channels (int): Number of input channels. Default: 3.
#         embed_dims (int): embedding dimension. Default: 768.
#         num_layers (int): depth of transformer. Default: 12.
#         num_heads (int): number of attention heads. Default: 12.
#         mlp_ratio (int): ratio of mlp hidden dim to embedding dim.
#             Default: 4.
#         out_indices (list | tuple | int): Output from which stages.
#             Default: -1.
#         qkv_bias (bool): enable bias for qkv if True. Default: True.
#         drop_rate (float): Probability of an element to be zeroed.
#             Default 0.0
#         attn_drop_rate (float): The drop out rate for attention layer.
#             Default 0.0
#         drop_path_rate (float): stochastic depth rate. Default 0.0
#         with_cls_token (bool): Whether concatenating class token into image
#             tokens as transformer input. Default: True.
#         output_cls_token (bool): Whether output the cls_token. If set True,
#             `with_cls_token` must be True. Default: False.
#         norm_cfg (dict): Config dict for normalization layer.
#             Default: dict(type='LN')
#         act_cfg (dict): The activation config for FFNs.
#             Default: dict(type='GELU').
#         patch_norm (bool): Whether to add a norm in PatchEmbed Block.
#             Default: False.
#         final_norm (bool): Whether to add a additional layer to normalize
#             final feature map. Default: False.
#         interpolate_mode (str): Select the interpolate mode for position
#             embeding vector resize. Default: bicubic.
#         num_fcs (int): The number of fully-connected layers for FFNs.
#             Default: 2.
#         norm_eval (bool): Whether to set norm layers to eval mode, namely,
#             freeze running stats (mean and var). Note: Effect on Batch Norm
#             and its variants only. Default: False.
#         with_cp (bool): Use checkpoint or not. Using checkpoint will save
#             some memory while slowing down the training speed. Default: False.
#         pretrained (str, optional): model pretrained path. Default: None.
#         init_cfg (dict or list[dict], optional): Initialization config dict.
#             Default: None.
#     """

#     def __init__(self,
#                  img_size=224,
#                  patch_size=16,
#                  in_channels=3,
#                  embed_dims=768,
#                  num_layers=12,
#                  num_heads=12,
#                  mlp_ratio=4,
#                  out_indices=-1,
#                  qkv_bias=True,
#                  drop_rate=0.,
#                  attn_drop_rate=0.,
#                  drop_path_rate=0.,
#                  with_cls_token=True,
#                  output_cls_token=False,
#                  norm_cfg=dict(type='LN'),
#                  act_cfg=dict(type='GELU'),
#                  patch_norm=False,
#                  final_norm=False,
#                  interpolate_mode='bicubic',
#                  num_fcs=2,
#                  norm_eval=False,
#                  with_cp=False,
#                  pretrained=None,
#                  init_cfg=None):
#         super(VisionTransformer, self).__init__(init_cfg=init_cfg)

#         if isinstance(img_size, int):
#             img_size = to_2tuple(img_size)
#         elif isinstance(img_size, tuple):
#             if len(img_size) == 1:
#                 img_size = to_2tuple(img_size[0])
#             assert len(img_size) == 2, \
#                 f'The size of image should have length 1 or 2, ' \
#                 f'but got {len(img_size)}'

#         if output_cls_token:
#             assert with_cls_token is True, f'with_cls_token must be True if' \
#                 f'set output_cls_token to True, but got {with_cls_token}'

#         assert not (init_cfg and pretrained), \
#             'init_cfg and pretrained cannot be set at the same time'
#         if isinstance(pretrained, str):
#             warnings.warn('DeprecationWarning: pretrained is deprecated, '
#                           'please use "init_cfg" instead')
#             self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
#         elif pretrained is not None:
#             raise TypeError('pretrained must be a str or None')

#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.interpolate_mode = interpolate_mode
#         self.norm_eval = norm_eval
#         self.with_cp = with_cp
#         self.pretrained = pretrained

#         self.patch_embed = PatchEmbed(
#             in_channels=in_channels,
#             embed_dims=embed_dims,
#             conv_type='Conv2d',
#             kernel_size=patch_size,
#             stride=patch_size,
#             padding='corner',
#             norm_cfg=norm_cfg if patch_norm else None,
#             init_cfg=None,
#         )

#         num_patches = (img_size[0] // patch_size) * \
#             (img_size[1] // patch_size)

#         self.with_cls_token = with_cls_token
#         self.output_cls_token = output_cls_token
#         self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims))
#         self.pos_embed = nn.Parameter(
#             torch.zeros(1, num_patches + 1, embed_dims))
#         self.drop_after_pos = nn.Dropout(p=drop_rate)

#         if isinstance(out_indices, int):
#             if out_indices == -1:
#                 out_indices = num_layers - 1
#             self.out_indices = [out_indices]
#         elif isinstance(out_indices, list) or isinstance(out_indices, tuple):
#             self.out_indices = out_indices
#         else:
#             raise TypeError('out_indices must be type of int, list or tuple')

#         dpr = [
#             x.item() for x in torch.linspace(0, drop_path_rate, num_layers)
#         ]  # stochastic depth decay rule

#         self.layers = ModuleList()
#         for i in range(num_layers):
#             self.layers.append(
#                 TransformerEncoderLayer(
#                     embed_dims=embed_dims,
#                     num_heads=num_heads,
#                     feedforward_channels=mlp_ratio * embed_dims,
#                     attn_drop_rate=attn_drop_rate,
#                     drop_rate=drop_rate,
#                     drop_path_rate=dpr[i],
#                     num_fcs=num_fcs,
#                     qkv_bias=qkv_bias,
#                     act_cfg=act_cfg,
#                     norm_cfg=norm_cfg,
#                     with_cp=with_cp,
#                     batch_first=True))

#         self.final_norm = final_norm
#         if final_norm:
#             self.norm1_name, norm1 = build_norm_layer(
#                 norm_cfg, embed_dims, postfix=1)
#             self.add_module(self.norm1_name, norm1)

#     @property
#     def norm1(self):
#         return getattr(self, self.norm1_name)

#     def init_weights(self):
#         if (isinstance(self.init_cfg, dict)
#                 and self.init_cfg.get('type') == 'Pretrained'):
#             logger = get_root_logger()
#             checkpoint = CheckpointLoader.load_checkpoint(
#                 self.init_cfg['checkpoint'], logger=logger, map_location='cpu')

#             if 'state_dict' in checkpoint:
#                 state_dict = checkpoint['state_dict']
#             else:
#                 state_dict = checkpoint

#             if 'pos_embed' in state_dict.keys():
#                 if self.pos_embed.shape != state_dict['pos_embed'].shape:
#                     logger.info(msg=f'Resize the pos_embed shape from '
#                                 f'{state_dict["pos_embed"].shape} to '
#                                 f'{self.pos_embed.shape}')
#                     h, w = self.img_size
#                     pos_size = int(
#                         math.sqrt(state_dict['pos_embed'].shape[1] - 1))
#                     state_dict['pos_embed'] = self.resize_pos_embed(
#                         state_dict['pos_embed'],
#                         (h // self.patch_size, w // self.patch_size),
#                         (pos_size, pos_size), self.interpolate_mode)

#             load_state_dict(self, state_dict, strict=False, logger=logger)
#         elif self.init_cfg is not None:
#             super(VisionTransformer, self).init_weights()
#         else:
#             # We only implement the 'jax_impl' initialization implemented at
#             # https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py#L353  # noqa: E501
#             trunc_normal_(self.pos_embed, std=.02)
#             trunc_normal_(self.cls_token, std=.02)
#             for n, m in self.named_modules():
#                 if isinstance(m, nn.Linear):
#                     trunc_normal_(m.weight, std=.02)
#                     if m.bias is not None:
#                         if 'ffn' in n:
#                             nn.init.normal_(m.bias, mean=0., std=1e-6)
#                         else:
#                             nn.init.constant_(m.bias, 0)
#                 elif isinstance(m, nn.Conv2d):
#                     kaiming_init(m, mode='fan_in', bias=0.)
#                 elif isinstance(m, (_BatchNorm, nn.GroupNorm, nn.LayerNorm)):
#                     constant_init(m, val=1.0, bias=0.)

#     def _pos_embeding(self, patched_img, hw_shape, pos_embed):
#         """Positiong embeding method.
#         Resize the pos_embed, if the input image size doesn't match
#             the training size.
#         Args:
#             patched_img (torch.Tensor): The patched image, it should be
#                 shape of [B, L1, C].
#             hw_shape (tuple): The downsampled image resolution.
#             pos_embed (torch.Tensor): The pos_embed weighs, it should be
#                 shape of [B, L2, c].
#         Return:
#             torch.Tensor: The pos encoded image feature.
#         """
#         assert patched_img.ndim == 3 and pos_embed.ndim == 3, \
#             'the shapes of patched_img and pos_embed must be [B, L, C]'
#         x_len, pos_len = patched_img.shape[1], pos_embed.shape[1]
#         if x_len != pos_len:
#             if pos_len == (self.img_size[0] // self.patch_size) * (
#                     self.img_size[1] // self.patch_size) + 1:
#                 pos_h = self.img_size[0] // self.patch_size
#                 pos_w = self.img_size[1] // self.patch_size
#             else:
#                 raise ValueError(
#                     'Unexpected shape of pos_embed, got {}.'.format(
#                         pos_embed.shape))
#             pos_embed = self.resize_pos_embed(pos_embed, hw_shape,
#                                               (pos_h, pos_w),
#                                               self.interpolate_mode)
#         return self.drop_after_pos(patched_img + pos_embed)

#     @staticmethod
#     def resize_pos_embed(pos_embed, input_shpae, pos_shape, mode):
#         """Resize pos_embed weights.
#         Resize pos_embed using bicubic interpolate method.
#         Args:
#             pos_embed (torch.Tensor): Position embedding weights.
#             input_shpae (tuple): Tuple for (downsampled input image height,
#                 downsampled input image width).
#             pos_shape (tuple): The resolution of downsampled origin training
#                 image.
#             mode (str): Algorithm used for upsampling:
#                 ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
#                 ``'trilinear'``. Default: ``'nearest'``
#         Return:
#             torch.Tensor: The resized pos_embed of shape [B, L_new, C]
#         """
#         assert pos_embed.ndim == 3, 'shape of pos_embed must be [B, L, C]'
#         pos_h, pos_w = pos_shape
#         cls_token_weight = pos_embed[:, 0]
#         pos_embed_weight = pos_embed[:, (-1 * pos_h * pos_w):]
#         pos_embed_weight = pos_embed_weight.reshape(
#             1, pos_h, pos_w, pos_embed.shape[2]).permute(0, 3, 1, 2)
#         pos_embed_weight = resize(
#             pos_embed_weight, size=input_shpae, align_corners=False, mode=mode)
#         cls_token_weight = cls_token_weight.unsqueeze(1)
#         pos_embed_weight = torch.flatten(pos_embed_weight, 2).transpose(1, 2)
#         pos_embed = torch.cat((cls_token_weight, pos_embed_weight), dim=1)
#         return pos_embed

#     def forward(self, inputs):
#         B = inputs.shape[0]

#         x, hw_shape = self.patch_embed(inputs)

#         # stole cls_tokens impl from Phil Wang, thanks
#         cls_tokens = self.cls_token.expand(B, -1, -1)
#         x = torch.cat((cls_tokens, x), dim=1)
#         x = self._pos_embeding(x, hw_shape, self.pos_embed)

#         if not self.with_cls_token:
#             # Remove class token for transformer encoder input
#             x = x[:, 1:]

#         outs = []
#         for i, layer in enumerate(self.layers):
#             x = layer(x)
#             if i == len(self.layers) - 1:
#                 if self.final_norm:
#                     x = self.norm1(x)
#             if i in self.out_indices:
#                 if self.with_cls_token:
#                     # Remove class token and reshape token for decoder head
#                     out = x[:, 1:]
#                 else:
#                     out = x
#                 B, _, C = out.shape
#                 out = out.reshape(B, hw_shape[0], hw_shape[1],
#                                   C).permute(0, 3, 1, 2).contiguous()
#                 if self.output_cls_token:
#                     out = [out, x[:, 0]]
#                 outs.append(out)

#         return tuple(outs)

#     def train(self, mode=True):
#         super(VisionTransformer, self).train(mode)
#         if mode and self.norm_eval:
#             for m in self.modules():
#                 if isinstance(m, nn.LayerNorm):
#                     m.eval()


# class TransformerEncoderLayer(BaseModule):
#     """Implements one encoder layer in Vision Transformer.
#     Args:
#         embed_dims (int): The feature dimension.
#         num_heads (int): Parallel attention heads.
#         feedforward_channels (int): The hidden dimension for FFNs.
#         drop_rate (float): Probability of an element to be zeroed
#             after the feed forward layer. Default: 0.0.
#         attn_drop_rate (float): The drop out rate for attention layer.
#             Default: 0.0.
#         drop_path_rate (float): stochastic depth rate. Default 0.0.
#         num_fcs (int): The number of fully-connected layers for FFNs.
#             Default: 2.
#         qkv_bias (bool): enable bias for qkv if True. Default: True
#         act_cfg (dict): The activation config for FFNs.
#             Default: dict(type='GELU').
#         norm_cfg (dict): Config dict for normalization layer.
#             Default: dict(type='LN').
#         batch_first (bool): Key, Query and Value are shape of
#             (batch, n, embed_dim)
#             or (n, batch, embed_dim). Default: True.
#         with_cp (bool): Use checkpoint or not. Using checkpoint will save
#             some memory while slowing down the training speed. Default: False.
#     """

#     def __init__(self,
#                  embed_dims,
#                  num_heads,
#                  feedforward_channels,
#                  drop_rate=0.,
#                  attn_drop_rate=0.,
#                  drop_path_rate=0.,
#                  num_fcs=2,
#                  qkv_bias=True,
#                  act_cfg=dict(type='GELU'),
#                  norm_cfg=dict(type='LN'),
#                  batch_first=True,
#                  attn_cfg=dict(),
#                  ffn_cfg=dict(),
#                  with_cp=False):
#         super(TransformerEncoderLayer, self).__init__()

#         self.norm1_name, norm1 = build_norm_layer(
#             norm_cfg, embed_dims, postfix=1)
#         self.add_module(self.norm1_name, norm1)

#         attn_cfg.update(
#             dict(
#                 embed_dims=embed_dims,
#                 num_heads=num_heads,
#                 attn_drop=attn_drop_rate,
#                 proj_drop=drop_rate,
#                 batch_first=batch_first,
#                 bias=qkv_bias))

#         self.build_attn(attn_cfg)

#         self.norm2_name, norm2 = build_norm_layer(
#             norm_cfg, embed_dims, postfix=2)
#         self.add_module(self.norm2_name, norm2)

#         ffn_cfg.update(
#             dict(
#                 embed_dims=embed_dims,
#                 feedforward_channels=feedforward_channels,
#                 num_fcs=num_fcs,
#                 ffn_drop=drop_rate,
#                 dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate)
#                 if drop_path_rate > 0 else None,
#                 act_cfg=act_cfg))
#         self.build_ffn(ffn_cfg)
#         self.with_cp = with_cp

#     def build_attn(self, attn_cfg):
#         self.attn = MultiheadAttention(**attn_cfg)

#     def build_ffn(self, ffn_cfg):
#         self.ffn = FFN(**ffn_cfg)

#     @property
#     def norm1(self):
#         return getattr(self, self.norm1_name)

#     @property
#     def norm2(self):
#         return getattr(self, self.norm2_name)

#     def forward(self, x):

#         def _inner_forward(x):
#             x = self.attn(self.norm1(x), identity=x)
#             x = self.ffn(self.norm2(x), identity=x)
#             return x

#         if self.with_cp and x.requires_grad:
#             x = cp.checkpoint(_inner_forward, x)
#         else:
#             x = _inner_forward(x)
#         return x
