import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from modules.extractor import BasicEncoder, ResidualBlock, ModifiedPWCNet, ModifiedPWCNetBig
from modules.vision_transformer import _create_vision_transformer, get_positional_encodings
from modules.perceiver import generate_fourier_features
import timm
import lietorch
from lietorch import SO3, SE3, Sim3

class ViTEss(nn.Module):
    def __init__(self, args):
        super(ViTEss, self).__init__()

        self.num_images = 2
        self.pose_size = 7

        self.fund_resid = False
        if 'fund_resid' in args and args.fund_resid:
            self.fund_resid = True

        self.cnn_decoder_use_essential = False
        if 'cnn_decoder_use_essential' in args and args.cnn_decoder_use_essential:
            self.cnn_decoder_use_essential = True

        self.cnn_attn_plus_feats = False
        if 'cnn_attn_plus_feats' in args and args.cnn_attn_plus_feats:
            self.cnn_attn_plus_feats = True

        self.attn_one_way = False
        if 'attn_one_way' in args and args.attn_one_way:
            self.attn_one_way = True

        self.no_pos_encoding = None
        if 'no_pos_encoding' in args and args.no_pos_encoding != '':
            self.no_pos_encoding = args.no_pos_encoding

        self.noess = None
        if 'noess' in args and args.noess != '':
            self.noess = args.noess

        self.num_classes = self.num_images * self.pose_size
        self.feat_size = 512
        self.total_num_features = 192


        self.fnet, self.cnet = None, None
        self.flatten = nn.Flatten(0,1)
        self.resnet = models.resnet18(pretrained=True) # this will be overridden if we are loading pretrained model
        self.resnet.fc = nn.Identity()
        self.feature_resolution = (24, 24)
            
        if args.outer_prod is None:
            args.outer_prod = []
            self.outer_prod = []
        else:
            self.outer_prod = args.outer_prod

        if args.positional_encoding is None:
            args.positional_encoding = []
            self.positional_encoding = []
        else:
            self.positional_encoding = args.positional_encoding

        self.use_essential_units = args.use_essential_units

        self.fusion_transformer = None
        if args.fusion_transformer:
            distilled = False
            self.num_tokens = 2 if distilled else 1

            self.num_patches = None
            num_patches = self.num_images
            num_patches = self.feature_resolution[0] * self.feature_resolution[1]
            self.num_patches = num_patches

            self.embed_dim = 192
            self.num_heads = self.embed_dim // 64
            self.total_num_features = self.embed_dim * 2
            
            if args.cross_indices is None:
                args.cross_indices = []

            if 'seperate_tf_qkv' not in args:
                args.seperate_tf_qkv = False

            
            model_kwargs = dict(patch_size=16, embed_dim=self.embed_dim, depth=args.transformer_depth, \
                                num_heads=self.num_heads, cross_image=args.cross_indices, \
                                positional_encoding=args.positional_encoding, outer_prod=args.outer_prod,
                                use_essential_units=args.use_essential_units, 
                                cross_features=args.cross_features,
                                get_attn_scores=(self.get_attn_scores or self.cnn_decoder_use_essential), 
                                not_get_outer_prods=(not args.cnn_decoder_use_essential),
                                attn_one_way=args.attn_one_way, cnn_attn_plus_feats=args.cnn_attn_plus_feats, use_single_softmax=args.use_single_softmax,
                                seperate_tf_qkv=args.seperate_tf_qkv, no_pos_encoding=args.no_pos_encoding,
                                noess=args.noess, l1_pos_encoding=args.l1_pos_encoding)
            self.fusion_transformer = _create_vision_transformer('vit_tiny_patch16_384', **model_kwargs)

            self.transformer_depth = args.transformer_depth
            self.fusion_transformer.blocks = self.fusion_transformer.blocks[:args.transformer_depth]
            self.fusion_transformer.patch_embed = nn.Identity()
            self.fusion_transformer.head = nn.Identity() 

            nearest = 1
            if args.feature_resolution > nearest:
                nearest = 7
            while args.feature_resolution > nearest:
                nearest *= 2

            mapping = {1: 512, 7: 512, 14: 256, 28: 128, 56: 64}
            kernel_size = max(1, nearest-args.feature_resolution+1)

            outdim = int(self.total_num_features / 2)
            self.extractor_final_conv = ResidualBlock(mapping[nearest], outdim, 'batch', kernel_size=kernel_size)
        else:
            nearest = 1
            if args.feature_resolution > nearest:
                nearest = 7
            while args.feature_resolution > nearest:
                nearest *= 2
            mapping = {1: 512, 7: 512, 14: 256, 28: 128, 56: 64}
            kernel_size = max(1, nearest-args.feature_resolution+1)
            self.extractor_final_conv = ResidualBlock(mapping[nearest], self.total_num_features, 'batch', kernel_size=kernel_size)
            self.num_patches = self.feature_resolution[0] * self.feature_resolution[1]

        self.pool_transformer_output = None
        if args.pool_transformer_output:
            self.pool_feat1 = min(96, 4 * args.pool_size)
            self.pool_feat2 = args.pool_size
            if (len(self.outer_prod)==0) or (self.transformer_depth-1 in self.positional_encoding):
                self.pool_transformer_output = nn.Sequential(
                    nn.Conv2d(self.total_num_features, self.pool_feat1, kernel_size=1, bias=True),
                    nn.BatchNorm2d(self.pool_feat1),
                    nn.ReLU(),
                    nn.Conv2d(self.pool_feat1, self.pool_feat2, kernel_size=1, bias=True),
                    nn.BatchNorm2d(self.pool_feat2),
                )

            if args.fusion_transformer:
                self.fusion_transformer.cls_token = None
                self.num_tokens = 0
        
        if args.fusion_transformer:
            self.pos_encoding = None

            if num_patches >= 24*24:
                self.fusion_transformer.pos_embed = nn.Parameter(torch.zeros([1,num_patches + self.num_tokens,self.embed_dim])) 
                nn.init.xavier_uniform_(self.fusion_transformer.pos_embed) # TODO: change!
            elif num_patches == 14*14:
                self.fusion_transformer.pos_embed.data = self.fusion_transformer.pos_embed.data[:,:num_patches + self.num_tokens] 

        self.H2 = args.fc_hidden_size

        self.H = self.total_num_features * 2
        if len(self.outer_prod) > 0:
            pos_enc = 6
            if self.no_pos_encoding or self.noess:
                pos_enc = 0
            self.H = int(self.num_heads*2*(self.total_num_features/2//self.num_heads + pos_enc) * (self.total_num_features/2//self.num_heads))
            if self.transformer_depth-1 in self.positional_encoding:
                self.H += self.pool_feat2 * self.feature_resolution[0] * self.feature_resolution[1]
        elif args.pool_transformer_output:
            self.H = self.pool_feat2 * self.feature_resolution[0] * self.feature_resolution[1]
        
        num_out_images = self.num_images

        activation = nn.ReLU()

        if self.noess: # has 576x192 input (110592) instead of 384x384/6!
            self.H0 = int(self.num_heads*(self.total_num_features/2) * (self.total_num_features/2))
            self.H = int(self.feature_resolution[0]*self.feature_resolution[1]*43) # 43 is slightly larger than MLP size.
            self.pool_feat2 = 43
            self.pool_attn = nn.Sequential(
                nn.Conv2d(self.total_num_features, self.pool_feat1, kernel_size=1, bias=True),
                nn.BatchNorm2d(self.pool_feat1),
                nn.ReLU(),
                nn.Conv2d(self.pool_feat1, self.pool_feat2, kernel_size=1, bias=True),
                nn.BatchNorm2d(self.pool_feat2)
            )
            self.pose_regressor = nn.Sequential(
                nn.Linear(self.H, self.H2),
                activation,
                nn.Linear(self.H2, self.H2),
                activation, 
                nn.Linear(self.H2, num_out_images * self.pose_size),
                nn.Unflatten(1, (num_out_images, self.pose_size))
            )
        else:
            self.pose_regressor = nn.Sequential(
                nn.Linear(self.H, self.H2), 
                activation, 
                nn.Linear(self.H2, self.H2), 
                activation, 
                nn.Linear(self.H2, num_out_images * self.pose_size),
                nn.Unflatten(1, (num_out_images, self.pose_size))
            )

    def update_intrinsics(self, input_shape, intrinsics):
        sizey, sizex = self.feature_resolution
        scalex = sizex / input_shape[-1]
        scaley = sizey / input_shape[-2]
        xidx = np.array([0,2])
        yidx = np.array([1,3])
        intrinsics[:,:,xidx] = scalex * intrinsics[:,:,xidx]
        intrinsics[:,:,yidx] = scaley * intrinsics[:,:,yidx]
            
        return intrinsics

    def extract_features(self, images, intrinsics=None):
        """ run feeature extraction networks """

        # normalize images
        images = images[:, :, [2,1,0]] / 255.0
        mean = torch.as_tensor([0.485, 0.456, 0.406], device=images.device)
        std = torch.as_tensor([0.229, 0.224, 0.225], device=images.device)
        images = images.sub_(mean[:, None, None]).div_(std[:, None, None])

        if intrinsics is not None:
            intrinsics = self.update_intrinsics(images.shape, intrinsics)

        # for resnet, we need 224x224 images
        input_images = self.flatten(images)
        input_images = F.interpolate(input_images, size=224)
        

        x = self.resnet.conv1(input_images)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x) # 64, 56, 56

        x = self.resnet.layer1(x) # 64, 56, 56
        if self.feature_resolution[0] < 29:
            x = self.resnet.layer2(x) # 128, 28, 28
        if self.feature_resolution[0] < 15:
            x = self.resnet.layer3(x) # 256, 14, 14
        if self.feature_resolution[0] < 8:
            x = self.resnet.layer4(x) # 512, 7, 7
        if self.feature_resolution[0] == 1: # 512 
            x = self.resnet.avgpool(x)        
        
        x = self.extractor_final_conv(x)

        x = x.reshape([input_images.shape[0], -1, self.num_patches])
        features = x[:,:self.total_num_features//2]
        features = features.permute([0,2,1])

        return features, intrinsics
    
    def forward(self, images, Gs, intrinsics=None, inference=False):
        """ Estimates SE3 or Sim3 between pair of frames """
        pose_preds = []

        if not self.use_essential_units:
            intrinsics = None

        if not inference:
            out_Gs, out_Gs_mtx = [], []

        features, intrinsics = self.extract_features(images, intrinsics)
        B, _, _, _, _ = images.shape

        if self.fusion_transformer is not None:
            x = features[:,:,:self.embed_dim]
            if self.pool_transformer_output is not None or len(self.outer_prod)>0:
                # we re-implement forward pass which returns all patch outputs!
                x = self.fusion_transformer.patch_embed(x)

                if self.pool_transformer_output is None and (len(self.outer_prod)==0 or self.transformer_depth-1 in self.positional_encoding):
                    cls_token = self.fusion_transformer.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
                    if self.fusion_transformer.dist_token is None:
                        x = torch.cat((cls_token, x), dim=1)
                    else:
                        x = torch.cat((cls_token, self.fusion_transformer.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
                    x = x + self.fusion_transformer.pos_embed
                    x = self.fusion_transformer.pos_drop(x)
                    x = self.fusion_transformer.blocks(x)
                    features = self.fusion_transformer.norm(x)
                else:
                    x = x + self.fusion_transformer.pos_embed
                    x = self.fusion_transformer.pos_drop(x)

                    last_fundamental = None
                    for layer in range(self.transformer_depth):
                        
                        if self.use_essential_units:
                            x = self.fusion_transformer.blocks[layer](x, intrinsics=intrinsics)
                        else:
                            if self.cnn_attn_plus_feats and layer == self.transformer_depth - 2:
                                x_previous = torch.clone(x)
                            x = self.fusion_transformer.blocks[layer](x)

                    if layer in self.outer_prod and layer not in self.positional_encoding:
                        if (self.get_attn_scores and not(not self.cnn_decoder_use_essential)) or (self.cnn_decoder_use_essential):
                            (x, attention_scores) = x
                        if not(not self.cnn_decoder_use_essential):
                            features = self.fusion_transformer.norm(x)

                    elif layer in self.outer_prod and layer in self.positional_encoding:
                        (x, fundamental) = x 
                        fundamental = self.fusion_transformer.norm(fundamental).reshape(B,-1)
                        x = self.fusion_transformer.norm(x)
                        
                        x = x.contiguous().reshape([-1, 2, self.feature_resolution[0], self.feature_resolution[1], self.total_num_features//2])
                        x = x.permute([0,2,3,1,4])
                        x = x.contiguous().reshape([-1, self.feature_resolution[0], self.feature_resolution[1], self.total_num_features])
                        x = x.contiguous().permute([0,3,1,2]).contiguous()
                        x = self.pool_transformer_output(x).reshape(B,-1)

                        features = torch.cat([x, fundamental],dim=-1)
                    else:
                        x = self.fusion_transformer.norm(x)

                        x = x.reshape([-1, 2, self.feature_resolution[0], self.feature_resolution[1], self.total_num_features//2])
                        x = x.permute([0,2,3,1,4])
                        x = x.reshape([-1, self.feature_resolution[0], self.feature_resolution[1], self.total_num_features])
                        x = x.contiguous().permute([0,3,1,2]).contiguous()
                        
                        if self.cnn_attn_plus_feats:
                            last_block = self.fusion_transformer.blocks[self.transformer_depth-1]
                            head_dim = self.total_num_features // self.num_heads
                            scale = head_dim ** -0.5
                            q, k, _ = last_block.attn.qkv(x_previous).reshape(B, self.num_patches, 3, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
                            attn = (q @ k.transpose(-2, -1)) * scale
                            attention_scores = attn.softmax(dim=-1)
                            attention_scores = attention_scores.permute([0,1,3,2]).reshape([B,-1, self.feature_resolution[0], self.feature_resolution[1]])

                            x = torch.cat([attention_scores, x], dim=1)

                        features = self.pool_transformer_output(x)

            else:
                features = self.fusion_transformer(x)
        else:
            reshaped_features = features.reshape([-1,self.feature_resolution[0],self.feature_resolution[1],self.total_num_features])
            features = self.pool_transformer_output(reshaped_features.permute(0,3,1,2))

        if not isinstance(Gs, SE3):
            Gs = SE3(torch.from_numpy(Gs).unsqueeze(0).cuda().float())

        if self.noess:
            # 12, 576, 192
            features = features.reshape([B,self.feature_resolution[0], self.feature_resolution[1],-1]).permute([0,3,1,2])
            pooled_features = self.pool_attn(features)
            pose_preds.append(self.pose_regressor(pooled_features.reshape([B, -1])))
        else:
            pose_preds.append(self.pose_regressor(features.reshape([B, -1])))

        for ppi in range(len(pose_preds)):
            this_out_Gs_mtx = None
            
            pred_out_Gs = SE3(pose_preds[ppi])
            
            normalized = pred_out_Gs.data[:,:,3:].norm(dim=-1).unsqueeze(2)
            eps = torch.ones_like(normalized) * .01
            pred_out_Gs_new = SE3(torch.clone(pred_out_Gs.data))
            pred_out_Gs_new.data[:,:,3:] = pred_out_Gs.data[:,:,3:] / torch.max(normalized, eps)
            these_out_Gs = SE3(torch.cat([Gs[:,:1].data, pred_out_Gs_new.data[:,1:]], dim=1))
                
            if inference:
                out_Gs = these_out_Gs.data[0].cpu().numpy()
                out_Gs_mtx = this_out_Gs_mtx
            else:
                out_Gs.append(these_out_Gs)
                out_Gs_mtx.append(this_out_Gs_mtx)
        
        if self.get_attn_scores:
            return out_Gs, out_Gs_mtx, attention_scores

        return out_Gs, out_Gs_mtx