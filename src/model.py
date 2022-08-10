import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .modules.extractor import ResidualBlock
from .modules.vision_transformer import _create_vision_transformer
from lietorch import SE3

class ViTEss(nn.Module):
    def __init__(self, args):
        super(ViTEss, self).__init__()

        # hyperparams
        self.noess = None
        if 'noess' in args and args.noess != '':
            self.noess = args.noess
        self.total_num_features = 192
        self.feature_resolution = (24, 24)
        self.num_images = 2
        self.pose_size = 7
        self.num_patches = self.feature_resolution[0] * self.feature_resolution[1]
        extractor_final_conv_kernel_size = max(1, 28-self.feature_resolution[0]+1)
        self.pool_feat1 = min(96, 4 * args.pool_size)
        self.pool_feat2 = args.pool_size
        self.H2 = args.fc_hidden_size

        # layers
        self.flatten = nn.Flatten(0,1)
        self.resnet = models.resnet18(pretrained=True) # this will be overridden if we are loading pretrained model
        self.resnet.fc = nn.Identity()
        self.extractor_final_conv = ResidualBlock(128, self.total_num_features, 'batch', kernel_size=extractor_final_conv_kernel_size)

        self.fusion_transformer = None
        if args.fusion_transformer:
            self.num_heads = 3
            model_kwargs = dict(patch_size=16, embed_dim=self.total_num_features, depth=args.transformer_depth, 
                                num_heads=self.num_heads, 
                                cross_features=args.cross_features,
                                use_single_softmax=args.use_single_softmax,
                                no_pos_encoding=args.no_pos_encoding,
                                noess=args.noess, l1_pos_encoding=args.l1_pos_encoding)
            self.fusion_transformer = _create_vision_transformer('vit_tiny_patch16_384', **model_kwargs)

            self.transformer_depth = args.transformer_depth
            self.fusion_transformer.blocks = self.fusion_transformer.blocks[:args.transformer_depth]
            self.fusion_transformer.patch_embed = nn.Identity()
            self.fusion_transformer.head = nn.Identity() 
            self.fusion_transformer.cls_token = None
            self.pos_encoding = None

            # we overwrite pos_embedding as we don't have class token
            self.fusion_transformer.pos_embed = nn.Parameter(torch.zeros([1,self.num_patches,self.total_num_features])) 
            # randomly initialize as usual 
            nn.init.xavier_uniform_(self.fusion_transformer.pos_embed) 

            pos_enc = 6
            if args.no_pos_encoding or self.noess:
                pos_enc = 0
            self.H = int(self.num_heads*2*(self.total_num_features//self.num_heads + pos_enc) * (self.total_num_features//self.num_heads))
        else:
            self.H = self.pool_feat2 * self.feature_resolution[0] * self.feature_resolution[1]
            self.pool_transformer_output = nn.Sequential(
                nn.Conv2d(self.total_num_features, self.pool_feat1, kernel_size=1, bias=True),
                nn.BatchNorm2d(self.pool_feat1),
                nn.ReLU(),
                nn.Conv2d(self.pool_feat1, self.pool_feat2, kernel_size=1, bias=True),
                nn.BatchNorm2d(self.pool_feat2),
            )
        
        if self.noess: # has 576x192 input (110592) instead of 384x384/6!
            self.H = int(self.feature_resolution[0]*self.feature_resolution[1]*43) # 43 is slightly larger than MLP size.
            self.pool_feat2 = 43
            self.pool_attn = nn.Sequential(
                nn.Conv2d(self.total_num_features*2, self.pool_feat1, kernel_size=1, bias=True),
                nn.BatchNorm2d(self.pool_feat1),
                nn.ReLU(),
                nn.Conv2d(self.pool_feat1, self.pool_feat2, kernel_size=1, bias=True),
                nn.BatchNorm2d(self.pool_feat2)
            )
            self.pose_regressor = nn.Sequential(
                nn.Linear(self.H, self.H2),
                nn.ReLU(),
                nn.Linear(self.H2, self.H2),
                nn.ReLU(), 
                nn.Linear(self.H2, self.num_images * self.pose_size),
                nn.Unflatten(1, (self.num_images, self.pose_size))
            )
        else:
            self.pose_regressor = nn.Sequential(
                nn.Linear(self.H, self.H2), 
                nn.ReLU(), 
                nn.Linear(self.H2, self.H2), 
                nn.ReLU(), 
                nn.Linear(self.H2, self.num_images * self.pose_size),
                nn.Unflatten(1, (self.num_images, self.pose_size))
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
        x = self.resnet.maxpool(x) 
        x = self.resnet.layer1(x) # 64, 56, 56
        x = self.resnet.layer2(x) # 128, 28, 28       
        
        x = self.extractor_final_conv(x) # 192, 24, 24 

        x = x.reshape([input_images.shape[0], -1, self.num_patches])
        if self.fusion_transformer is None:
            features = x[:,:self.total_num_features//2]
        else:
            features = x[:,:self.total_num_features]
        features = features.permute([0,2,1])

        return features, intrinsics
    
    def normalize_preds(self, Gs, pose_preds, inference):
        pred_out_Gs = SE3(pose_preds)
        
        normalized = pred_out_Gs.data[:,:,3:].norm(dim=-1).unsqueeze(2)
        eps = torch.ones_like(normalized) * .01
        pred_out_Gs_new = SE3(torch.clone(pred_out_Gs.data))
        pred_out_Gs_new.data[:,:,3:] = pred_out_Gs.data[:,:,3:] / torch.max(normalized, eps)
        these_out_Gs = SE3(torch.cat([Gs[:,:1].data, pred_out_Gs_new.data[:,1:]], dim=1))
            
        if inference:
            out_Gs = these_out_Gs.data[0].cpu().numpy()
        else:
            out_Gs = [these_out_Gs]

        return out_Gs

    def forward(self, images, Gs, intrinsics=None, inference=False):
        """ Estimates SE3 between pair of frames """
        if not isinstance(Gs, SE3):
            Gs = SE3(torch.from_numpy(Gs).unsqueeze(0).cuda().float())

        features, intrinsics = self.extract_features(images, intrinsics)
        B, _, _, _, _ = images.shape

        if self.fusion_transformer is not None:
            x = features[:,:,:self.total_num_features]
            x = self.fusion_transformer.patch_embed(x)
            x = x + self.fusion_transformer.pos_embed
            x = self.fusion_transformer.pos_drop(x)

            for layer in range(self.transformer_depth):
                x = self.fusion_transformer.blocks[layer](x, intrinsics=intrinsics)

            features = self.fusion_transformer.norm(x)
        else:
            reshaped_features = features.reshape([-1,self.feature_resolution[0],self.feature_resolution[1],self.total_num_features])
            features = self.pool_transformer_output(reshaped_features.permute(0,3,1,2))

        if self.noess:
            # 12, 576, 192
            features = features.reshape([B,self.feature_resolution[0], self.feature_resolution[1],-1]).permute([0,3,1,2])
            pooled_features = self.pool_attn(features)
            pose_preds = self.pose_regressor(pooled_features.reshape([B, -1]))
        else:
            pose_preds = self.pose_regressor(features.reshape([B, -1]))
        
        return self.normalize_preds(Gs, pose_preds, inference)
