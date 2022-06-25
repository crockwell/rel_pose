import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from modules.extractor import BasicEncoder, ResidualBlock, ModifiedPWCNet, ModifiedPWCNetBig
from modules.cnn_decoder import TartanVONet
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

        if args.prediction_pose_type == 'classify':
            if 'streetlearn' in args.dataset:
                self.pose_size = 540
            else:
                self.pose_size = 32

        self.get_attn_scores = False
        if ('use_sigmoid_attn' in args and args.use_sigmoid_attn) or ('supervise_epi' in args and args.supervise_epi):
            self.get_attn_scores = True

        self.weird_feats = False
        if 'weird_feats' in args and args.weird_feats:
            self.weird_feats = True

        self.fund_resid = False
        if 'fund_resid' in args and args.fund_resid:
            self.fund_resid = True

        self.supervise_epi = False
        if 'supervise_epi' in args and args.supervise_epi:
            self.supervise_epi = True
        
        self.use_cnn_decoder = False
        if 'use_cnn_decoder' in args and args.use_cnn_decoder:
            self.use_cnn_decoder = True

        self.use_positional_images = False
        if 'use_positional_images' in args and args.use_positional_images:
            self.use_positional_images = True

        self.cnn_decoder_use_essential = False
        if 'cnn_decoder_use_essential' in args and args.cnn_decoder_use_essential:
            self.cnn_decoder_use_essential = True

        self.cnn_decode_each_head = False
        if 'cnn_decode_each_head' in args and args.cnn_decode_each_head:
            self.cnn_decode_each_head = True

        self.cnn_attn_plus_feats = False
        if 'cnn_attn_plus_feats' in args and args.cnn_attn_plus_feats:
            self.cnn_attn_plus_feats = True

        self.sparse_plane_baseline = False
        if 'sparse_plane_baseline' in args and args.sparse_plane_baseline:
            self.sparse_plane_baseline = True
        self.attn_one_way = False
        if 'attn_one_way' in args and args.attn_one_way:
            self.attn_one_way = True

        self.use_fixed_intrinsics = False
        if 'use_fixed_intrinsics' in args and args.use_fixed_intrinsics:
            self.use_fixed_intrinsics = True

        self.optical_flow_input = None
        if 'optical_flow_input' in args and args.optical_flow_input != '':
            self.optical_flow_input = args.optical_flow_input

        self.no_pos_encoding = None
        if 'no_pos_encoding' in args and args.no_pos_encoding != '':
            self.no_pos_encoding = args.no_pos_encoding

        self.noess = None
        if 'noess' in args and args.noess != '':
            self.noess = args.noess

        #self.clustered_dim = 0
        #if 'clustered_dim' in args and args.clustered_dim > 0:
        #    self.clustered_dim = args.clustered_dim
        #    self.pose_size = args.clustered_dim + 3

        self.num_classes = self.num_images * self.pose_size
        self.feat_size = 512
        self.total_num_features = 192
        self.normalize_quats = args.normalize_quats
        self.prediction_pose_type = args.prediction_pose_type


        self.fnet, self.cnet = None, None
        self.use_droidslam_encoder = False
        self.flatten = nn.Flatten(0,1)
        self.use_pwc_encoder = False
        if args.use_droidslam_encoder:
            self.use_droidslam_encoder = True
            self.fnet = BasicEncoder(output_dim=128, norm_fn='instance')
            self.feature_resolution = (48,64)
            #self.cnet = BasicEncoder(output_dim=256, norm_fn='none')
            if self.optical_flow_input in ['input', 'both']:
                # change first layer of ds encoder
                import pdb; pdb.set_trace()
        elif args.use_pwc_encoder:
            self.use_pwc_encoder = True
            self.modified_pwcnet = ModifiedPWCNet()
            self.feature_resolution = (args.feature_resolution, args.feature_resolution)

            if self.optical_flow_input in ['input', 'both']:
                # change first layer of pwcnet
                import pdb; pdb.set_trace()
        elif self.sparse_plane_baseline:
            pass
        else:
            print('resnet pretrained status:', not args.no_pretrained_resnet)
            self.resnet = models.resnet18(pretrained=(not args.no_pretrained_resnet)) # this will be overridden if we are loading pretrained model
            self.resnet.fc = nn.Identity()
            self.feature_resolution = (args.feature_resolution, args.feature_resolution)

            if self.optical_flow_input in ['input', 'both']:
                # change first layer of resnet
                self.resnet.conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, bias=False) 
            
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
            self.transformer_connectivity = args.transformer_connectivity
            if args.transformer_connectivity == 'all':
                num_patches = self.feature_resolution[0] * self.feature_resolution[1] * self.num_images
            else:
                num_patches = self.feature_resolution[0] * self.feature_resolution[1]
            self.num_patches = num_patches

            pt = not args.no_pretrained_transformer# and args.transformer_connectivity != 'cross_image'

            
            if args.fundamental_temp is None:
                args.fundamental_temp = 1.0
                self.fundamental_temp = 1.0
            else:
                self.fundamental_temp = args.fundamental_temp

            if args.transformer_connectivity == 'cross_image': 
                pt = not args.no_pretrained_transformer and 'map_location' in args and args.map_location != ''

                # not pretrained!
                #kwargs = dict(input_size=(3, 384, 384), crop_pct=1.0)
                if 'use_small_cross' in args and args.use_small_cross:
                    self.embed_dim = 96
                    self.num_heads = 3
                elif args.use_medium_transformer:
                    self.embed_dim = 384
                    self.num_heads = 6
                elif args.use_medium_transformer_3head:
                    self.embed_dim = 384
                    self.num_heads = 3
                else:
                    self.embed_dim = 192
                    self.num_heads = self.embed_dim // 64
                self.total_num_features = self.embed_dim * 2
                if args.transformer_connectivity != 'cross_image':
                    self.total_num_features = self.embed_dim
                
                if args.cross_indices is None:
                    args.cross_indices = []

                if 'seperate_tf_qkv' not in args:
                    args.seperate_tf_qkv = False

                
                model_kwargs = dict(patch_size=16, embed_dim=self.embed_dim, depth=args.transformer_depth, \
                                    num_heads=self.num_heads, cross_image=args.cross_indices, \
                                    positional_encoding=args.positional_encoding, outer_prod=args.outer_prod,
                                    fundamental_temp=args.fundamental_temp, 
                                    use_essential_units=args.use_essential_units, 
                                    cross_features=args.cross_features,
                                    get_attn_scores=(self.get_attn_scores or self.cnn_decoder_use_essential), use_sigmoid_attn=args.use_sigmoid_attn,
                                    attn_scale=args.attn_scale, attn_shift=args.attn_shift, epipolar_both_dirs=args.epipolar_loss_both_dirs,
                                    first_head_only=args.first_head_only, not_get_outer_prods=(args.use_cnn_decoder and not args.cnn_decoder_use_essential),
                                    attn_one_way=args.attn_one_way, cnn_attn_plus_feats=args.cnn_attn_plus_feats, use_single_softmax=args.use_single_softmax,
                                    seperate_tf_qkv=args.seperate_tf_qkv, use_fixed_intrinsics=self.use_fixed_intrinsics, no_pos_encoding=args.no_pos_encoding,
                                    noess=args.noess, l1_pos_encoding=args.l1_pos_encoding)#, pretrained_strict=False)
                self.fusion_transformer = _create_vision_transformer('vit_tiny_patch16_384', **model_kwargs)

                if pt:
                    print('loading state dict from pretrained TF')
                    pt_tf_state_dict = timm.create_model('vit_tiny_patch16_384', pretrained=pt).state_dict()
                    #pretrained_dict = OrderedDict([(k.replace("module.", ""), v) for (k, v) in loaded_weights.items()])
                    model_dict = self.fusion_transformer.state_dict()
                    pretrained_dict = {k: v for k, v in pt_tf_state_dict.items() if k in model_dict}
                    model_dict.update(pretrained_dict) 
                    #import pdb; pdb.set_trace()
                    self.fusion_transformer.load_state_dict(model_dict)

                
            elif num_patches > 50 and num_patches < 224:
                if args.transformer_connectivity == 'in_image_stacked' or args.transformer_connectivity == 'difference':
                    self.fusion_transformer = timm.create_model('vit_small_patch16_224', pretrained=pt)
                    self.total_num_features = 384 # to be consistent with pretrained ViT
                    self.embed_dim = 384 # we could choose I think? but just be consistent
                else:
                    self.fusion_transformer = timm.create_model('vit_tiny_patch16_224_in21k', pretrained=pt)
                    self.total_num_features = 192 # to be consistent with pretrained ViT
                    self.embed_dim = 192 # we could choose I think? but just be consistent
                # vit_small_patch16_224 this has 384! -- however, this would require a bigger resnet feature size!
                # vit_base_patch16_224 this has 768!

                # vit_tiny_patch16_384 this has resolution of 384x384 and same embed dim (192)
            elif num_patches < 50:
                self.fusion_transformer = timm.create_model('vit_small_patch32_224_in21k', pretrained=pt)
                self.total_num_features = 384 # to be consistent with pretrained ViT
                self.embed_dim = 384 # we could choose I think? but just be consistent
            elif num_patches > 224 and num_patches < 785:                    
                self.fusion_transformer = timm.create_model('vit_tiny_patch16_384', pretrained=pt) # depth 12, 3 heads. 24 patch
                self.total_num_features = 192 # to be consistent with pretrained ViT
                self.embed_dim = 192
                self.num_heads = 3
            elif num_patches == int(48*48):
                self.fusion_transformer = timm.create_model('vit_tiny_patch16_384', pretrained=pt) # depth 12, 3 heads. 24 patch
                # we'll cut this down to 3 or 6 depth
                self.total_num_features = 192 # to be consistent with pretrained ViT
                self.embed_dim = 192
            elif num_patches == int(48*64):
                self.fusion_transformer = timm.create_model('vit_tiny_patch16_384', pretrained=pt) # depth 12, 3 heads. 24 patch
                # we'll cut this down to 3 or 6 depth
                self.total_num_features = 192 # to be consistent with pretrained ViT
                self.embed_dim = 192

            self.transformer_depth = args.transformer_depth
            self.fusion_transformer.blocks = self.fusion_transformer.blocks[:args.transformer_depth]
            self.fusion_transformer.patch_embed = nn.Identity()
            self.fusion_transformer.head = nn.Identity() 

            if not args.use_droidslam_encoder:
                nearest = 1
                if args.feature_resolution > nearest:
                    nearest = 7
                while args.feature_resolution > nearest:
                    nearest *= 2

                mapping = {1: 512, 7: 512, 14: 256, 28: 128, 56: 64}
                kernel_size = max(1, nearest-args.feature_resolution+1)

                if args.transformer_connectivity == 'in_image_stacked' or args.transformer_connectivity == 'cross_image':
                    outdim = int(self.total_num_features / 2)
                    if self.optical_flow_input in ['before_tf', 'both']:
                        outdim -= 2
                    self.extractor_final_conv = ResidualBlock(mapping[nearest], outdim, 'batch', kernel_size=kernel_size) #nn.Conv2d(nearest, self.total_num_features / 2, 3)
                else:
                    outdim = self.total_num_features
                    if self.optical_flow_input in ['before_tf', 'both']:
                        outdim -= 2
                    self.extractor_final_conv = ResidualBlock(mapping[nearest], outdim, 'batch', kernel_size=kernel_size) #nn.Conv2d(nearest, self.total_num_features, 3)
            else:
                indim = 128
                if args.use_pwc_encoder:
                    indim = 128
                if args.transformer_connectivity == 'in_image_stacked' or args.transformer_connectivity == 'cross_image':
                    self.extractor_final_conv = ResidualBlock(indim, int(self.total_num_features / 2), 'batch') #nn.Conv2d(nearest, self.total_num_features / 2, 3)
                else:
                    self.extractor_final_conv = ResidualBlock(indim, self.total_num_features, 'batch') #nn.Conv2d(nearest, self.total_num_features, 3)

        elif args.sparse_plane_baseline:
            from modules.sparse_plane import PlaneRCNNCameraHead
            self.sparse_plane_camera_cnn = PlaneRCNNCameraHead()
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
            self.transformer_connectivity = ''

        self.pool_transformer_output = None
        if args.pool_transformer_output:
            if self.use_cnn_decoder:
                if len(self.outer_prod)>0:
                    input_size = self.num_patches
                    if self.cnn_attn_plus_feats:
                        input_size += 64
                    if not self.cnn_decode_each_head:
                        input_size *= self.num_heads # self.num_images -- only use first image!
                else:
                    input_size = 192
                    if self.cnn_attn_plus_feats:
                        input_size += self.num_patches * 3
                if self.optical_flow_input in ['after_tf','both']:
                    input_size += 2
                self.pool_transformer_output = TartanVONet(False, self.use_positional_images, self.cnn_decoder_use_essential, input_size, self.cnn_decode_each_head)
            else:
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
            #self.fusion_transformer.pos_embed = None
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
            #elif num_patches < 
            #nn.Parameter(torch.need to initialize(1, num_patches + self.num_tokens, self.embed_dim)) ***FIXED THIS!
            #import pdb; pdb.set_trace()
            #self.fusion_transformer.pos_embed = nn.Parameter(torch.zeros([1,num_patches,self.embed_dim])) 
            #nn.init.xavier_uniform_(self.fusion_transformer.pos_embed)
            #self.fusion_transformer.pos_embed.data = self.fusion_transformer.pos_embed.data[:,:num_patches + self.num_tokens] 

        self.H2 = args.fc_hidden_size

        self.H = self.total_num_features * 2 # + self.num_images * self.pose_size
        if len(self.outer_prod) > 0:
            pos_enc = 6
            if self.no_pos_encoding or self.noess:
                pos_enc = 0
            self.H = int(self.num_heads*2*(self.total_num_features/2//self.num_heads + pos_enc) * (self.total_num_features/2//self.num_heads))
            if args.transformer_connectivity == 'in_image':
                self.H *= 2
            if self.transformer_depth-1 in self.positional_encoding:
                self.H += self.pool_feat2 * self.feature_resolution[0] * self.feature_resolution[1]
        elif args.pool_transformer_output and not self.use_cnn_decoder:
            self.H = self.pool_feat2 * self.feature_resolution[0] * self.feature_resolution[1]
            if args.transformer_connectivity == 'in_image':
                self.H *= 2
        
        num_out_images = self.num_images

        activation = nn.ReLU()

        self.use_vo_mlp = False
        if args.use_vo_mlp:
            self.use_vo_mlp = True

            self.rot_regressor = nn.Sequential(
                nn.Linear(self.H, self.H2), # self.H2
                activation, #nn.ReLU(), #nn.Dropout(.5), # 
                nn.Linear(self.H2, self.H2), # self.H2
                activation, #nn.ReLU(), # nn.Dropout(.5), # 
                nn.Linear(self.H2, num_out_images * 4),
                nn.Unflatten(1, (num_out_images, 4))
            )

            self.trans_regressor = nn.Sequential(
                nn.Linear(self.H, self.H2), # self.H2
                activation, #nn.ReLU(), #nn.Dropout(.5), # 
                nn.Linear(self.H2, self.H2), # self.H2
                activation, #nn.ReLU(), # nn.Dropout(.5), # 
                nn.Linear(self.H2, num_out_images * 3),
                nn.Unflatten(1, (num_out_images, 3))
            )
        elif self.prediction_pose_type == 'classify':
            self.pose_regressor = nn.Sequential(
                    nn.Linear(self.H, self.H2), 
                    activation, 
                    nn.Linear(self.H2, num_out_images * self.pose_size)
                )
        else:
            if not self.use_cnn_decoder and not self.sparse_plane_baseline:
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
                        nn.Linear(self.H, self.H2), # self.H2
                        activation, #nn.ReLU(), #nn.Dropout(.5), # 
                        nn.Linear(self.H2, self.H2), # self.H2
                        activation, #nn.ReLU(), # nn.Dropout(.5), # 
                        nn.Linear(self.H2, num_out_images * self.pose_size),
                        nn.Unflatten(1, (num_out_images, self.pose_size))
                    )
                else:
                    self.pose_regressor = nn.Sequential(
                        nn.Linear(self.H, self.H2), # self.H2
                        activation, #nn.ReLU(), #nn.Dropout(.5), # 
                        nn.Linear(self.H2, self.H2), # self.H2
                        activation, #nn.ReLU(), # nn.Dropout(.5), # 
                        nn.Linear(self.H2, num_out_images * self.pose_size),
                        nn.Unflatten(1, (num_out_images, self.pose_size))
                    )

    def update_intrinsics(self, input_shape, intrinsics, optical_flow=None):
        #import pdb; pdb.set_trace()
        sizey, sizex = self.feature_resolution
        scalex = sizex / input_shape[-1]
        scaley = sizey / input_shape[-2]
        xidx = np.array([0,2])
        yidx = np.array([1,3])
        intrinsics[:,:,xidx] = scalex * intrinsics[:,:,xidx]
        intrinsics[:,:,yidx] = scaley * intrinsics[:,:,yidx]

        if optical_flow is not None:
            import pdb; pdb.set_trace()
            
        return intrinsics, optical_flow

    def extract_features(self, images, intrinsics=None, optical_flow=None):
        """ run feeature extraction networks """

        # normalize images
        images = images[:, :, [2,1,0]] / 255.0
        mean = torch.as_tensor([0.485, 0.456, 0.406], device=images.device)
        std = torch.as_tensor([0.229, 0.224, 0.225], device=images.device)
        images = images.sub_(mean[:, None, None]).div_(std[:, None, None])

        #print(optical_flow.min(), optical_flow.max(), images.min(), images.max())

        if self.use_fixed_intrinsics and intrinsics is not None:
            intrinsics, optical_flow = self.update_intrinsics(images.shape, intrinsics, optical_flow)

        if self.use_droidslam_encoder or self.use_pwc_encoder:
            if self.use_droidslam_encoder:
                input_images = self.flatten(images)
                input_images = F.interpolate(input_images, size=[384, 512]).unsqueeze(0)
                if self.optical_flow_input in ['input', 'both']:
                    optical_flow = F.interpolate(optical_flow, size=[384, 512]).unsqueeze(0)
                    input_images = torch.cat([input_images, optical_flow], dim=1)
                
                fmaps = self.fnet(input_images)
                fmaps = self.flatten(fmaps)
                fmaps = self.extractor_final_conv(fmaps)
                fmaps = fmaps.reshape([fmaps.shape[0], -1, self.num_patches])
            else:
                # we also use 224x224 images for now.
                input_images = self.flatten(images)
                input_images = F.interpolate(input_images, size=224)
                if self.optical_flow_input in ['input', 'both']:
                    optical_flow = F.interpolate(optical_flow, size=224)
                    input_images = torch.cat([input_images, optical_flow], dim=1)

                fmaps = self.modified_pwcnet(input_images)
                fmaps = self.extractor_final_conv(fmaps)

            if self.transformer_connectivity == 'in_image_stacked':
                # want to use more resnet features, so that can stack half of them from each input image -- add some features from both!
                x = fmaps.reshape([images.shape[0], self.num_images, -1, self.num_patches])
                features = torch.cat([x[:,0,:int(self.total_num_features/2)], x[:,1,:int(self.total_num_features/2)]], dim=1)
            elif self.transformer_connectivity == 'difference':
                x = fmaps.reshape([images.shape[0], self.num_images, -1, self.num_patches])
                if self.num_images == 3:
                    features_tgt = torch.cat([x[:,1,:self.total_num_features//2], x[:,2,:self.total_num_features//2]], dim=1)
                    if self.weird_feats:
                        features_ref = x[:,0]
                    else:
                        features_ref = torch.cat([x[:,0,:self.total_num_features//2], x[:,0,:self.total_num_features//2]], dim=1)
                elif self.num_images == 4:
                    features_tgt = torch.cat([x[:,1,:self.total_num_features//3], x[:,2,:self.total_num_features//3], x[:,3,:self.total_num_features//3]], dim=1)
                    if self.weird_feats:
                        features_ref = x[:,0]
                    else:
                        features_ref = torch.cat([x[:,0,:self.total_num_features//3], x[:,0,:self.total_num_features//3], x[:,0,:self.total_num_features//3]], dim=1)
                else:
                    features_tgt = x[:,1]
                    features_ref = x[:,0]
                features = features_tgt - features_ref
            elif self.transformer_connectivity == 'cross_image':
                x = fmaps.reshape([input_images.shape[0], -1, self.num_patches])
                features = x[:,:self.total_num_features//2]
            else:
                features = fmaps[:,:self.total_num_features]
            features = features.permute([0,2,1])
        else:
            # for resnet, we need 224x224 images
            input_images = self.flatten(images)
            input_images = F.interpolate(input_images, size=224)
            if self.optical_flow_input in ['input', 'both']:
                optical_flow = F.interpolate(optical_flow, size=224)
                input_images = torch.cat([input_images, optical_flow], dim=1)
            

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
            if self.transformer_connectivity == 'in_image_stacked':
                # want to use more resnet features, so that can stack half of them from each input image -- add some features from both!
                x = x.reshape([images.shape[0], self.num_images, -1, self.num_patches])
                features = torch.cat([x[:,0,:self.total_num_features//2], x[:,1,:self.total_num_features//2]], dim=1)
            elif self.transformer_connectivity == 'difference':
                x = x.reshape([images.shape[0], self.num_images, -1, self.num_patches])
                features_tgt = x[:,1]
                features_ref = x[:,0]
                features = features_tgt - features_ref
            elif self.transformer_connectivity == 'cross_image' or self.fusion_transformer is None:
                features = x[:,:self.total_num_features//2]
            else:
                features = x[:,:self.total_num_features]
            features = features.permute([0,2,1])

        return features, intrinsics
    
    def forward(self, images, Gs, intrinsics=None, inference=False, optical_flow=None):
        """ Estimates SE3 or Sim3 between pair of frames """
        pose_preds = []

        if not self.use_essential_units and not (self.use_cnn_decoder):
            intrinsics = None

        if not inference:
            out_Gs, out_Gs_mtx = [], []

        if optical_flow is not None:
            # optical_flow = (optical_flow) / 200 - 1.6            
            B,_,_,H,W = optical_flow.shape
            optical_flow0 = torch.zeros_like(optical_flow)
            optical_flow = torch.cat([optical_flow0, optical_flow], dim=1).reshape(B*2,2,H,W)

        if not self.sparse_plane_baseline:
            features, intrinsics = self.extract_features(images, intrinsics, optical_flow)
        B, _, _, _, _ = images.shape

        if self.fusion_transformer is not None:
            if self.optical_flow_input in ['before_tf', 'both']:
                optical_flow_local = optical_flow.reshape(B,2,2,H,W)[:,1]
                optical_flow_local = F.interpolate(optical_flow_local, size=[self.feature_resolution[0], self.feature_resolution[1]])
                optical_flow_local = optical_flow_local.reshape(B,2,self.feature_resolution[0]*self.feature_resolution[1]).permute(0,2,1)
                features = torch.cat([features, optical_flow_local], dim=2)
                #features[:,:,-2:] = optical_flow_local #torch.cat([features, optical_flow_local], dim=1)
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
                        if layer in self.outer_prod and layer < self.transformer_depth-1:
                            # intermediate layer supervision
                            # if we are using the fundamental matrix to S&E the next layer, we don't supervise it; pass it along instead
                            if layer in self.positional_encoding:
                                (x, fundamental) = x 
                            if self.fund_resid:
                                if last_fundamental is not None:
                                    fundamental = last_fundamental + self.fusion_transformer.norm(fundamental).reshape(B,-1)
                                else:
                                    fundamental = self.fusion_transformer.norm(fundamental).reshape(B,-1)
                                last_fundamental = fundamental
                            else:
                                fundamental = self.fusion_transformer.norm(fundamental).reshape(B,-1)

                            if not inference:
                                if self.transformer_depth-1 in self.positional_encoding:
                                    x_ = self.fusion_transformer.norm(x)
                        
                                    x_ = x_.contiguous().reshape([-1, 2, self.feature_resolution[0], self.feature_resolution[1], self.total_num_features//2])
                                    x_ = x_.permute([0,2,3,1,4])
                                    x_ = x_.contiguous().reshape([-1, self.feature_resolution[0], self.feature_resolution[1], self.total_num_features])
                                    x_ = x_.contiguous().permute([0,3,1,2]).contiguous()
                                    x_ = self.pool_transformer_output(x_).reshape(B,-1)

                                    features = torch.cat([x_, fundamental],dim=-1)
                                    pose_preds.append(self.pose_regressor(features))
                                else:
                                    pose_preds.append(self.pose_regressor(fundamental))

                    if layer in self.outer_prod and layer not in self.positional_encoding:# len(self.outer_prod) > 0 and len(self.positional_encoding) == 0:
                        if (self.get_attn_scores and not(self.use_cnn_decoder and not self.cnn_decoder_use_essential)) or (self.cnn_decoder_use_essential):
                            (x, attention_scores) = x
                        if not(self.use_cnn_decoder and not self.cnn_decoder_use_essential):
                            features = self.fusion_transformer.norm(x)
                        if self.use_cnn_decoder:
                            if not self.cnn_decoder_use_essential:
                                attention_scores = x
                            cnn_attention_scores = attention_scores[:,:1].permute([0,1,2,4,3]) # put second HxW first, which will be features; first HxW will be passed spatially into CNN.
                            if self.cnn_decode_each_head:
                                cnn_attention_scores = cnn_attention_scores.reshape([B, 3, -1, self.feature_resolution[0], self.feature_resolution[1]])
                            else:
                                cnn_attention_scores = cnn_attention_scores.reshape([B, -1, self.feature_resolution[0], self.feature_resolution[1]])
                            if self.use_positional_images:
                                positional_images = get_positional_encodings(B, self.feature_resolution[0] * self.feature_resolution[1], 
                                        intrinsics=intrinsics, use_fixed_intrinsics=self.use_fixed_intrinsics).cuda()
                                if self.cnn_decode_each_head:
                                    positional_images = positional_images.reshape(B,self.feature_resolution[0], self.feature_resolution[1],6).permute([0,3,1,2]).unsqueeze(1).repeat([1,3,1,1,1])
                                    cnn_attention_scores = torch.cat([cnn_attention_scores, positional_images], dim=2)
                                else:
                                    positional_images = positional_images.reshape(B,self.feature_resolution[0], self.feature_resolution[1],6).permute([0,3,1,2])
                                    cnn_attention_scores = torch.cat([cnn_attention_scores, positional_images], dim=1)
                            if self.cnn_decoder_use_essential:
                                features = self.pool_transformer_output(cnn_attention_scores, features)
                            else:
                                features = self.pool_transformer_output(cnn_attention_scores)

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

                        # maybe this reshape doesn't work for our case?
                        if self.transformer_connectivity == 'in_image_stacked' or self.transformer_connectivity == 'difference':
                            x = x.contiguous().reshape([-1, self.feature_resolution[0], self.feature_resolution[1], self.total_num_features])
                        else: # cross_image
                            x = x.reshape([-1, 2, self.feature_resolution[0], self.feature_resolution[1], self.total_num_features//2])
                            x = x.permute([0,2,3,1,4])
                            x = x.reshape([-1, self.feature_resolution[0], self.feature_resolution[1], self.total_num_features])
                        x = x.contiguous().permute([0,3,1,2]).contiguous()
                        
                        if self.cnn_attn_plus_feats:
                            last_block = self.fusion_transformer.blocks[self.transformer_depth-1]
                            head_dim = self.total_num_features // self.num_heads
                            scale = head_dim ** -0.5
                            q, k, _ = last_block.attn.qkv(x_previous).reshape(B, self.num_patches, 3, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
                            attn = (q @ k.transpose(-2, -1)) * scale * self.fundamental_temp
                            attention_scores = attn.softmax(dim=-1)
                            attention_scores = attention_scores.permute([0,1,3,2]).reshape([B,-1, self.feature_resolution[0], self.feature_resolution[1]])

                            x = torch.cat([attention_scores, x], dim=1)

                        if self.use_positional_images:
                            positional_images = get_positional_encodings(B, self.num_patches, intrinsics=intrinsics, use_fixed_intrinsics=self.use_fixed_intrinsics).cuda()
                            positional_images = positional_images.reshape(B, self.feature_resolution[0], self.feature_resolution[1], 6).permute([0,3,1,2])
                            x = torch.cat([x, positional_images], dim=1)
                        
                        
                        if self.optical_flow_input in ['after_tf', 'both']:
                            optical_flow_local = optical_flow.reshape(B,2,2,H,W)[:,1]
                            optical_flow_local = F.interpolate(optical_flow_local, size=[self.feature_resolution[0], self.feature_resolution[1]])
                            #optical_flow_local = optical_flow_local.reshape(B,2,self.feature_resolution[0]*self.feature_resolution[1]).permute(0,2,1)
                            x = torch.cat([x, optical_flow_local], dim=1)

                        features = self.pool_transformer_output(x)

            else:
                features = self.fusion_transformer(x)
        else:
            reshaped_features = features.reshape([-1,self.feature_resolution[0],self.feature_resolution[1],self.total_num_features])
            features = self.pool_transformer_output(reshaped_features.permute(0,3,1,2))

        if not isinstance(Gs, SE3) and self.prediction_pose_type != 'classify':
            Gs = SE3(torch.from_numpy(Gs).unsqueeze(0).cuda().float())

        if self.use_vo_mlp:
            trans = self.trans_regressor(features.reshape([B, -1]))
            rot = self.rot_regressor(features.reshape([B, -1]))
            pose_preds.append(torch.cat([trans,rot],dim=2))
        elif self.use_cnn_decoder:
            pose_preds.append(features)
        elif self.sparse_plane_baseline:
            pose_preds.append(self.sparse_plane_camera_cnn(images))
        elif self.noess:
            # 12, 576, 192
            features = features.reshape([B,self.feature_resolution[0], self.feature_resolution[1],-1]).permute([0,3,1,2])
            pooled_features = self.pool_attn(features)
            pose_preds.append(self.pose_regressor(pooled_features.reshape([B, -1])))
        else:
            pose_preds.append(self.pose_regressor(features.reshape([B, -1]))) # FC might have trouble learning all mappings equally??

        for ppi in range(len(pose_preds)):
            this_out_Gs_mtx = None
            
            if self.prediction_pose_type == 'change':
                pred_out_Gs = SE3(Gs.data.clone())
                pred_out_Gs.data = pred_out_Gs.data + pose_preds[ppi]
            else:
                pred_out_Gs = SE3(pose_preds[ppi])
            
            if self.normalize_quats:
                normalized = pred_out_Gs.data[:,:,3:].norm(dim=-1).unsqueeze(2)
                eps = torch.ones_like(normalized) * .01
                pred_out_Gs_new = SE3(torch.clone(pred_out_Gs.data))
                pred_out_Gs_new.data[:,:,3:] = pred_out_Gs.data[:,:,3:] / torch.max(normalized, eps)
                these_out_Gs = SE3(torch.cat([Gs[:,:1].data, pred_out_Gs_new.data[:,1:]], dim=1))
            else:
                these_out_Gs = SE3(torch.cat([Gs[:,:1].data, pred_out_Gs.data[:,1:]], dim=1))
                
            if inference:
                out_Gs = these_out_Gs.data[0].cpu().numpy()
                out_Gs_mtx = this_out_Gs_mtx
            else:
                out_Gs.append(these_out_Gs)
                out_Gs_mtx.append(this_out_Gs_mtx)
        
        if self.get_attn_scores:
            return out_Gs, out_Gs_mtx, attention_scores

        return out_Gs, out_Gs_mtx