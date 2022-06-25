import torch
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F


class RGBDAugmentor:
    """ perform augmentation on RGB-D video """

    def __init__(self, crop_size, scale_aug=True, max_scale=0.25, use_fixed_intrinsics=False, blackwhite=False, blackwhite_pt5=False,
                    datapath=None):
        self.crop_size = crop_size
        self.scale_aug = scale_aug
        p_gray = 0.1
        if blackwhite:
            p_gray = 1.0
            print('using grayscale for all training imgs!')
        if blackwhite_pt5:
            p_gray = 0.5
            print('using 50 prob of gray')
        self.augcolor = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.4/3.14),
            transforms.RandomGrayscale(p=p_gray),
            transforms.ToTensor()])

        self.max_scale = max_scale
        self.use_fixed_intrinsics = use_fixed_intrinsics

        self.streetlearn = False
        if 'streetlearn' in datapath:
            self.streetlearn = True

    def spatial_transform(self, images, disps, poses, intrinsics):
        """ cropping and resizing """
        ht, wd = images.shape[2:]

        max_scale = self.max_scale
        min_scale = np.log2(np.maximum(
            (self.crop_size[0] + 1) / float(ht),
            (self.crop_size[1] + 1) / float(wd)))

        """
        Summary: performs crop & scale which affects resolution
        and center location, modifying intrinsics. pose not affected.
        depth / image not affected other than resolution & center crop.
        
        Q: does intrinsics changing matter?
        A: yes. Say focal length increases, in this setup resolution increases.
        Then we crop back to output resolution. We've lost some of our input "zoomed in".
        Thus, stuff will appear to be closer, so predicted depth will be smaller.
        """
        #print('b4', poses, intrinsics, images.mean(), disps.mean())

        scale = 2 ** np.random.uniform(min_scale, max_scale)
        intrinsics = scale * intrinsics # make focal length slightly larger (or smaller), meaning lens is narrower and zoomed in (or wider and zoomed out)
        disps = disps.unsqueeze(dim=1)

        images = F.interpolate(images, scale_factor=scale, mode='bilinear', 
            align_corners=False, recompute_scale_factor=True) # increased scale -> increased focal length & zoomed in -> increased resolution
        
        disps = F.interpolate(disps, scale_factor=scale, recompute_scale_factor=True) 

        #print('after', intrinsics, images.shape, images.mean(), disps.shape, disps.mean(), scale)

        # always perform center crop (TODO: try non-center crops)
        y0 = (images.shape[2] - self.crop_size[0]) // 2
        x0 = (images.shape[3] - self.crop_size[1]) // 2

        intrinsics = intrinsics - torch.tensor([0.0, 0.0, x0, y0])
        images = images[:, :, y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        disps = disps[:, :, y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]

        disps = disps.squeeze(dim=1)

        # out size is 384, 512. x0 = amount have to crop / 2 from input 480, 640 +- resolution
        #print('crop', intrinsics, images.shape, images.shape, disps.shape, self.crop_size[0], x0, y0)
        #print('end', poses, intrinsics, images.mean(), disps.mean())
        return images, poses, intrinsics, disps

    def color_transform(self, images):
        """ color jittering """
        num, ch, ht, wd = images.shape
        images = images.permute(1, 2, 3, 0).reshape(ch, ht, wd*num)
        images = 255 * self.augcolor(images[[2,1,0]] / 255.0)
        return images[[2,1,0]].reshape(ch, ht, wd, num).permute(3,0,1,2).contiguous()

    def __call__(self, images, poses, intrinsics, disps=None):
        images = self.color_transform(images)
        #print(self.scale_aug)
        if self.scale_aug:
            return self.spatial_transform(images, disps, poses, intrinsics) # check intrin if augment..

        if self.streetlearn:
            return images, poses, intrinsics, disps

        if hasattr(self, 'use_fixed_intrinsics') and self.use_fixed_intrinsics:
            sizey, sizex = self.crop_size
            scalex = sizex / images.shape[-1]
            scaley = sizey / images.shape[-2]
            xidx = np.array([0,2])
            yidx = np.array([1,3])
            intrinsics[:,xidx] = scalex * intrinsics[:,xidx]
            intrinsics[:,yidx] = scaley * intrinsics[:,yidx]
            
        #disps = F.interpolate(disps.unsqueeze(dim=1), size=self.crop_size).squeeze(dim=1)
        images = F.interpolate(images, size=self.crop_size)
        return images, poses, intrinsics, disps