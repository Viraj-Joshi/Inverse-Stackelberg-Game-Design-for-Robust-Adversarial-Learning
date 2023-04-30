import os
from socket import gaierror
import time
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T

import random

from torch.utils.data import DataLoader, Dataset
from skimage.io import imread
from PIL import Image


from pytorch3d.io import load_objs_as_meshes, load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    OpenGLPerspectiveCameras,
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    HardPhongShader,
    TexturesUV,
    BlendParams,
    SoftSilhouetteShader
)

from MeshDataset import MeshDataset
from BackgroundDataset import BackgroundDataset
from darknet import Darknet
from loss import TotalVariation, dis_loss, calc_acc, TotalVariation_3d

from torchvision.utils import save_image
import torchvision
import random

from PIL import ImageDraw
from faster_rcnn.dataset.base import Base as DatasetBase
from faster_rcnn.backbone.base import Base as BackboneBase
from faster_rcnn.bbox import BBox
from faster_rcnn.model import Model as FasterRCNN
from faster_rcnn.roi.pooler import Pooler
from faster_rcnn.config.eval_config import EvalConfig as Config
from resnet.resnet import *
from resnet.MLP import *
from tqdm import tqdm
from Extragradient import Extragradient
from torch import autograd

class Generator():
    def __init__(self, config, device):
        self.config = config
        self.device = device
        
        # Create pytorch3D renderer
        self.renderer = self.create_renderer()

        # Datasets
        self.mesh_dataset = MeshDataset(config.mesh_dir, device, max_num=config.num_meshes)
        self.bg_dataset = BackgroundDataset(config.bg_dir, config.img_size, max_num=config.num_bgs)
        self.test_bg_dataset = BackgroundDataset(config.test_bg_dir, config.img_size, max_num=config.num_test_bgs)

        self.idx = None

        self.test_bgs = DataLoader(
          self.test_bg_dataset, 
          batch_size=1, 
          shuffle=True, 
          num_workers=1)
  
        self.min_contrast = 0.8
        self.max_contrast = 1.2
        self.min_brightness = -0.1
        self.max_brightness = 0.1
        self.noise_factor = 0.10
        
        # self.G = resnet50().to(self.device)
        self.G = MLP(4691*3,4691*3).to(self.device)

    def create_adv_images(self,patch,img_batch):
        if patch is None or self.idx is None:
            if patch is None: 
                print("Patch has not been initialized ...")
            else:
                print("Index has not been initialized ...")
        
        mesh = self.mesh_dataset.meshes[0]
        total_images = []

       

        for mesh in self.mesh_dataset:
            # Copy mesh for each camera angle
            mesh = mesh.extend(self.num_angles_train)

            # Random patch augmentation
            contrast = torch.FloatTensor(1).uniform_(self.min_contrast, self.max_contrast).to(self.device)
            brightness = torch.FloatTensor(1).uniform_(self.min_brightness, self.max_brightness).to(self.device)
            noise = torch.FloatTensor(patch.shape).uniform_(-1, 1) * self.noise_factor
            noise = noise.to(self.device)
            augmented_patch = (patch * contrast) + brightness + noise

            # Clamp patch to avoid PyTorch3D issues
            clamped_patch = augmented_patch.clone().clamp(min=1e-6, max=0.99999)
            mesh.textures._atlas_padded[:,self.idx,:,:,:] = clamped_patch

            mesh.textures.atlas = mesh.textures._atlas_padded
            mesh.textures._atlas_list = None

            # Render mesh onto background image
            rand_translation = torch.randint(
            -self.config.rand_translation, 
            self.config.rand_translation, 
            (2,)
            )

            images = self.render_mesh_on_bg_batch(mesh, img_batch, self.num_angles_train, x_translation=rand_translation[0].item(),
                                                y_translation=rand_translation[1].item())
            # from PIL import Image
            # import torchvision.transforms as T
            # transform = T.ToPILImage()
            reshape_img = images[:,:,:,:3].permute(0, 3, 1, 2)
            # if i % 10:
            #     img = transform(reshape_img[0,:,:,:])
            #     img.save(f"test{i}.png")
            reshape_img = reshape_img.to(self.device)
            # i+=1
            total_images.append(reshape_img)
        return total_images

    def create_patch(self,adv_patch):
        if adv_patch is None:
            adv_patch = self.initialize_patch().flatten()
        else:
            adv_patch = self.G(adv_patch.permute(0, 3, 1, 2).flatten())
        return adv_patch


    # update the generator model weights and the patch
    def update(self,learner,adv_patch,train_logger,iteration,exp_relay):
        print("GENERATOR TURN")
        mesh = self.mesh_dataset.meshes[0]

        angle_success = None
        unseen_success_rate = None
        accuracy_rate = None

        train_bgs_loader = DataLoader(
            self.bg_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True, 
            num_workers=1)

        self.epoch_length = len(train_bgs_loader)

        # use patch from last round of the game or start anew
        # if not start and self.config.patch_dir is not None: 
        #     print('continuing off!')
        #     patch = torch.load(self.config.patch_dir + '/patch_save.pt').to(self.device)
        # elif start:
        #     patch = None
        
        
        # adv_patch.to(self.device)

        total_variation = TotalVariation_3d(mesh, self.idx).to(self.device)

        et0 = time.time()
        adv_images_to_return = None

        generator_optimizer = torch.optim.Adam(self.G.parameters(), lr=1e-3)
        self.optimizer = Extragradient(generator_optimizer, self.G.parameters())
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', min_lr=1e-8,patience = 5, verbose=True)
        for epoch in range(self.config.generator_epochs):
            ep_tv_loss = 0
            ep_score_loss = 0
            ep_loss = 0
            adv_images_to_return = []

            for batch_index, (img_batch) in tqdm(enumerate(train_bgs_loader), desc=f'Running epoch {epoch}',total=self.epoch_length):
                with torch.autograd.set_detect_anomaly(True):
                    img_batch = img_batch.to(self.device)
                    adv_patch_out = self.G(adv_patch.detach().permute(0, 3, 1, 2).flatten()).to(self.device)
                    adv_patch_out = torch.reshape(adv_patch_out,(4691,1,1,3))
                    adv_images = self.create_adv_images(adv_patch_out,img_batch) # images with adversarial patch
                    
                    if epoch ==  self.config.generator_epochs-1: 
                        adv_images_to_return.extend(adv_images)    

                    score,accuracy_rate = learner.evaluate(adv_images)

                    tv = total_variation(adv_patch_out)
                    tv_loss = tv * 2.5
                    loss = score + tv_loss

                    ep_score_loss += score.detach().cpu().numpy()
                    ep_tv_loss += tv_loss.detach().cpu().numpy()
                    ep_loss += loss

                    grad_gen = autograd.grad(loss, self.G.parameters(), retain_graph=True, allow_unused=True)
                    for p, g in zip(self.G.parameters(), grad_gen):
                        p.grad = g
                    self.optimizer.step()

                    if train_logger is not None:
                        actual_iteration = iteration * (self.epoch_length * self.config.generator_epochs) + self.epoch_length * epoch + batch_index
                        train_logger.add_images('image with patch', adv_images[0],  actual_iteration)
                        train_logger.add_scalar('missed detection rate by learner', accuracy_rate,   actual_iteration)
                        train_logger.add_scalar('generator loss', loss.item(),   actual_iteration)

                    # adv_patch_cpu.clamp(min=1e-6, max=0.99999)
                    # adv_patch_cpu = adv_patch.clone().cpu()

                    if batch_index + 1 >= len(train_bgs_loader):
                        print('\n')
                    else:
                        del loss, tv_loss, score, img_batch, adv_patch_out
                        torch.cuda.empty_cache()

            scheduler.step(ep_loss)

            if epoch ==  self.config.generator_epochs-1:
                # if iteration % 10 == 0:
                #     angle_success, unseen_success_rate = self.test_patch(learner.dnet,adv_patch_out)
                #     self.change_cameras('train')

                 # save image and patch
                patch_save = adv_patch_out.cpu().detach().clone()
                idx_save = self.idx.cpu().detach().clone()
                torch.save(patch_save, 'patch_save.pt')
                torch.save(idx_save, 'idx_save.pt')

            if iteration % 10 == 0:
                state = {
                    'epoch': iteration,
                    'state_dict': self.G.state_dict(),
                    'optimizer': self.optimizer.state_dict()
                }

                torch.save(state, f"Generator_Weights/generator_{iteration}.pt")

            et1 = time.time()
            ep_score_loss = ep_score_loss/len(train_bgs_loader)
            ep_loss = ep_loss/len(train_bgs_loader)
            ep_tv_loss = ep_tv_loss/len(train_bgs_loader)

            print('  EPOCH NR: ', epoch),
            print('EPOCH LOSS: ', ep_loss.item())
            print('SCORE LOSS: ', ep_score_loss)
            print('   TV LOSS: ', ep_tv_loss)
            print('EPOCH TIME: ', et1-et0)
            et0 = time.time()

            if train_logger is not None and epoch == self.config.generator_epochs-1:
                actual_iteration =  iteration * self.config.generator_epochs + self.config.generator_epochs
                train_logger.add_scalar('epoch loss', ep_loss.item(),actual_iteration)
            # if iteration % 10 == 0:
            #     train_logger.add_scalar('unseen_success_rate', unseen_success_rate, actual_iteration)

        return adv_images_to_return
    
    def test_patch(self,dnet,patch):
        self.change_cameras('test')
        angle_success = torch.zeros(self.num_angles_test)
        n = 0.0
        for mesh in self.mesh_dataset:
            mesh = mesh.extend(self.num_angles_test)
            for bg_batch in self.test_bgs:
                bg_batch = bg_batch.to(self.device)

                texture_image=mesh.textures.atlas_padded()
                                
                clamped_patch = patch.clone().clamp(min=1e-6, max=0.99999)
                mesh.textures._atlas_padded[:,self.idx,:,:,:] = clamped_patch
      
                mesh.textures.atlas = mesh.textures._atlas_padded
                mesh.textures._atlas_list = None
                
                rand_translation = torch.randint(
                  -self.config.rand_translation, 
                  self.config.rand_translation, 
                  (2,)
                  )

                images = self.render_mesh_on_bg_batch(mesh, bg_batch, self.num_angles_test, x_translation=rand_translation[0].item(),
                                                y_translation=rand_translation[1].item())
            
                reshape_img = images[:,:,:,:3].permute(0, 3, 1, 2)
                reshape_img = reshape_img.to(self.device)

                output = dnet(reshape_img)
                
                for angle in range(self.num_angles_test):
                    acc_loss = calc_acc(output[angle], dnet.num_classes, dnet.num_anchors, 0)
                    angle_success[angle] += acc_loss.item()

                n += bg_batch.shape[0]
        
        save_image(reshape_img[0].cpu().detach(), "TEST.png")
        unseen_success_rate = torch.sum(angle_success) / (n * self.num_angles_test)
        print('   ANGLE SUCCESS RATES: ', angle_success / n)
        print('UNSEEN BG SUCCESS RATE: ', unseen_success_rate.item())
 
        return angle_success / n, unseen_success_rate.item()

    def test_patch_faster_rcnn(self, path_to_checkpoint: str, dataset_name: str, backbone_name: str, prob_thresh: float,patch):
        dataset_class = DatasetBase.from_name(dataset_name)
        backbone = BackboneBase.from_name(backbone_name)(pretrained=False)
        model = FasterRCNN(backbone, dataset_class.num_classes(), pooler_mode=Config.POOLER_MODE,
                  anchor_ratios=Config.ANCHOR_RATIOS, anchor_sizes=Config.ANCHOR_SIZES,
                  rpn_pre_nms_top_n=Config.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=Config.RPN_POST_NMS_TOP_N).cuda()
        model.load(path_to_checkpoint)

        angle_success = torch.zeros(self.num_angles_test)
        self.change_cameras('test')
        total_loss = 0.0
        n = 0.0
        with torch.no_grad():
            for mesh in self.mesh_dataset:
                mesh = mesh.extend(self.num_angles_test)
                for bg_batch in self.test_bgs:
                    bg_batch = bg_batch.to(self.device)
                    
                    texture_image=mesh.textures.atlas_padded()
                    clamped_patch = patch.clone().clamp(min=1e-6, max=0.99999)
                    mesh.textures._atlas_padded[:,self.idx,:,:,:] = clamped_patch
          
                    mesh.textures.atlas = mesh.textures._atlas_padded
                    mesh.textures._atlas_list = None

                    rand_translation = torch.randint(
                      -self.config.rand_translation, 
                      self.config.rand_translation, 
                      (2,)
                      )

                    images = self.render_mesh_on_bg_batch(
                      mesh, 
                      bg_batch, 
                      self.num_angles_test, 
                      x_translation=rand_translation[0].item(),
                      y_translation=rand_translation[1].item()
                      )

                    reshape_img = images[:,:,:,:3].permute(0, 3, 1, 2)

                    # output = self.dnet(reshape_img)
                    save_image(reshape_img[0].cpu().detach(), "TEST_PRE.png")

                    for angle in range(self.num_angles_test):
                        # acc_loss = calc_acc(output[angle], self.dnet.num_classes, self.dnet.num_anchors, 0)
                        # angle_success[angle] += acc_loss.item()

                        image = torchvision.transforms.ToPILImage()(reshape_img[angle,:,:,:].cpu())
                        # image_tensor, scale = dataset_class.preprocess(image, Config.IMAGE_MIN_SIDE, Config.IMAGE_MAX_SIDE)
                        image_tensor = reshape_img[angle, ..., :]
                        scale = 1.0
                        save_image(image_tensor.cpu().detach(), "TEST_POST.png")

                        img = Image.open('TEST_POST.png').convert('RGB')
                        img = torchvision.transforms.ToTensor()(image)
                        image_tensor = img.cuda()

                        detection_bboxes, detection_classes, detection_probs, _ = \
                            model.eval().forward(image_tensor.unsqueeze(dim=0).cuda())
                        detection_bboxes /= scale

                        kept_indices = detection_probs > prob_thresh
                        detection_bboxes = detection_bboxes[kept_indices]
                        detection_classes = detection_classes[kept_indices]
                        detection_probs = detection_probs[kept_indices]

                        draw = ImageDraw.Draw(image)

                        for bbox, cls, prob in zip(detection_bboxes.tolist(), detection_classes.tolist(), detection_probs.tolist()):
                            color = random.choice(['red', 'green', 'blue', 'yellow', 'purple', 'white'])
                            bbox = BBox(left=bbox[0], top=bbox[1], right=bbox[2], bottom=bbox[3])
                            category = dataset_class.LABEL_TO_CATEGORY_DICT[cls]

                            draw.rectangle(((bbox.left, bbox.top), (bbox.right, bbox.bottom)), outline=color, width=3)
                            draw.text((bbox.left, bbox.top), text=f'{category:s} {prob:.3f}', fill=color)
                        if angle==0:
                            image.save("out/images/test_%d.png" % n)

                    n += 1.0

            # save_image(reshape_img[0].cpu().detach(), "TEST.png")

            unseen_success_rate = torch.sum(angle_success) / (n * self.num_angles_test)
            print('Angle success rates: ', angle_success / n)
            print('Unseen bg success rate: ', unseen_success_rate.item())
        return angle_success / n, unseen_success_rate.item()
    def initialize_patch(self):
        # print('Initializing patch...')
        # Code for sampling faces:
        # mesh = self.mesh_dataset.meshes[0]
        # box = mesh.get_bounding_boxes()
        # max_x = box[0,0,1]
        # max_y = box[0,1,1]
        # max_z = box[0,2,1]
        # min_x = box[0,0,0]
        # min_y = box[0,1,0]
        # min_z = box[0,2,0]

        # len_z = max_z - min_z
        # len_x = max_x - min_x
        # len_y = max_y - min_y

        # verts = mesh.verts_padded()
        # v_shape = verts.shape
        # sampled_verts = torch.zeros(v_shape[1]).to('cuda')

        # for i in range(v_shape[1]):
        #   #original human1 not SMPL
        #   #if verts[0,i,2] > min_z + len_z * 0.55 and verts[0,i,0] > min_x + len_x*0.3 and verts[0,i,0] < min_x + len_x*0.7 and verts[0,i,1] > min_y + len_y*0.6 and verts[0,i,1] < min_y + len_y*0.7:
        #   #SMPL front
        #   if verts[0,i,2] > min_z + len_z * 0.55 and verts[0,i,0] > min_x + len_x*0.35 and verts[0,i,0] < min_x + len_x*0.65 and verts[0,i,1] > min_y + len_y*0.65 and verts[0,i,1] < min_y + len_y*0.75:
        #   #back
        #   #if verts[0,i,2] < min_z + len_z * 0.5 and verts[0,i,0] > min_x + len_x*0.35 and verts[0,i,0] < min_x + len_x*0.65 and verts[0,i,1] > min_y + len_y*0.65 and verts[0,i,1] < min_y + len_y*0.75:
        #   #leg
        #   #if verts[0,i,0] > min_x + len_x*0.5 and verts[0,i,0] < min_x + len_x and verts[0,i,1] > min_y + len_y*0.2 and verts[0,i,1] < min_y + len_y*0.3:
        #     sampled_verts[i] = 1

        # faces = mesh.faces_padded()
        # f_shape = faces.shape

        # sampled_planes = list()
        # for i in range(faces.shape[1]):
        #   v1 = faces[0,i,0]
        #   v2 = faces[0,i,1]
        #   v3 = faces[0,i,2]
        #   if sampled_verts[v1]+sampled_verts[v2]+sampled_verts[v3]>=1:
        #     sampled_planes.append(i)
        
        # Sample faces from index file:
        sampled_planes = np.load(self.config.idx).tolist()
        idx = torch.Tensor(sampled_planes).long().to(self.device)
        self.idx = idx
        patch = []
        for _ in range(self.config.patch_num):
            patch.append(torch.rand(len(sampled_planes), 1, 1, 3, device=(self.device), requires_grad=True))
        return patch[0]

        # print(type(patch))

    def create_renderer(self):
        self.num_angles_train = self.config.num_angles_train
        self.num_angles_test = self.config.num_angles_test

        azim_train = torch.linspace(-1 * self.config.angle_range_train, self.config.angle_range_train, self.num_angles_train)
        azim_test = torch.linspace(-1 * self.config.angle_range_test, self.config.angle_range_test, self.num_angles_test)

        # Cameras for SMPL meshes:
        camera_dist = 2.2
        R, T = look_at_view_transform(camera_dist, 6, azim_train)
        train_cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)
        self.train_cameras = train_cameras

        R, T = look_at_view_transform(camera_dist, 6, azim_test)
        test_cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)
        self.test_cameras = test_cameras
        
        raster_settings = RasterizationSettings(
            image_size=self.config.img_size, 
            blur_radius=0.0, 
            faces_per_pixel=1, 
        )

        lights = PointLights(device=self.device, location=[[0.0, 85, 100.0]])

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=train_cameras, 
                raster_settings=raster_settings
            ),
            shader=HardPhongShader(
                device=self.device, 
                cameras=train_cameras,
                lights=lights
            )
        )

        return renderer
    
    def change_cameras(self, mode, camera_dist=2.2):
      azim_train = torch.linspace(-1 * self.config.angle_range_train, self.config.angle_range_train, self.num_angles_train)
      azim_test = torch.linspace(-1 * self.config.angle_range_test, self.config.angle_range_test, self.num_angles_test)

      R, T = look_at_view_transform(camera_dist, 6, azim_train)
      train_cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)
      self.train_cameras = train_cameras

      R, T = look_at_view_transform(camera_dist, 6, azim_test)
      test_cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)
      self.test_cameras = test_cameras

      if mode == 'train':
        self.renderer.rasterizer.cameras=self.train_cameras
        self.renderer.shader.cameras=self.train_cameras
      elif mode == 'test':
        self.renderer.rasterizer.cameras=self.test_cameras
        self.renderer.shader.cameras=self.test_cameras

    def render_mesh_on_bg(self, mesh, bg_img, num_angles, location=None, x_translation=0, y_translation=0):
        images = self.renderer(mesh)
        bg = bg_img.unsqueeze(0)
        bg_shape = bg.shape
        new_bg = torch.zeros(bg_shape[2], bg_shape[3], 3)
        new_bg[:,:,0] = bg[0,0,:,:]
        new_bg[:,:,1] = bg[0,1,:,:]
        new_bg[:,:,2] = bg[0,2,:,:]

        human = images[:, ..., :3]
        
        human_size = self.renderer.rasterizer.raster_settings.image_size

        if location is None:
            dH = bg_shape[2] - human_size
            dW = bg_shape[3] - human_size
            location = (
                dW // 2 + x_translation,
                dW - (dW // 2) - x_translation,
                dH // 2 + y_translation,
                dH - (dH // 2) - y_translation
            )

        contour = torch.where((human == 1).cpu(), torch.zeros(1).cpu(), torch.ones(1).cpu())
        new_contour = torch.zeros(num_angles, bg_shape[2], bg_shape[3], 3)
        
        new_contour[:,:,:,0] = F.pad(contour[:,:,:,0], location, "constant", value=0)
        new_contour[:,:,:,1] = F.pad(contour[:,:,:,1], location, "constant", value=0)
        new_contour[:,:,:,2] = F.pad(contour[:,:,:,2], location, "constant", value=0)

        new_human = torch.zeros(num_angles, bg_shape[2], bg_shape[3], 3)
        new_human[:,:,:,0] = F.pad(human[:,:,:,0], location, "constant", value=0)
        new_human[:,:,:,1] = F.pad(human[:,:,:,1], location, "constant", value=0)
        new_human[:,:,:,2] = F.pad(human[:,:,:,2], location, "constant", value=0)

        final = torch.where((new_contour == 0).cpu(), new_bg.cpu(), new_human.cpu())
        return final

    def render_mesh_on_bg_batch(self, mesh, bg_imgs, num_angles,  location=None, x_translation=0, y_translation=0):
        num_bgs = bg_imgs.shape[0]

        images = self.renderer(mesh) # (num_angles, 416, 416, 4)
        images = torch.cat(num_bgs*[images], dim=0) # (num_angles * num_bgs, 416, 416, 4)

        bg_shape = bg_imgs.shape

        # bg_imgs: (num_bgs, 3, 416, 416) -> (num_bgs, 416, 416, 3)
        bg_imgs = bg_imgs.permute(0, 2, 3, 1)

        # bg_imgs: (num_bgs, 416, 416, 3) -> (num_bgs * num_angles, 416, 416, 3)
        bg_imgs = bg_imgs.repeat_interleave(repeats=num_angles, dim=0)

        # human: RGB channels of render (num_angles * num_bgs, 416, 416, 3)
        human = images[:, ..., :3]
        human_size = self.renderer.rasterizer.raster_settings.image_size

        if location is None:
            dH = bg_shape[2] - human_size
            dW = bg_shape[3] - human_size
            location = (
                dW // 2 + x_translation,
                dW - (dW // 2) - x_translation,
                dH // 2 + y_translation,
                dH - (dH // 2) - y_translation
            )

        contour = torch.where((human == 1), torch.zeros(1).to(self.device), torch.ones(1).to(self.device))
        new_contour = torch.zeros(num_angles * num_bgs, bg_shape[2], bg_shape[3], 3, device=self.device)
        
        new_contour[:,:,:,0] = F.pad(contour[:,:,:,0], location, "constant", value=0)
        new_contour[:,:,:,1] = F.pad(contour[:,:,:,1], location, "constant", value=0)
        new_contour[:,:,:,2] = F.pad(contour[:,:,:,2], location, "constant", value=0)

        new_human = torch.zeros(num_angles * num_bgs, bg_shape[2], bg_shape[3], 3, device=self.device)
        new_human[:,:,:,0] = F.pad(human[:,:,:,0], location, "constant", value=0)
        new_human[:,:,:,1] = F.pad(human[:,:,:,1], location, "constant", value=0)
        new_human[:,:,:,2] = F.pad(human[:,:,:,2], location, "constant", value=0)

        # output: (num_angles * num_bgs, 416, 416, 3)
        final = torch.where((new_contour == 0), bg_imgs, new_human)
        return final

    
#     # Faster RCNN setup to match the checkpoints
#     Config.setup(image_min_side=800, image_max_side=1333, anchor_sizes="[64, 128, 256, 512]", rpn_post_nms_top_n=1000)

#     # Uncomment this to manually run faster rcnn test on a trained patch
#     # trainer.test_patch_faster_rcnn(
#     #   path_to_checkpoint='/content/drive/My Drive/3D_Logo/model-180000.pth',
#     #   dataset_name="coco2017", 
#     #   backbone_name="resnet101", 
#     #   prob_thresh=0.6)

#     if config.test_only:
#         if config.detector == 'yolov2':
#             trainer.test_patch() 
#         elif config.detector == 'faster_rcnn':
#             trainer.test_patch_faster_rcnn(
#                 path_to_checkpoint='faster_rcnn/model-180000.pth',
#                 dataset_name="coco2017", 
#                 backbone_name="resnet101", 
#                 prob_thresh=0.6
#             )
#     # else:
#     #     if config.detector == 'yolov2':
#     #         trainer.attack() 
#     #     elif config.detector == 'faster_rcnn':
#     #         trainer.attack_faster_rcnn()










 # save_image(reshape_img[0].cpu().detach(), "TEST_RENDER.png")

            # ep_loss = 0.0
            # ep_acc = 0.0
            # n = 0.0

            # for mesh in self.mesh_dataset:
            #     mesh = mesh.extend(self.num_angles_train)

            #     for bg_batch in train_bgs:

            #         self.optimizer.zero_grad()
                    
            #         # Run detection model on images
            #         output = self.dnet(reshape_img)

            #         d_loss = dis_loss(output, self.dnet.num_classes, self.dnet.anchors, self.dnet.num_anchors, 0)
            #         acc_loss = calc_acc(output, self.dnet.num_classes, self.dnet.num_anchors, 0)

            #         # tv = total_variation(self.patch[0])
            #         # tv_loss = tv * 2.5
                    
            #         loss = d_loss

            #         ep_loss += loss.item()
            #         ep_acc += acc_loss.item()
                    
            #         n += bg_batch.shape[0]

            #         loss.backward(retain_graph=True)
            #         optimizer.step()
        
            # print('epoch={} loss={} success_rate={}'.format(
            # epoch, 
            # (ep_loss / n), 
            # (ep_acc / n) / self.num_angles_train)
            # )
