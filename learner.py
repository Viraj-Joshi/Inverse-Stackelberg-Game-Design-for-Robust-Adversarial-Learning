import os
import time
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T

from torch.utils.data import DataLoader, Dataset

from darknet import Darknet
from loss import dis_loss, dis_loss2, calc_acc

import random

from Extragradient import Extragradient
from torch import autograd

from load_data import *
from tqdm import tqdm
from region_loss import RegionLoss

class Learner():
    def __init__(self, config, device):
        self.config = config
        self.device = device
        
        # Yolo model:
        self.dnet = Darknet(self.config.cfgfile)

        self.dnet.load_weights(self.config.weightfile)
        self.dnet.print_network()
        self.dnet = self.dnet.to(self.device)

        self.dnet_optimizer = torch.optim.Adam(self.dnet.parameters(), lr=1e-3)
        self.dnet_optimizer = Extragradient(self.dnet_optimizer, self.dnet.parameters())
       
        self.BCE = torch.nn.BCELoss(reduction='none').to(device)

        self.train_loader = torch.utils.data.DataLoader(InriaDataset(self.config.img_dir, self.config.lab_dir, 14, self.dnet.height,
                                                                shuffle=True),
                                                   batch_size=self.config.batch_size,
                                                   shuffle=True,
                                                   num_workers=1)
        self.region_loss = RegionLoss(self.dnet.num_classes, self.dnet.anchors, self.dnet.num_anchors)
        
    
    def evaluate(self, images):
        images_as_tensor = torch.Tensor(images[0].cpu()).to(self.device)
        for i in range(1,len(images)):
            images_as_tensor = torch.cat([images_as_tensor,images[i]],axis = 0)
        images_as_tensor.retain_grad()

        output = self.dnet(images_as_tensor)

        d_loss = dis_loss(output, self.dnet.num_classes, self.dnet.anchors, self.dnet.num_anchors, 0)
        number_of_detections_failed = calc_acc(output, self.dnet.num_classes, self.dnet.num_anchors, 0)
        return d_loss, number_of_detections_failed/images_as_tensor.shape[0]

    def evaluate2(self, images):
        images_as_tensor = torch.Tensor(images[0].cpu()).to(self.device)
        for i in range(1,len(images)):
            images_as_tensor = torch.cat([images_as_tensor,images[i]],axis = 0)
        images_as_tensor.retain_grad()

        output = self.dnet(images_as_tensor)
        # print(min(output.flatten()),max(output.flatten()))

        target_confs = dis_loss2(output, self.dnet.num_classes, self.dnet.anchors, self.dnet.num_anchors, 0)
        number_of_detections_failed = calc_acc(output, self.dnet.num_classes, self.dnet.num_anchors, 0)
        return target_confs, number_of_detections_failed/images_as_tensor.shape[0]
    
    def update(self,adv_images,iteration,train_logger):
        et0 = time.time()
        print('')
        print("LEARNER TURN")
        epoch_length = len(self.train_loader)

        gamma = 2
        alpha = .25
        eps = 1e-7
        
        # for batch_index, (img_batch, lab_batch) in tqdm(enumerate(self.train_loader), desc=f'Running epoch {iteration}', total=epoch_length):
        #     img_batch = img_batch.to(self.device)
        #     lab_batch = lab_batch.to(self.device)

        #     output = self.dnet(img_batch)
        #     target_confs = dis_loss2(output, self.dnet.num_classes, self.dnet.anchors, self.dnet.num_anchors, 0)
        #     # region_loss.seen = region_loss.seen + img_batch.data.size(0)
        #     # self.region_loss.seen  += img_batch.data.size(0)
        #     # self.dnet_optimizer.zero_grad()
        #     # loss = self.region_loss(output, lab_batch)

        #     gt = torch.ones(target_confs.shape[0]).float().to(self.device)

        #     bce_loss = self.BCE(target_confs,gt).clamp(0,1)
        #     # pt = torch.exp(-bce_loss)
        #     focal_loss = (alpha*((1-bce_loss)**gamma) * bce_loss).mean()

        #     grad_model = autograd.grad(focal_loss, self.dnet.parameters())
        #     for p, g in zip(self.dnet.parameters(), grad_model):
        #         p.grad = g
        #     # loss.backward()
        #     self.dnet_optimizer.step()

        #     del img_batch,lab_batch

        target_confs,success_rate = self.evaluate2(adv_images)
        print(' MODEL ACC: ', 1-success_rate.item())
        

        # print(conf_mask)
        gt = torch.ones(target_confs.shape[0]).float().to(self.device)

        bce_loss = self.BCE(target_confs,gt).clamp(0,1)
        # pt = torch.exp(-bce_loss)
        focal_loss = (alpha*((1-bce_loss)**gamma) * bce_loss).mean()
        
        train_logger.add_scalar('learner loss', focal_loss.item(), iteration)
        grad_model = autograd.grad(focal_loss, self.dnet.parameters())
        for p, g in zip(self.dnet.parameters(), grad_model):
            p.grad = g

        self.dnet_optimizer.step()
        et1 = time.time()

        print('LEARNER TIME: ', et1-et0)
        print('')


        if iteration % 10 == 0:
            # state = {
            #     'epoch': iteration,
            #     'state_dict': self.dnet.state_dict(),
            #     'optimizer': self.dnet_optimizer.state_dict()
            # }
            self.dnet.save_weights(f"YOLOv2_Weights/YOLOv2_{iteration}.weights")

    
