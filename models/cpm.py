# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import division
import time, math
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from copy import deepcopy
from .model_utils import get_parameters,find_tensor_peak_batch,weights_init_cpm


class CPM(nn.Module):
  def __init__(self, config, pts_num):
    super(CPM, self).__init__()

    self.config = deepcopy(config)
    self.downsample = 8
    self.pts_num = pts_num
  
    self.features = nn.Sequential(
          nn.Conv2d(  3,  64, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
          nn.Conv2d( 64,  64, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=2, stride=2),
          nn.Conv2d( 64, 128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
          nn.Conv2d(128, 128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=2, stride=2),
          nn.Conv2d(128, 256, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
          nn.Conv2d(256, 256, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
          nn.Conv2d(256, 256, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=2, stride=2),
          nn.Conv2d(256, 512, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
          nn.Conv2d(512, 512, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
          nn.Conv2d(512, 512, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True))


    self.CPM_feature = nn.Sequential(
          nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True), #CPM_1
          nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True)) #CPM_2

    assert self.config.stages >= 1, 'stages of cpm must >= 1 not : {:}'.format(self.config.stages)
    stage1 = nn.Sequential(
          nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
          nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
          nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
          nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
          nn.Conv2d(128, 512, kernel_size=1, padding=0), nn.ReLU(inplace=True),
          nn.Conv2d(512, pts_num, kernel_size=1, padding=0))
    stages = [stage1]
    for i in range(1, self.config.stages):
      stagex = nn.Sequential(
          nn.Conv2d(128+pts_num, 128, kernel_size=7, dilation=1, padding=3), nn.ReLU(inplace=True),
          nn.Conv2d(128,         128, kernel_size=7, dilation=1, padding=3), nn.ReLU(inplace=True),
          nn.Conv2d(128,         128, kernel_size=7, dilation=1, padding=3), nn.ReLU(inplace=True),
          nn.Conv2d(128,         128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
          nn.Conv2d(128,         128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
          nn.Conv2d(128,         128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
          nn.Conv2d(128,         128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
          nn.Conv2d(128,         128, kernel_size=1, padding=0), nn.ReLU(inplace=True),
          nn.Conv2d(128,     pts_num, kernel_size=1, padding=0))
      stages.append( stagex )
    self.stages = nn.ModuleList(stages)

  def specify_parameter(self, base_lr, base_weight_decay):
    params_dict = [ 
                    {'params': get_parameters(self.CPM_feature, bias=False), 'lr': base_lr  , 'weight_decay': base_weight_decay},
                    {'params': get_parameters(self.CPM_feature, bias=True ), 'lr': base_lr*2, 'weight_decay': 0},
                  ]
    for stage in self.stages:
      params_dict.append( {'params': get_parameters(stage, bias=False), 'lr': base_lr*4, 'weight_decay': base_weight_decay} )
      params_dict.append( {'params': get_parameters(stage, bias=True ), 'lr': base_lr*8, 'weight_decay': 0} )
    return params_dict

  # return : cpm-stages, locations
  def forward(self, inputs):
    assert inputs.dim() == 4, 'This model accepts 4 dimension input tensor: {}'.format(inputs.size())
    batch_size, feature_dim = inputs.size(0), inputs.size(1)
    batch_cpms, batch_locs, batch_scos = [], [], []

    feature  = self.features(inputs)

    print("----------")
    print("inputs:", inputs.shape)
    print("feature:", feature.shape)    

    xfeature = self.CPM_feature(feature)
    print("xfeature:", xfeature.shape)

    for i in range(self.config.stages):
      if i == 0: cpm = self.stages[i]( xfeature )
      else:      cpm = self.stages[i]( torch.cat([xfeature, batch_cpms[i-1]], 1) )

      print(i, cpm.shape)

      batch_cpms.append( cpm )

    # The location of the current batch
    for ibatch in range(batch_size):
      batch_location, batch_score = find_tensor_peak_batch(batch_cpms[-1][ibatch], self.config.argmax, self.downsample)
      batch_locs.append( batch_location )
      batch_scos.append( batch_score )
    batch_locs, batch_scos = torch.stack(batch_locs), torch.stack(batch_scos)

    return batch_cpms, batch_locs, batch_scos


class CPM_ori(nn.Module):
  def __init__(self, config, pts_num):
    super(CPM_ori, self).__init__()

    self.config = deepcopy(config)
    self.downsample = 8
    self.pts_num = pts_num

    stage1 = nn.Sequential(
      nn.Conv2d( 3, 128, kernel_size=9, padding=4), nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
      nn.Conv2d(128, 128, kernel_size=9, padding=4), nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
      nn.Conv2d(128, 128, kernel_size=9, padding=4), nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
      nn.Conv2d(128, 32, kernel_size=5, padding=2), nn.ReLU(inplace=True),
      nn.Conv2d(32, 512, kernel_size=9, padding=4), nn.ReLU(inplace=True),
      nn.Conv2d(512, 512, kernel_size=1), nn.ReLU(inplace=True),
      nn.Conv2d(512, pts_num , kernel_size=1)
    )  
    stages = [stage1]

    self.middle = nn.Sequential(
      nn.Conv2d( 3, 128, kernel_size=9, padding=4), nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
      nn.Conv2d(128, 128, kernel_size=9, padding=4), nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
      nn.Conv2d(128, 128, kernel_size=9, padding=4), nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
      nn.Conv2d(128, 32, kernel_size=5, padding=2), nn.ReLU(inplace=True)
    )

    for i in range(1, self.config.stages):
      stagex = nn.Sequential(
        nn.Conv2d(32 + pts_num, 128, kernel_size=11, padding=5), nn.ReLU(inplace=True),
        nn.Conv2d(128,          128, kernel_size=11, padding=5), nn.ReLU(inplace=True),
        nn.Conv2d(128,          128, kernel_size=11, padding=5), nn.ReLU(inplace=True),
        nn.Conv2d(128,          128, kernel_size=11, padding=5), nn.ReLU(inplace=True),
        nn.Conv2d(128,          128, kernel_size=1, padding=0), nn.ReLU(inplace=True),
        nn.Conv2d(128,      pts_num, kernel_size=1, padding=0)
      )
      stages.append( stagex )
    self.stages = nn.ModuleList(stages)

  # def specify_parameter(self, base_lr, base_weight_decay):
  #   params_dict = [ 
  #                   {'params': get_parameters(self.middle, bias=False), 'lr': base_lr  , 'weight_decay': base_weight_decay},
  #                   {'params': get_parameters(self.middle, bias=True ), 'lr': base_lr*2, 'weight_decay': 0},
  #                 ]
  #   for stage in self.stages:
  #     params_dict.append( {'params': get_parameters(stage, bias=False), 'lr': base_lr*4, 'weight_decay': base_weight_decay} )
  #     params_dict.append( {'params': get_parameters(stage, bias=True ), 'lr': base_lr*8, 'weight_decay': 0} )
  #   return params_dict    

  def forward(self, inputs):
    assert inputs.dim() == 4, 'This model accepts 4 dimension input tensor: {}'.format(inputs.size())
    batch_size, feature_dim = inputs.size(0), inputs.size(1)
    batch_cpms, batch_locs, batch_scos = [], [], []

    cpm = self.stages[0](inputs)
    batch_cpms.append(cpm)

    middle = self.middle(inputs)
    for i in range(1, self.config.stages):
      cpm = self.stages[1]( torch.cat([middle, batch_cpms[i-1]], dim=1))
      batch_cpms.append(cpm)    

    # The location of the current batch
    for ibatch in range(batch_size):
      batch_location, batch_score = find_tensor_peak_batch(batch_cpms[-1][ibatch], self.config.argmax, self.downsample)
      batch_locs.append( batch_location )
      batch_scos.append( batch_score )
    batch_locs, batch_scos = torch.stack(batch_locs), torch.stack(batch_scos)

    return batch_cpms, batch_locs, batch_scos

def cpm(config, pts):

  print ('Initialize cpm with configure : {}'.format(config))
  #model = CPM(config, pts)
  model = CPM_ori(config, pts)
  model.apply(weights_init_cpm)

  return model



