'''
@author: niceliu
@contact: nicehuster@gmail.com
@file: basic.py
@time: 1/1/19 9:58 PM
@desc:
'''

from .cpm_vgg16 import cpm_vgg16
from .cpm import cpm
def obtain_model(configure, points):
  if configure.arch == 'cpm_vgg16':
    net = cpm_vgg16(configure, points)
  elif configure.arch == 'cpm':
    net = cpm(configure, points)
    print('using only CPM features!')
  else:
    raise TypeError('Unkonw type : {:}'.format(configure.arch))
  return net