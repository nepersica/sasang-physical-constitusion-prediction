# **Categories - 15 body parts
color = ['000000', '1beaac', '27b73c', '56bcec', '6600cc', '0033cc', 'cc66ff',
         '328aa0', 'f1513c', 'fea518', 'e9bdff', 'f8d016', '95177f', 'fba39b', '9999ff']
part_KOR = ['배경', '머리', '몸통', '위왼팔', '위오른팔', '아래왼팔', '아래오른팔',
            '왼손', '오른손', '위왼쪽다리', '위오른쪽다리', '아래왼쪽다리', '아래오른쪽다리', '왼발', '오른발']
part_ENG = ['background,' 'head', 'torso', 'leftupperarm' ,'rightupperarm', 'leftforearm', 'rightforearm',
           'lefthand', 'righthand', 'leftthigh', 'rightthigh', 'leftshank', 'rightshank', 'leftfoot', 'rightfoot']
human_parT_ID = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
seg_num = 1

import numpy as np
import scipy.io
import torch

image_size = 256
channel = 3
batch_size = 16

grad_clip = 5.
