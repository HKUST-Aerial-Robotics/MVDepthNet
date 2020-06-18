import time
import cv2
import pickle
import numpy as np
from numpy.linalg import inv

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch import Tensor

from depthNet_model import depthNet
from visualize import *

with open('sample_data.pkl', 'rb') as fp:
    sample_datas = pickle.load(fp, encoding='latin1')

# model
depthnet = depthNet()
model_data = torch.load('opensource_model.pth.tar')
depthnet.load_state_dict(model_data['state_dict'])
depthnet = depthnet.cuda()
cudnn.benchmark = True
depthnet.eval()

# for warp the image to construct the cost volume
pixel_coordinate = np.indices([320, 256]).astype(np.float32)
pixel_coordinate = np.concatenate(
    (pixel_coordinate, np.ones([1, 320, 256])), axis=0)
pixel_coordinate = np.reshape(pixel_coordinate, [3, -1])

cv2.namedWindow('result')
cv2.moveWindow('result', 200, 200)

for this_sample in sample_datas:

    # get data
    with torch.no_grad():
        depth_image_cuda = Tensor(this_sample['depth_image']).cuda()
        depth_image_cuda = Variable(depth_image_cuda)

        left_image_cuda = Tensor(this_sample['left_image']).cuda()
        left_image_cuda = Variable(left_image_cuda)

        right_image_cuda = Tensor(this_sample['right_image']).cuda()
        right_image_cuda = Variable(right_image_cuda)

    left_in_right_T = this_sample['left2right'][0:3, 3]
    left_in_right_R = this_sample['left2right'][0:3, 0:3]
    K = this_sample['K']
    K_inverse = inv(K)
    KRK_i = K.dot(left_in_right_R.dot(K_inverse))
    KRKiUV = KRK_i.dot(pixel_coordinate)
    KT = K.dot(left_in_right_T)
    KT = np.expand_dims(KT, -1)
    KT = np.expand_dims(KT, 0)
    KT = KT.astype(np.float32)
    KRKiUV = KRKiUV.astype(np.float32)
    KRKiUV = np.expand_dims(KRKiUV, 0)
    KRKiUV_cuda_T = Tensor(KRKiUV).cuda()
    KT_cuda_T = Tensor(KT).cuda()

    predict_depths = depthnet(left_image_cuda, right_image_cuda, KRKiUV_cuda_T, KT_cuda_T)

    # visualize the results
    np_left = np2Img(np.squeeze(this_sample['left_image']), True)
    np_right = np2Img(np.squeeze(this_sample['right_image']), True)
    idepth = np.squeeze(predict_depths[0].cpu().data.numpy())
    gt_idepth = 1.0 / np.clip(np.squeeze(this_sample['depth_image']), 0.1, 50.0)
    # invalid_mask is used to mask invalid values in RGB-D images
    invalid_mask = gt_idepth > 5.0
    invalid_mask = np.expand_dims(invalid_mask, -1)
    invalid_mask = np.repeat(invalid_mask, 3, axis=2)
    np_gtdepth = np2Depth(gt_idepth, invalid_mask)
    np_depth = np2Depth(idepth, np.zeros(invalid_mask.shape, dtype=bool))
    result_image = np.concatenate(
        (np_left, np_right, np_gtdepth, np_depth), axis=1)
    cv2.imshow("result", result_image)
    if cv2.waitKey(1000) == 27:
        break