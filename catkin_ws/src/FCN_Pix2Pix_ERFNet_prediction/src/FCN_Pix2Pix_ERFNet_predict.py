#!/usr/bin/env python
import numpy as np
import cv2
import roslib
import rospy
import tf
import struct
import math
import time
import os
import math
import time
import sys
import PIL
import rospkg
import pandas as pd
import scipy.misc
import random
from sensor_msgs import point_cloud2
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo, CompressedImage, PointCloud2, PointField
from geometry_msgs.msg import PoseArray, PoseStamped, Point
from nav_msgs.msg import Path
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Header
import message_filters

from torchvision import transforms, utils, datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import models
from torchvision.models.vgg import VGG
from sklearn.metrics import confusion_matrix
from artifact_search.msg import arti_input
#from arti_input import arti_input
# import torch.nn.init as init
# import torchvision


class FCN16s(nn.Module):

    def __init__(self, pretrained_net, n_class):
        super(FCN16s, self).__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(
            512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(
            512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(
            256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(
            128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(
            64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        output = self.pretrained_net(x)
        x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)
        x4 = output['x4']  # size=(N, 512, x.H/16, x.W/16)

        # size=(N, 512, x.H/16, x.W/16)
        score = self.relu(self.deconv1(x5))
        # element-wise add, size=(N, 512, x.H/16, x.W/16)
        score = self.bn1(score + x4)
        # size=(N, 256, x.H/8, x.W/8)
        score = self.bn2(self.relu(self.deconv2(score)))
        # size=(N, 128, x.H/4, x.W/4)
        score = self.bn3(self.relu(self.deconv3(score)))
        # size=(N, 64, x.H/2, x.W/2)
        score = self.bn4(self.relu(self.deconv4(score)))
        score = self.bn5(self.relu(self.deconv5(score))
                         )  # size=(N, 32, x.H, x.W)
        # size=(N, n_class, x.H/1, x.W/1)
        score = self.classifier(score)

        return score


class VGGNet(VGG):
    def __init__(self, cfg, pretrained=True, model='vgg16', requires_grad=True, remove_fc=True, show_params=False):
        super(VGGNet, self).__init__(self.make_layers(cfg[model]))
        ranges = {
            'vgg11': ((0, 3), (3, 6),  (6, 11),  (11, 16), (16, 21)),
            'vgg13': ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25)),
            'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),
            'vgg19': ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37))
        }
        self.ranges = ranges[model]

        if pretrained:
            exec("self.load_state_dict(models.%s(pretrained=True).state_dict())" % model)

        if not requires_grad:
            for param in super().parameters():
                param.requires_grad = False

        if remove_fc:  # delete redundant fully-connected layer params, can save memory
            del self.classifier

        if show_params:
            for name, param in self.named_parameters():
                print(name, param.size())

    def forward(self, x):
        output = {}

        # get the output of each maxpooling layer (5 maxpool in VGG net)
        for idx in range(len(self.ranges)):
            for layer in range(self.ranges[idx][0], self.ranges[idx][1]):
                x = self.features[layer](x)
            output["x%d" % (idx+1)] = x
        return output

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d,
                               nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
class DownsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv = nn.Conv2d(ninput, noutput-ninput, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        return F.relu(output)
    

class non_bottleneck_1d (nn.Module):
    def __init__(self, chann, dropprob, dilated):        
        super().__init__()

        self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1,0), bias=True)

        self.conv1x3_1 = nn.Conv2d(chann, chann, (1,3), stride=1, padding=(0,1), bias=True)

        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1*dilated,0), bias=True, dilation = (dilated,1))

        self.conv1x3_2 = nn.Conv2d(chann, chann, (1,3), stride=1, padding=(0,1*dilated), bias=True, dilation = (1, dilated))

        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)
        

    def forward(self, input):

        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)
        
        return F.relu(output+input)    #+input = identity (residual connection)


class Encoder(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.initial_block = DownsamplerBlock(9,16)

        self.layers = nn.ModuleList()

        self.layers.append(DownsamplerBlock(16,64))

        for x in range(0, 5):    #5 times
            self.layers.append(non_bottleneck_1d(64, 0.03, 1)) 

        self.layers.append(DownsamplerBlock(64,128))

        for x in range(0, 2):    #2 times
            self.layers.append(non_bottleneck_1d(128, 0.3, 2))
            self.layers.append(non_bottleneck_1d(128, 0.3, 4))
            self.layers.append(non_bottleneck_1d(128, 0.3, 8))
            self.layers.append(non_bottleneck_1d(128, 0.3, 16))

        #Only in encoder mode:
        self.output_conv = nn.Conv2d(128, n_class, 1, stride=1, padding=0, bias=True)

    def forward(self, input, predict=False):
        output = self.initial_block(input)

        for layer in self.layers:
            output = layer(output)

        if predict:
            output = self.output_conv(output)

        return output


class UpsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)

class Decoder (nn.Module):
    def __init__(self, n_class):
        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(UpsamplerBlock(128,64))
        self.layers.append(non_bottleneck_1d(64, 0, 1))
        self.layers.append(non_bottleneck_1d(64, 0, 1))

        self.layers.append(UpsamplerBlock(64,16))
        self.layers.append(non_bottleneck_1d(16, 0, 1))
        self.layers.append(non_bottleneck_1d(16, 0, 1))

        self.output_conv = nn.ConvTranspose2d( 16, n_class, 2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, input):
        output = input

        for layer in self.layers:
            output = layer(output)

        output = self.output_conv(output)

        return output

#ERFNet
class ERFNet(nn.Module):
    def __init__(self, n_class, encoder=None):  #use encoder to pass pretrained encoder
        super().__init__()

        if (encoder == None):
            self.encoder = Encoder(n_class)
        else:
            self.encoder = encoder
        self.decoder = Decoder(n_class)

    def forward(self, input, only_encode=False):
        if only_encode:
            return self.encoder.forward(input, predict=True)
        else:
            output = self.encoder(input)    #predict=False by default
            return self.decoder.forward(output)
#---------------------------------------------------------------------------------------------------


class FCN_Pix2Pix_PREDICT():
    def __init__(self):
        # rospy.loginfo("[%s] Initializing " %(self.node_name))
        self.bridge = CvBridge()
        self.cfg = {
            'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
        }
        self.means = np.array([103.939, 116.779, 123.68]) / \
            255.  # mean of three channels in the order of BGR
        self.h, self.w = 480, 640
        self.n_class = 5
        fcn_model_dir = "/home/alex/FCN-Pix2Pix-ERFNet_pytorch/models/FCN_Pix2Pix"   
        fcn_model_name = "FCN_Pix2Pix_batch4_epoch60_RMSprop_lr0.0001.pkl"
        erfnet_model_dir = "/home/alex/FCN-Pix2Pix-ERFNet_pytorch/models/ERFNet_dual_stream"
        erfnet_model_name = "ERFNet_batch3_epoch164_Adam_lr0.0005.pkl"

        self.vgg_model = VGGNet(
            self.cfg, requires_grad=True, remove_fc=True)
        self.fcn_model = FCN16s(
            pretrained_net=self.vgg_model, n_class=self.n_class)
        self.erfnet_model = ERFNet(n_class=self.n_class)

        use_gpu = torch.cuda.is_available()
        num_gpu = list(range(torch.cuda.device_count()))
        rospy.loginfo("Cuda available: %s", use_gpu)

        if use_gpu:
            ts = time.time()
            self.vgg_model = self.vgg_model.cuda()
            self.fcn_model = self.fcn_model.cuda()
            self.fcn_model = nn.DataParallel(
            self.fcn_model, device_ids=num_gpu)
            self.erfnet_model = self.erfnet_model.cuda()
            print("Finish cuda loading, time elapsed {}".format(time.time() - ts))
            
            fcn_state_dict = torch.load(os.path.join(fcn_model_dir, fcn_model_name))
            self.fcn_model.load_state_dict(fcn_state_dict)
            erfnet_state_dict = torch.load(os.path.join(erfnet_model_dir, erfnet_model_name))
            erfnet_model.load_state_dict(erfnet_state_dict)

            self.mask1 = np.zeros((self.h, self.w))
            self.MAXAREA = 18000
            self.MINAREA = 1000
            self.brand = ['','extinguisher', 'backpack', 'drill', 'survivor']
            rospy.loginfo("Node ready!")

            # -------point cloud with color-------
            self.depth_sub = message_filters.Subscriber(
                "/camera/aligned_depth_to_color/image_raw", Image)
            self.image_sub = message_filters.Subscriber(
                "/camera/color/image_raw", Image)
            self.ts = message_filters.ApproximateTimeSynchronizer(
                [self.image_sub, self.depth_sub], 5, 5)
            self.ts.registerCallback(self.img_cb)
            # ------------------------------------

            # self.pc_pub = rospy.Publisher("/pointcloud2_transformed", PointCloud2, queue_size=1)
            self.rgb_pub = rospy.Publisher("/rgb_img", Image, queue_size=1)
            self.image_pub = rospy.Publisher("/generate_dp", Image, queue_size=1)
            self.msg_pub = rospy.Publisher("/mask_to_point", bb_input, queue_size = 1)
            self.points = []
            self.time_total = 0
            self.time_count = 0
            rospy.loginfo("Start Generating depth image")


    def img_cb(self, rgb_data, depth_data):
        cv_image = self.bridge.imgmsg_to_cv2(rgb_data, "bgr8")
        cv_depthimage = self.bridge.imgmsg_to_cv2(depth_data, "16UC1")
        now = rospy.get_time()
        generate_img = self.predict(cv_image)
        self.time_total = self.time_total + rospy.get_time() - now
        self.time_count = self.time_count + 1
        rospy.loginfo("Average time : %f , Hz : %f ", self.time_total /
                      self.time_count, self.time_count/self.time_total)
        
        msg = arti_input
        msg.image = rgb_data
        msg.depth = depth_data
        msg.mask = self.bridge.cv2_to_imgmsg(generate_img, "8UC1")
        self.msg_pub.publish(msg)
        # self.image_pub.publish(self.bridge.cv2_to_imgmsg(generate_img, "8UC1"))
        # self.image_pub.publish(self.bridge.cv2_to_imgmsg(self.generate_img, "8UC1"))

    def predict(self, img):
        origin  = img
        img = img[:, :, ::-1]  # switch to BGR
        
        img = np.transpose(img, (2, 0, 1)) / 255.
        img[0] -= self.means[0]
        img[1] -= self.means[1]
        img[2] -= self.means[2]

		# convert to tensor
        img = img[np.newaxis,:]
        img = torch.from_numpy(img.copy()).float() 
        thermals = 
        
        output = self.fcn_model(img)
        new_input = torch.cat((img,output,thermals), dim=1, out=None)
        new_output = self.erfnet_model(new_input)
        new_output = new_output.data.cpu().numpy()
        
        N, _, h, w = output.shape
        mask = output.transpose(0, 2, 3, 1).reshape(-1, self.n_class).argmax(axis = 1).reshape(N, h, w)[0]
        
        rospy.loginfo("Predict time : %f", rospy.get_time() - now)
        
        mask = np.asarray(mask, np.uint8)
        mask1 = mask.copy()
        mask1[mask1 > 0] = 255
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(mask1, "8UC1"))
        self.rgb_pub.publish(self.bridge.cv2_to_imgmsg(origin, "bgr8"))
        return mask


if __name__ == '__main__':
    rospy.init_node('FCN_Pix2Pix_PREDICT')
    foo = FCN_Pix2Pix_PREDICT()
    rospy.spin()
