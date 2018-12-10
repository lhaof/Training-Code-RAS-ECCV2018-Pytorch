from __future__ import division
import sys
caffe_root = '../caffe_dss-master/'
sys.path.insert(0, caffe_root + 'python')
import caffe
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np

class rasnet(nn.Module):
    def __init__(self):
        super(rasnet, self).__init__()

        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True) # conv1_1
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True) # conv1_2
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True) 

        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1, bias=True) # conv2_1
        self.relu2_1 = nn.ReLU(inplace=True) 
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True) 

        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.relu3_1 = nn.ReLU(inplace=True) 
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.relu3_2 = nn.ReLU(inplace=True) 
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True) 

        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.relu4_1 = nn.ReLU(inplace=True) 
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.relu4_2 = nn.ReLU(inplace=True) 
        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) 

        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.relu5_1 = nn.ReLU(inplace=True) 
        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.relu5_2 = nn.ReLU(inplace=True) 
        self.conv5_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) 

        self.conv1_dsn6 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0, dilation=1, bias=True)
        self.conv2_dsn6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, stride=1, padding=2, dilation=1, bias=True) 
        self.relu1_dsn6 = nn.ReLU(inplace=True) 
        self.conv3_dsn6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, stride=1, padding=2, dilation=1, bias=True) 
        self.relu2_dsn6 = nn.ReLU(inplace=True) 
        self.conv4_dsn6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, stride=1, padding=2, dilation=1, bias=True) 
        self.relu3_dsn6 = nn.ReLU(inplace=True) 
        self.conv5_dsn6 = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0, dilation=1, bias=True) 

        self.upsample32_dsn6 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=64, stride=32, padding=0, bias=True)
        self.sigmoid_score6 = nn.Sigmoid()

        self.upsample2_dsn5 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=4, stride=2, padding=0, bias=True)
        self.sigmoid_dsn5 = nn.Sigmoid()
        
        self.conv1_dsn5 = nn.Conv2d(in_channels=512, out_channels=64, kernel_size=1, stride=1, padding=0, dilation=1, bias=True) 
        self.conv2_dsn5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True) 
        self.relu1_dsn5 = nn.ReLU(inplace=True) 
        self.conv3_dsn5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True) 
        self.relu2_dsn5 = nn.ReLU(inplace=True) 
        self.conv4_dsn5 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, dilation=1, bias=True) 

        self.upsample16_dsn5 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=32, stride=16, padding=0, bias=True) 
        self.sigmoid_score5 = nn.Sigmoid()

        self.upsample2_dsn4 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=4, stride=2, padding=0, bias=True)
        self.sigmoid_dsn4 = nn.Sigmoid()
        
        self.conv1_dsn4 = nn.Conv2d(in_channels=512, out_channels=64, kernel_size=1, stride=1, padding=0, dilation=1, bias=True) 
        self.conv2_dsn4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True) 
        self.relu1_dsn4 = nn.ReLU(inplace=True) 
        self.conv3_dsn4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True) 
        self.relu2_dsn4 = nn.ReLU(inplace=True) 
        self.conv4_dsn4 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, dilation=1, bias=True) 

        self.upsample8_dsn4 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=16, stride=8, padding=0, bias=True) 
        self.sigmoid_score4 = nn.Sigmoid()

        self.upsample2_dsn3 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=4, stride=2, padding=0, bias=True)
        self.sigmoid_dsn3 = nn.Sigmoid()
        
        self.conv1_dsn3 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, stride=1, padding=0, dilation=1, bias=True) 
        self.conv2_dsn3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True) 
        self.relu1_dsn3 = nn.ReLU(inplace=True) 
        self.conv3_dsn3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True) 
        self.relu2_dsn3 = nn.ReLU(inplace=True) 
        self.conv4_dsn3 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, dilation=1, bias=True) 

        self.upsample4_dsn3 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=8, stride=4, padding=0, bias=True) 
        self.sigmoid_score3 = nn.Sigmoid()

        self.upsample2_1_dsn2 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=4, stride=2, padding=0, bias=True)
        self.sigmoid_dsn2 = nn.Sigmoid()
        
        self.conv1_dsn2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0, dilation=1, bias=True) 
        self.conv2_dsn2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True) 
        self.relu1_dsn2 = nn.ReLU(inplace=True) 
        self.conv3_dsn2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True) 
        self.relu2_dsn2 = nn.ReLU(inplace=True) 
        self.conv4_dsn2 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, dilation=1, bias=True) 

        self.upsample2_2_dsn2 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=4, stride=2, padding=0, bias=True) 
        self.sigmoid_score2 = nn.Sigmoid()

        self.sigmoid_dsn1 = nn.Sigmoid()

        self.conv1_dsn1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0, dilation=1, bias=True) 
        self.conv2_dsn1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True) 
        self.relu1_dsn1 = nn.ReLU(inplace=True) 
        self.conv3_dsn1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True) 
        self.relu2_dsn1 = nn.ReLU(inplace=True) 
        self.conv4_dsn1 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, dilation=1, bias=True) 

        self.sigmoid_score1 = nn.Sigmoid()

    ###############################################
    # In Caffe's Crop_Layer, when the value of offset is default, it will set offset=0 rather than do a center_crop operation
    ###############################################
    def crop(self, x, dsn):
        height, width = x.shape[-2], x.shape[-1]
        # offset = 0 crop
        return dsn[:, :, :height, :width]

        # center crop
        # crop_h = torch.FloatTensor([dsn.shape[-2]]).sub(height).div(-2)
        # crop_w = torch.FloatTensor([dsn.shape[-1]]).sub(width).div(-2)

        # return F.pad(dsn, [
        #         crop_w.ceil().int().item(), crop_w.floor().int().item(),
        #         crop_h.ceil().int().item(), crop_h.floor().int().item()
        #         ])

        # fuse
        # concat = torch.cat((crop_score_dsn1_up, crop_score_dsn2_up, crop_score_dsn3_up, crop_score_dsn4_up, crop_score_dsn5_up, crop_score_dsn6_up), dim=1)
        # new_score_weighting = self.conv_fuse(concat)
        # return new_score_weighting, crop_score_dsn1_up, crop_score_dsn2_up, crop_score_dsn3_up, crop_score_dsn4_up, crop_score_dsn5_up, crop_score_dsn6_up


    def forward(self, data):

        conv1_1 = self.conv1_1(data)
        conv1_1 = self.relu1_1(conv1_1)
        conv1_2 = self.conv1_2(conv1_1)
        conv1_2 = self.relu1_2(conv1_2)
        pool1 = self.pool1(conv1_2)

        conv2_1 = self.conv2_1(pool1)
        conv2_1 = self.relu2_1(conv2_1)
        conv2_2 = self.conv2_2(conv2_1)
        conv2_2 = self.relu2_2(conv2_2)
        pool2 = self.pool2(conv2_2)

        conv3_1 = self.conv3_1(pool2)
        conv3_1 = self.relu3_1(conv3_1)
        conv3_2 = self.conv3_2(conv3_1)
        conv3_2 = self.relu3_2(conv3_2)
        conv3_3 = self.conv3_3(conv3_2)
        conv3_3 = self.relu3_3(conv3_3)
        pool3 = self.pool3(conv3_3)

        conv4_1 = self.conv4_1(pool3)
        conv4_1 = self.relu4_1(conv4_1)
        conv4_2 = self.conv4_2(conv4_1)
        conv4_2 = self.relu4_2(conv4_2)
        conv4_3 = self.conv4_3(conv4_2)
        conv4_3 = self.relu4_3(conv4_3)
        pool4 = self.pool4(conv4_3)

        conv5_1 = self.conv5_1(pool4)
        conv5_1 = self.relu5_1(conv5_1)
        conv5_2 = self.conv5_2(conv5_1)
        conv5_2 = self.relu5_2(conv5_2)
        conv5_3 = self.conv5_3(conv5_2)
        conv5_3 = self.relu5_3(conv5_3)
        pool5 = self.pool5(conv5_3)

        # dsn6 

        conv1_dsn6 = self.conv1_dsn6(pool5) 
        conv2_dsn6 = self.conv2_dsn6(conv1_dsn6) 
        conv2_dsn6 = self.relu1_dsn6(conv2_dsn6) 
        conv3_dsn6 = self.conv3_dsn6(conv2_dsn6) 
        conv3_dsn6 = self.relu2_dsn6(conv3_dsn6) 
        conv4_dsn6 = self.conv4_dsn6(conv3_dsn6) 
        conv4_dsn6 = self.relu3_dsn6(conv4_dsn6) 
        conv5_dsn6 = self.conv5_dsn6(conv4_dsn6) 

        conv5_dsn6_up = self.upsample32_dsn6(conv5_dsn6) 
        upscore_dsn6 = self.crop(data, conv5_dsn6_up) 
        sigmoid_score6 = self.sigmoid_score6(upscore_dsn6) 

        # dsn5

        conv5_dsn6_5 = self.upsample2_dsn5(conv5_dsn6)
        weight_dsn6_5 = self.crop(conv5_3, conv5_dsn6_5) 
        sigmoid_dsn5 = self.sigmoid_dsn5(weight_dsn6_5) 
        rev_dsn5 = torch.add( torch.mul(sigmoid_dsn5, -1.0), 1.0 )
        weight_dsn5 = rev_dsn5.repeat((1, 512, 1, 1)) 
        prod1_dsn5 = conv5_3 * weight_dsn5 

        conv1_dsn5 = self.conv1_dsn5(prod1_dsn5)
        conv2_dsn5 = self.conv2_dsn5(conv1_dsn5) 
        conv2_dsn5 = self.relu1_dsn5(conv2_dsn5) 
        conv3_dsn5 = self.conv3_dsn5(conv2_dsn5) 
        conv3_dsn5 = self.relu2_dsn5(conv3_dsn5) 
        conv4_dsn5 = self.conv4_dsn5(conv3_dsn5) 
        sum_dsn5 = conv4_dsn5 + weight_dsn6_5 

        sum_dsn5_up = self.upsample16_dsn5(sum_dsn5) 
        upscore_dsn5 = self.crop(data, sum_dsn5_up) 
        sigmoid_score5 = self.sigmoid_score5(upscore_dsn5) 

        # dsn4

        sum_dsn5_4 = self.upsample2_dsn4(sum_dsn5) 
        weight_dsn5_4 = self.crop(conv4_3, sum_dsn5_4) 
        sigmoid_dsn4 = self.sigmoid_dsn4(weight_dsn5_4) 
        rev_dsn4 = torch.add( torch.mul(sigmoid_dsn4, -1.0), 1.0 ) 
        weight_dsn4 = rev_dsn4.repeat((1, 512, 1, 1))
        prod1_dsn4 = conv4_3 * weight_dsn4 

        conv1_dsn4 = self.conv1_dsn4(prod1_dsn4) 
        conv2_dsn4 = self.conv2_dsn4(conv1_dsn4) 
        conv2_dsn4 = self.relu1_dsn4(conv2_dsn4) 
        conv3_dsn4 = self.conv3_dsn4(conv2_dsn4) 
        conv3_dsn4 = self.relu2_dsn4(conv3_dsn4)
        conv4_dsn4 = self.conv4_dsn4(conv3_dsn4) 
        sum_dsn4 = conv4_dsn4 + weight_dsn5_4 
        sum_dsn4_up = self.upsample8_dsn4(sum_dsn4) 
        upscore_dsn4 = self.crop(data, sum_dsn4_up) 
        sigmoid_score4 = self.sigmoid_score4(upscore_dsn4) 

        # dsn3

        sum_dsn4_3 = self.upsample2_dsn3(sum_dsn4) 
        weight_dsn4_3 = self.crop(conv3_3, sum_dsn4_3) 
        sigmoid_dsn3 = self.sigmoid_dsn3(weight_dsn4_3) 
        rev_dsn3 = torch.add( torch.mul(sigmoid_dsn3, -1.0), 1.0 ) 
        weight_dsn3 = rev_dsn3.repeat((1, 256, 1, 1)) 
        prod1_dsn3 = conv3_3 * weight_dsn3 

        conv1_dsn3 = self.conv1_dsn3(prod1_dsn3) 
        conv2_dsn3 = self.conv2_dsn3(conv1_dsn3) 
        conv2_dsn3 = self.relu1_dsn3(conv2_dsn3) 
        conv3_dsn3 = self.conv3_dsn3(conv2_dsn3) 
        conv3_dsn3 = self.relu2_dsn3(conv3_dsn3) 
        conv4_dsn3 = self.conv4_dsn3(conv3_dsn3) 
        sum_dsn3 = conv4_dsn3 + weight_dsn4_3
        sum_dsn3_up = self.upsample4_dsn3(sum_dsn3) 
        upscore_dsn3 = self.crop(data, sum_dsn3_up) 
        sigmoid_score3 = self.sigmoid_score3(upscore_dsn3) 

        # dsn2

        sum_dsn3_2 = self.upsample2_1_dsn2(sum_dsn3)
        weight_dsn3_2 = self.crop(conv2_2, sum_dsn3_2) 
        sigmoid_dsn2 = self.sigmoid_dsn2(weight_dsn3_2)
        rev_dsn2 = torch.add( torch.mul(sigmoid_dsn2, -1.0), 1.0)
        weight_dsn2 = rev_dsn2.repeat((1, 128, 1, 1)) 
        prod1_dsn2 = conv2_2 * weight_dsn2 

        conv1_dsn2 = self.conv1_dsn2(prod1_dsn2) 
        conv2_dsn2 = self.conv2_dsn2(conv1_dsn2) 
        conv2_dsn2 = self.relu1_dsn2(conv2_dsn2) 
        conv3_dsn2 = self.conv3_dsn2(conv2_dsn2) 
        conv3_dsn2 = self.relu2_dsn2(conv3_dsn2) 
        conv4_dsn2 = self.conv4_dsn2(conv3_dsn2) 
        sum_dsn2 = conv4_dsn2 + weight_dsn3_2 
        sum_dsn2_up = self.upsample2_2_dsn2(sum_dsn2) 
        upscore_dsn2 = self.crop(data, sum_dsn2_up) 
        sigmoid_score2 = self.sigmoid_score2(upscore_dsn2) 

        # dsn1

        sigmoid_dsn1 = self.sigmoid_dsn1(upscore_dsn2) 
        rev_dsn1 = torch.add( torch.mul(sigmoid_dsn1, -1.0), 1.0)
        weight_dsn1 = rev_dsn1.repeat((1, 64, 1, 1))
        prod1_dsn1 = conv1_2 * weight_dsn1 

        conv1_dsn1 = self.conv1_dsn1(prod1_dsn1) 
        conv2_dsn1 = self.conv2_dsn1(conv1_dsn1) 
        conv2_dsn1 = self.relu1_dsn1(conv2_dsn1) 
        conv3_dsn1 = self.conv3_dsn1(conv2_dsn1) 
        conv3_dsn1 = self.relu2_dsn1(conv3_dsn1) 
        conv4_dsn1 = self.conv4_dsn1(conv3_dsn1)
        sum_dsn1 = conv4_dsn1 + upscore_dsn2 
        upscore_dsn1 = self.crop(data, sum_dsn1)
        sigmoid_score1 = self.sigmoid_score1(upscore_dsn1) 
        

        return torch.cat([upscore_dsn1, upscore_dsn2, upscore_dsn3, upscore_dsn4, upscore_dsn5, upscore_dsn6], 0)
        '''
        if self.training == True:
            return torch.cat([upscore_dsn1, upscore_dsn2, upscore_dsn3, upscore_dsn4, upscore_dsn5, upscore_dsn6], 0)
        else:
            return sigmoid_score1
        '''


    def reverse_attention(self, highfeat, lowfeat):
        n, c, h, w = lowfeat.size()
        upsample2_highfeat = self.upsample2(highfeat)
        crop_highfeat = self.crop(lowfeat, upsample2_highfeat) 
        weight = self.sigmoid(crop_highfeat) 
        reverse_weight = torch.add( torch.mul(weight, -1), 1 )
        tile_weight = reverse_weight.repeat(1, c, 1, 1) 
        weighted_lowfeat = tile_weight * lowfeat 
        return weighted_lowfeat, weight

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.normal_(m.weight.data, std=0.01)
        m.bias.data.zero_()

def load_vgg16conv_fromcaffe(net, vgg16caffemodel_pth):
    vgg16_weights = torch.load(vgg16caffemodel_pth)
    vgg16_keys = vgg16_weights.keys()
    for k in vgg16_keys:
        if k.endswith('_weight'):
            k_ = k.replace('_weight', '.weight')
            assert(k_ in net.state_dict().keys()) 
            net.state_dict()[k_].copy_(vgg16_weights[k].cpu().data)
        elif k.endswith('_bias'):
            k_ = k.replace('_bias', '.bias')
            assert(k_ in net.state_dict().keys()) 
            net.state_dict()[k_].copy_(vgg16_weights[k].cpu().data)
        else:
            print('error key: ',k)
            assert(False)
    return net 

def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)

# set parameters s.t. deconvolutional layers compute bilinear interpolation
# N.B. this is for deconvolution without groups
def interp_surgery(net):
    net_keys = net.state_dict().keys()
    layers = []
    for k in net_keys:
        if k.startswith('upsample') and k.endswith('weight'):
            layers.append(k)
        if k.startswith('upsample') and k.endswith('bias'):
            net.state_dict()[k].zero_()

    for l in layers:
        m, k, h, w = net.state_dict()[l].size()
        if m != k:
            print 'input + output channels need to be the same'
            raise
        if h != w:
            print 'filters need to be square'
            raise
        filt = upsample_filt(h)
        filt = np.reshape(filt, (1, 1, h, w))
        filt = np.tile(filt, (m, k, 1, 1))
        net.state_dict()[l].copy_(torch.from_numpy(filt))

    return net


def init_parameters(model):
    for m in model.modules():
        weights_init(m)
    return model 


def get_parameters(model):
    modules_skipped = (
        rasnet,
        nn.Sequential,
        nn.ReLU,
        nn.MaxPool2d,
        nn.Sigmoid
    )
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            yield m.weight
            yield m.bias
        elif isinstance(m, nn.ConvTranspose2d):
            # weight is frozen because it is just a bilinear upsampling
            # assert m.bias is None
            continue
        elif isinstance(m, modules_skipped):
            continue
        else:
            raise ValueError('Unexpected module: %s' % str(m))

if __name__ == '__main__':
    net = rasnet().cuda()
    

    '''
    img = torch.tensor(torch.randn(1, 3, 256, 512), requires_grad=False).cuda()
    out = net(img)
    number = 0
    params = net.state_dict()
    for k,v in params.items():
        print(k, v.shape)
        number += 1
    print("the number of layers is : %d " % (number//2))
    print 'output.size: ', type(out), len(out), out[-1].size()
    '''

    '''
    caffe_root = '../caffe_dss-master/'
    import sys
    sys.path.insert(0, caffe_root + 'python')
    import caffe
    '''


    '''
    vgg16weight_pth = torch.load(vgg16weight_pth_path)
    vgg16_keys = vgg16weight_pth.keys()

    net_keys =  net.state_dict().keys()
    for i in range(len(vgg16_keys)):
        print net_keys[i], net.state_dict()[net_keys[i]].size(), vgg16_keys[i], vgg16weight_pth[vgg16_keys[i]].size()
    '''

    # print net.state_dict().keys()
    # net_keys = net.state_dict().keys()
    # print type(net.state_dict()[net_keys[0]])
    net = interp_surgery(net)
    
    '''
    caffe.set_mode_cpu()
    solver = caffe.SGDSolver('solver.prototxt')
    solver.net.copy_from('vgg16.caffemodel')
    vgg_keys = solver.net.blobs.keys()
    vgg_keys = [x for x in vgg_keys if x.startswith('conv')]

    net.state_dict()['conv1.0.weight'].copy_(torch.from_numpy(solver.net.params['conv1_1'][0].data))
    net.state_dict()['conv1.0.bias'].copy_(torch.from_numpy(solver.net.params['conv1_1'][1].data))
    net.state_dict()['conv1.2.weight'].copy_(torch.from_numpy(solver.net.params['conv1_2'][0].data))
    net.state_dict()['conv1.2.bias'].copy_(torch.from_numpy(solver.net.params['conv1_2'][1].data))

    net.state_dict()['conv2.1.weight'].copy_(torch.from_numpy(solver.net.params['conv2_1'][0].data))
    net.state_dict()['conv2.1.bias'].copy_(torch.from_numpy(solver.net.params['conv2_1'][1].data))
    net.state_dict()['conv2.3.weight'].copy_(torch.from_numpy(solver.net.params['conv2_2'][0].data))
    net.state_dict()['conv2.3.bias'].copy_(torch.from_numpy(solver.net.params['conv2_2'][1].data))

    net.state_dict()['conv3.1.weight'].copy_(torch.from_numpy(solver.net.params['conv3_1'][0].data))
    net.state_dict()['conv3.1.bias'].copy_(torch.from_numpy(solver.net.params['conv3_1'][1].data))
    net.state_dict()['conv3.3.weight'].copy_(torch.from_numpy(solver.net.params['conv3_2'][0].data))
    net.state_dict()['conv3.3.bias'].copy_(torch.from_numpy(solver.net.params['conv3_2'][1].data))
    net.state_dict()['conv3.5.weight'].copy_(torch.from_numpy(solver.net.params['conv3_3'][0].data))
    net.state_dict()['conv3.5.bias'].copy_(torch.from_numpy(solver.net.params['conv3_3'][1].data))

    net.state_dict()['conv4.1.weight'].copy_(torch.from_numpy(solver.net.params['conv4_1'][0].data))
    net.state_dict()['conv4.1.bias'].copy_(torch.from_numpy(solver.net.params['conv4_1'][1].data))
    net.state_dict()['conv4.3.weight'].copy_(torch.from_numpy(solver.net.params['conv4_2'][0].data))
    net.state_dict()['conv4.3.bias'].copy_(torch.from_numpy(solver.net.params['conv4_2'][1].data))
    net.state_dict()['conv4.5.weight'].copy_(torch.from_numpy(solver.net.params['conv4_3'][0].data))
    net.state_dict()['conv4.5.bias'].copy_(torch.from_numpy(solver.net.params['conv4_3'][1].data))

    net.state_dict()['conv5.1.weight'].copy_(torch.from_numpy(solver.net.params['conv5_1'][0].data))
    net.state_dict()['conv5.1.bias'].copy_(torch.from_numpy(solver.net.params['conv5_1'][1].data))
    net.state_dict()['conv5.3.weight'].copy_(torch.from_numpy(solver.net.params['conv5_2'][0].data))
    net.state_dict()['conv5.3.bias'].copy_(torch.from_numpy(solver.net.params['conv5_2'][1].data))
    net.state_dict()['conv5.5.weight'].copy_(torch.from_numpy(solver.net.params['conv5_3'][0].data))
    net.state_dict()['conv5.5.bias'].copy_(torch.from_numpy(solver.net.params['conv5_3'][1].data))
    '''

    net = load_vgg16conv_fromcaffe(net, 'vgg16.pth')
    print 'done'