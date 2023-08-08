import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import math
# from transformers import BertModel
from torch.autograd import Variable

from experiments.modules import Scaler


def init_param(m):
    if isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.bias.data.zero_()
    return m

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


HIDDEN_SIZE_CONV = [64, 128, 256, 512]
HIDDEN_SIZE_RESNET = [64, 128, 256, 512]

def model_pull(args, model_rate=1, track=False):
    if args['model_type'] == 'Lenet':  
        return conv(model_rate=model_rate, track=track, args=args)

    elif args['model_type'] == 'ResNet_18':  
        return resnet18(model_rate=model_rate, track=track, args=args)
    
    elif args['model_type'] == 'ResNet_20':  
        return resnet20(model_rate=model_rate, track=track, args=args)

    elif args['model_type'] == 'ResNet_34':  
        return resnet34(model_rate=model_rate, track=track, args=args)
    
    elif args['model_type'] == 'ResNet_50':  
        return resnet50(model_rate=model_rate, track=track, args=args)
    
    elif args['model_type'] == 'ResNet_101':  
        return resnet101(model_rate=model_rate, track=track, args=args)
    
    elif args['model_type'] == 'ResNet_152':  
        return resnet152(model_rate=model_rate, track=track, args=args)

    else:
        raise Exception(f"unkonw model_type: {args['model_type']}")


def generator_model_pull(args):
    if args['generator_model_type'] == 'ACGAN':  
        return Generator_ACGan(args).apply(weights_init)
    

class Conv(nn.Module):
    def __init__(self, hidden_size, rate, track, args):
        super().__init__()
        self.args = args

        if args['data_set'] in ['MNIST', 'FMNIST']:
            data_shape = [1, 32, 32]
            self.classes_size = 10
        elif args['data_set'] in ['SVHN', 'CIFAR-10', 'CINIC-10', 'CIFAR-100']:
            data_shape = [3, 32, 32]
            if args['data_set'] in ['SVHN', 'CIFAR-10', 'CINIC-10']:
                self.classes_size = 10
            elif args['data_set'] == 'CIFAR-100':
                self.classes_size = 100
        elif args['data_set'] in ['Tiny-Imagenet']:
            data_shape = [3, 64, 64]
            self.classes_size = 200
        elif args['data_set'] in ['FOOD101']:
            data_shape = [3, 64, 64]
            self.classes_size = 101
        elif args['data_set'] in ['GTSRB']:
            data_shape = [3, 64, 64]
            self.classes_size = 43

        if args['model_norm'] == 'bn':
            norm = nn.BatchNorm2d(hidden_size[0], momentum=None, track_running_stats=track)
        elif args['model_norm'] == 'in':
            norm = nn.GroupNorm(hidden_size[0], hidden_size[0])
        elif args['model_norm'] == 'ln':
            norm = nn.GroupNorm(1, hidden_size[0])
        elif args['model_norm'] == 'gn':
            norm = nn.GroupNorm(4, hidden_size[0])
        elif args['model_norm'] == 'none':
            norm = nn.Identity()
        else:
            raise ValueError('Not valid norm')
        if args['scale']:
            scaler = Scaler(rate)
        else:
            scaler = nn.Identity()
        self.conv1 = nn.Sequential(nn.Conv2d(data_shape[0], hidden_size[0], 3, 1, 1), scaler)
        self.n1 = norm
        self.ReLU_pool = nn.Sequential(nn.ReLU(inplace=True), nn.MaxPool2d(2))
                                       
        for i in range(len(hidden_size) - 1):
            if args['model_norm'] == 'bn':
                norm = nn.BatchNorm2d(hidden_size[i + 1], momentum=None, track_running_stats=track)
            elif args['model_norm'] == 'in':
                norm = nn.GroupNorm(hidden_size[i + 1], hidden_size[i + 1])
            elif args['model_norm'] == 'ln':
                norm = nn.GroupNorm(1, hidden_size[i + 1])
            elif args['model_norm'] == 'gn':
                norm = nn.GroupNorm(4, hidden_size[i + 1])
            elif args['model_norm'] == 'none':
                norm = nn.Identity()
            else:
                raise ValueError('Not valid norm')
            if args['scale']:
                scaler = Scaler(rate)
            else:
                scaler = nn.Identity()
            
            exec(f'self.conv{i+2} = nn.Sequential(nn.Conv2d(hidden_size[i], hidden_size[i + 1], 3, 1, 1), scaler)')
            exec(f'self.n{i+2} = norm')

        self.pool_Flatten = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.linear = nn.Linear(hidden_size[-1], self.classes_size)

    def forward(self, x, label_list=None, bn_or_not=False):
        bn_input_list = []

        out = self.conv1(x)
        bn_input_list.append(out)
        out = self.n1(out)
        out = self.ReLU_pool(out)

        out = self.conv2(out)
        bn_input_list.append(out)
        out = self.n2(out)
        out = self.ReLU_pool(out)

        out = self.conv3(out)
        bn_input_list.append(out)
        out = self.n3(out)
        out = self.ReLU_pool(out)

        out = self.conv4(out)
        bn_input_list.append(out)
        out = self.n4(out)
        out = self.ReLU_pool(out)

        out = self.pool_Flatten(out)
        out = self.linear(out)
        if label_list and self.args['mask']:
            label_mask = torch.zeros(self.classes_size, device=out.device)
            label_mask[torch.tensor(label_list)] = 1
            out = out.masked_fill(label_mask == 0, 0)
        if bn_or_not == False:
            return out
        else:
            return out, bn_input_list


def conv(model_rate=1, track=False, args=None):
    hidden_size = [int(np.ceil(model_rate * x)) for x in HIDDEN_SIZE_CONV]
    try:
        scaler_rate = model_rate / args['global_model_rate']
    except:
        scaler_rate = 1.0
    model = Conv(hidden_size, scaler_rate, track, args)
    model.apply(init_param)
    return model


########Resnet#############
class Block(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, mom, stride, rate, track, args):
        super(Block, self).__init__()
        if args['model_norm'] == 'bn':
            n1 = nn.BatchNorm2d(in_planes, momentum=mom, track_running_stats=track)
            n2 = nn.BatchNorm2d(planes, momentum=mom, track_running_stats=track)
        elif args['model_norm'] == 'in':
            n1 = nn.GroupNorm(in_planes, in_planes)
            n2 = nn.GroupNorm(planes, planes)
        elif args['model_norm'] == 'ln':
            n1 = nn.GroupNorm(1, in_planes)
            n2 = nn.GroupNorm(1, planes)
        elif args['model_norm'] == 'gn':
            n1 = nn.GroupNorm(4, in_planes)
            n2 = nn.GroupNorm(4, planes)
        elif args['model_norm'] == 'none':
            n1 = nn.Identity()
            n2 = nn.Identity()
        else:
            raise ValueError('Not valid norm')
        self.n1 = n1
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.n2 = n2
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        if args['scale']:
            self.scaler = Scaler(rate)
        else:
            self.scaler = nn.Identity()

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        out0 = self.scaler(x)
        out1 = self.n1(out0)
        out2 = F.relu(out1)
        shortcut = self.shortcut(out2) if hasattr(self, 'shortcut') else x
        out3 = self.conv1(out2)

        out4 = self.scaler(out3) 
        out5 = self.n2(out4)
        out6 = F.relu(out5)
        out7 = self.conv2(out6)
        out7 += shortcut
        return out7, [out0, out4]


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, mom, stride, rate, track, args):
        super(Bottleneck, self).__init__()
        if args['model_norm'] == 'bn':
            n1 = nn.BatchNorm2d(in_planes, momentum=mom, track_running_stats=track)
            n2 = nn.BatchNorm2d(planes, momentum=mom, track_running_stats=track)
            n3 = nn.BatchNorm2d(planes, momentum=mom, track_running_stats=track)
        elif args['model_norm'] == 'in':
            n1 = nn.GroupNorm(in_planes, in_planes)
            n2 = nn.GroupNorm(planes, planes)
            n3 = nn.GroupNorm(planes, planes)
        elif args['model_norm'] == 'ln':
            n1 = nn.GroupNorm(1, in_planes)
            n2 = nn.GroupNorm(1, planes)
            n3 = nn.GroupNorm(1, planes)
        elif args['model_norm'] == 'gn':
            n1 = nn.GroupNorm(4, in_planes)
            n2 = nn.GroupNorm(4, planes)
            n3 = nn.GroupNorm(4, planes)
        elif args['model_norm'] == 'none':
            n1 = nn.Identity()
            n2 = nn.Identity()
            n3 = nn.Identity()
        else:
            raise ValueError('Not valid norm')
        self.n1 = n1
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.n2 = n2
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.n3 = n3
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        if args['scale']:
            self.scaler = Scaler(rate)
        else:
            self.scaler = nn.Identity()

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        out0 = self.scaler(x)
        out1 = self.n1(out0)
        out2 = F.relu(out1)

        shortcut = self.shortcut(out2) if hasattr(self, 'shortcut') else x

        out3 = self.conv1(out2)

        out4 = self.scaler(out3)
        out5 = self.n2(out4)
        out6 = F.relu(out5)
        out7 = self.conv2(out6)

        out8 = self.scaler(out7)
        out9 = self.n3(out8)
        out10 = F.relu(out9)
        out11 = self.conv3(out10)
        out11 += shortcut
        return out11, [out0, out4, out8]


class ResNet(nn.Module):
    def __init__(self, hidden_size, block, num_blocks, rate, track, args):
        super(ResNet, self).__init__()
        self.args = args
        self.num_blocks = num_blocks
        if args['data_set'] in ['MNIST', 'FMNIST']:
            data_shape = [1, 32, 32]
            self.classes_size = 10
        elif args['data_set'] in ['CIFAR-10', 'CIFAR-100', 'CINIC-10', 'SVHN']:
            data_shape = [3, 32, 32]
            if args['data_set'] in ['CIFAR-10', 'SVHN', 'CINIC-10']:
                self.classes_size = 10
            elif args['data_set'] == 'CIFAR-100':
                self.classes_size = 100
        elif args['data_set'] in ['Tiny-Imagenet']:
            data_shape = [3, 64, 64]
            self.classes_size = 200
        elif args['data_set'] in ['FOOD101']:
            data_shape = [3, 64, 64]
            self.classes_size = 101
        elif args['data_set'] in ['GTSRB']:
            data_shape = [3, 64, 64]
            self.classes_size = 43

        self.in_planes = hidden_size[0]
        self.conv1 = nn.Conv2d(data_shape[0], hidden_size[0], kernel_size=3, stride=1, padding=1, bias=False)
        
        layer1 = self._make_layer(block, hidden_size[0], num_blocks[0], mom=None, stride=1, rate=rate, track=track)
        for i in range(num_blocks[0]):
            exec(f'self.layer1_{i} = layer1[i]')

        layer2 = self._make_layer(block, hidden_size[1], num_blocks[1], mom=None, stride=2, rate=rate, track=track)
        for i in range(num_blocks[1]):
            exec(f'self.layer2_{i} = layer2[i]')

        layer3 = self._make_layer(block, hidden_size[2], num_blocks[2], mom=None, stride=2, rate=rate, track=track)
        for i in range(num_blocks[2]):
            exec(f'self.layer3_{i} = layer3[i]')

        layer4 = self._make_layer(block, hidden_size[3], num_blocks[3], mom=None, stride=2, rate=rate, track=track)
        for i in range(num_blocks[3]):
            exec(f'self.layer4_{i} = layer4[i]')

        if args['model_norm'] == 'bn':
            n4 = nn.BatchNorm2d(hidden_size[3] * block.expansion, momentum=None, track_running_stats=track)
        elif args['model_norm'] == 'in':
            n4 = nn.GroupNorm(hidden_size[3] * block.expansion, hidden_size[3] * block.expansion)
        elif args['model_norm'] == 'ln':
            n4 = nn.GroupNorm(1, hidden_size[3] * block.expansion)
        elif args['model_norm'] == 'gn':
            n4 = nn.GroupNorm(4, hidden_size[3] * block.expansion)
        elif args['model_norm'] == 'none':
            n4 = nn.Identity()
        else:
            raise ValueError('Not valid norm')
        self.n4 = n4
        if args['scale']:
            self.scaler = Scaler(rate)
        else:
            self.scaler = nn.Identity()
        self.linear = nn.Linear(hidden_size[3] * block.expansion, self.classes_size)

    def _make_layer(self, block, planes, num_blocks, mom, stride, rate, track):
        args = self.args
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, mom, stride, rate, track, args))
            self.in_planes = planes * block.expansion
        return layers #nn.Sequential(*layers)

    def forward(self, x, label_list=None, bn_or_not=False):
        bn_input_list = []
        # x = input['img']
        out = self.conv1(x)

        for b_i in range(len(self.num_blocks)):
            for i in range(self.num_blocks[b_i]):
                layer = getattr(self, f'layer{b_i+1}_{i}')
                out, bn_input = layer(out)
                bn_input_list.extend(bn_input)

        out = self.scaler(out)
        bn_input_list.append(out)
        out = self.n4(out)
        out = F.relu(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        if label_list and self.args['mask']:
            label_mask = torch.zeros(self.classes_size, device=out.device)
            label_mask[torch.tensor(label_list)] = 1
            out = out.masked_fill(label_mask == 0, 0)
        # output['score'] = out
        # output['loss'] = F.cross_entropy(output['score'], input['label'])
        if bn_or_not == False:
            return out
        else:
            return out, bn_input_list


def resnet18(model_rate=1, track=True, args=None):
    hidden_size = [int(np.ceil(model_rate * x)) for x in HIDDEN_SIZE_RESNET]
    try:
        scaler_rate = model_rate / args['global_model_rate']
    except:
        scaler_rate = 1.0
    model = ResNet(hidden_size, Block, [2, 2, 2, 2], scaler_rate, track, args)
    model.apply(init_param)
    return model

def resnet20(model_rate=1, track=True, args=None):
    hidden_size = [int(np.ceil(model_rate * x)) for x in HIDDEN_SIZE_RESNET]
    try:
        scaler_rate = model_rate / args['global_model_rate']
    except:
        scaler_rate = 1.0
    model = ResNet(hidden_size, Block, [2, 3, 2, 2], scaler_rate, track, args)
    model.apply(init_param)
    return model


def resnet34(model_rate=1, track=True, args=None):
    hidden_size = [int(np.ceil(model_rate * x)) for x in HIDDEN_SIZE_RESNET]
    try:
        scaler_rate = model_rate / args['global_model_rate']
    except:
        scaler_rate = 1.0
    model = ResNet(hidden_size, Block, [3, 4, 6, 3], scaler_rate, track, args)
    model.apply(init_param)
    return model


def resnet50(model_rate=1, track=True, args=None):
    hidden_size = [int(np.ceil(model_rate * x)) for x in HIDDEN_SIZE_RESNET]
    try:
        scaler_rate = model_rate / args['global_model_rate']
    except:
        scaler_rate = 1.0
    model = ResNet(hidden_size, Bottleneck, [3, 4, 6, 3], scaler_rate, track, args)
    model.apply(init_param)
    return model


def resnet101(model_rate=1, track=True, args=None):
    hidden_size = [int(np.ceil(model_rate * x)) for x in HIDDEN_SIZE_RESNET]
    try:
        scaler_rate = model_rate / args['global_model_rate']
    except:
        scaler_rate = 1.0
    model = ResNet(hidden_size, Bottleneck, [3, 4, 23, 3], scaler_rate, track, args)
    model.apply(init_param)
    return model


def resnet152(model_rate=1, track=True, args=None):
    hidden_size = [int(np.ceil(model_rate * x)) for x in HIDDEN_SIZE_RESNET]
    try:
        scaler_rate = model_rate / args['global_model_rate']
    except:
        scaler_rate = 1.0
    model = ResNet(hidden_size, Bottleneck, [3, 8, 36, 3], scaler_rate, track, args)
    model.apply(init_param)
    return model


###########Generator########## 
class Generator_ACGan(nn.Module):
    
    def __init__(self, args):
        super(Generator_ACGan,self).__init__()

        self.data_set = args['data_set']
        self.latent_dim = args['latent_dim']
        self.noise_label_combine = args['noise_label_combine']
        if self.data_set in ['CIFAR-10', 'FMNIST', 'MNIST', 'SVHN', 'CINIC-10']:
            self.n_classes = 10
        elif self.data_set in ['CIFAR-100']:
            self.n_classes = 100
        elif self.data_set in ['Tiny-Imagenet']:
            self.n_classes = 200
        elif self.data_set in ['FOOD101']:
            self.n_classes = 101
        elif self.data_set in ['GTSRB']:
            self.n_classes = 43
        
        if self.noise_label_combine in ['cat']:
            input_dim = 2 * self.latent_dim
        elif self.noise_label_combine in ['cat_naive']:
            input_dim = self.latent_dim + self.n_classes
        else:
            input_dim = self.latent_dim

        #input 100*1*1
        self.layer1 = nn.Sequential(nn.ConvTranspose2d(input_dim, 512, 4, 1, 0, bias = False),
                                   nn.ReLU(True))

        #input 512*4*4
        self.layer2 = nn.Sequential(nn.ConvTranspose2d(512, 256, 4, 2, 1, bias = False),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU(True))
        #input 256*8*8
        self.layer3 = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(True))
        
        self.layer4_1 = nn.Sequential(nn.ConvTranspose2d(128, 3, 4, 2, 1, bias = False),
                                   nn.Tanh()) 
        ## Tiny-imagenet
        #input 128*16*16
        self.layer4_2 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, 2, 1, bias = False),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(True)) ##for Tiny-Imagenet
        #input 64*32*32
        self.layer5 = nn.Sequential(nn.ConvTranspose2d(64, 3, 4, 2, 1, bias = False),
                                   nn.Tanh()) ##for Tiny-Imagenet
        #output 3*64*64 MNIST and FMNIST
        self.layer4_3 = nn.Sequential(nn.ConvTranspose2d(128, 1, 4, 2, 1, bias = False),
                                   nn.Tanh())  ##for mnist or Fmnist
        
        ## FOOD101
        self.layer4_4 = nn.Sequential(nn.ConvTranspose2d(128, 128, 4, 2, 1, bias = False),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(True))
        
        self.layer4_5 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, 2, 1, bias = False),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(True))
        
        self.layer4_6 = nn.Sequential(nn.ConvTranspose2d(64, 3, 4, 2, 1, bias = False),
                                   nn.Tanh())
        self.embedding = nn.Embedding(self.n_classes, self.latent_dim)
        
        
    def forward(self, noise, label):
        
        if self.noise_label_combine == 'mul':
            label_embedding = self.embedding(label)
            h = torch.mul(noise, label_embedding)
        elif self.noise_label_combine == 'add':
            label_embedding = self.embedding(label)
            h = torch.add(noise, label_embedding)
        elif self.noise_label_combine == 'cat':
            label_embedding = self.embedding(label)
            h = torch.cat((noise, label_embedding), dim=1)
        elif self.noise_label_combine == 'cat_naive':
            label_embedding = Variable(torch.cuda.FloatTensor(len(label), self.n_classes))
            label_embedding.zero_()
            #labels = labels.view
            label_embedding.scatter_(1, label.view(-1,1), 1)
            h = torch.cat((noise, label_embedding), dim=1)
        else:
            label_embedding = noise
            h = noise
            
        # x = torch.mul(noise, label_embedding) # element-wise multiplication
        x = h.view(-1, h.shape[1], 1, 1)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.data_set in ['Tiny-Imagenet', 'FOOD101', 'GTSRB']:
            x = self.layer4_2(x)
            x = self.layer5(x)
        elif self.data_set in ['FMNIST', 'MNIST']:
            x = self.layer4_3(x)
        elif self.data_set in ['']:
            x = self.layer4_4(x)
            x = self.layer4_5(x)
            x = self.layer4_6(x)
        else:
            x = self.layer4_1(x)

        return x, h, label_embedding

