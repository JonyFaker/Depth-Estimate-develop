import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler
import numpy as np
import pdb


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

    
###################################
def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)
###################################

def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

# def define_fcrn(self, input_nc, output_nc, ngf=64, which_model, init_type='normal', gpu_ids=[]):
# 	net_fcrn = None
# 	use_gpu = len(gpu_ids) > 0
# 	# norm_layer for what?

# 	if use_gpu:
# 		assert(torch.cuda.is_available())
# 	if which_model == 'resnet50':
# 		net_fcrn = FCRN_Res50(input_nc, output_nc, ngf, gpu_ids=gpu_ids)
# 	else:
# 		raise NotImplementedError('model name [%s] is not recognized' %
#                                   which_model)

# 	if use_gpu:
# 		net_fcrn.cuda(gpu_ids[0])
# 	init_weights(net_fcrn, init_type=init_type)
# 	return net_fcrn


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def define_FCRN(input_nc=3, output_nc=1, which_model_fcrn='resnet50', use_dropout=False, init_type='normal', gpu_ids=[]):
	net_fcrn = None
	use_gpu = len(gpu_ids) > 0

	if use_gpu:
		assert(torch.cuda.is_available())
	if which_model_fcrn == 'resnet50':
		net_fcrn = FCRN_Res50(input_nc=input_nc, output_nc=output_nc, use_dropout=use_dropout, gpu_ids=gpu_ids)
	elif which_model_fcrn == 'vgg16':
		raise NotImplementedError('vgg16 based fcrn not implemented!')
	elif which_model_fcrn == 'alex':
		raise NotImplementedError('alex based fcrn not implemented!')
	else:
		raise NotImplementedError('unkonwn net version')

	if use_gpu:
		net_fcrn.cuda(gpu_ids[0])
	init_weights(net_fcrn, init_type=init_type)
	return net_fcrn


class FCRN_Vgg16():
	pass

class FCRN_Alex():
	pass


def batch_normalization():
	pass

class PadLayer(nn.Module):
	def __init__(self, kernel_size=1):
		super(PadLayer, self).__init__()
		self.kernel_size = kernel_size

	def forward(self, input):
		return nn.functional.pad(input, (0, 0, 0, 0, (self.kernel_size-1)//2, (self.kernel_size-1)//2, (self.kernel_size-1)//2, (self.kernel_size-1)//2))

class FCRN_Res50(torch.nn.Module):
	def __init__(self, input_nc=3, output_nc=1, use_dropout=False, gpu_ids=[]):
		super(FCRN_Res50, self).__init__()
		self.input_nc = input_nc
		self.output_nc = output_nc
		self.gpu_ids = gpu_ids
		self.training = True

		model = [nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=((7-1)//2, (7-1)//2), bias=True),
				 # nn.functional.pad(input, (0, 0, 0, 0, (7-1)//2, (7-1)//2, (7-1)//2, (7-1)//2)),
				 # PadLayer(7),
				 nn.BatchNorm2d(64),
				 nn.ReLU(),
				 nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1),
				 ResidualBlock_Projection(64, 64, 256),
				 nn.ReLU(),
				 ResidualBlock_Skip(256, 64, 256),
				 nn.ReLU(),
				 ResidualBlock_Skip(256, 64 ,256),
				 nn.ReLU(),
				 ResidualBlock_Projection(256, 128, 512, stride_1=(2,2)),
				 nn.ReLU(),
				 ResidualBlock_Skip(512, 128, 512),
				 nn.ReLU(),
				 ResidualBlock_Skip(512, 128, 512),
				 nn.ReLU(),
				 ResidualBlock_Skip(512, 128, 512),
				 nn.ReLU(),
				 ResidualBlock_Projection(512, 256, 1024, stride_1=(2,2)),
				 nn.ReLU(),
				 ResidualBlock_Skip(1024, 256, 1024),
				 nn.ReLU(),
				 ResidualBlock_Skip(1024, 256, 1024),
				 nn.ReLU(),
				 ResidualBlock_Skip(1024, 256, 1024),
				 nn.ReLU(),
				 ResidualBlock_Skip(1024, 256, 1024),
				 nn.ReLU(),
				 ResidualBlock_Skip(1024, 256, 1024),
				 nn.ReLU(),
				 ResidualBlock_Projection(1024, 512, 2048, stride_1=(2,2)),
				 nn.ReLU(),
				 ResidualBlock_Skip(2048, 512, 2048),
				 nn.ReLU(),
				 ResidualBlock_Skip(2048, 512, 2048),
				 nn.ReLU(),

				 nn.Conv2d(2048, 1024, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=True),
				 nn.BatchNorm2d(1024),

				 up_project(1024, 512),
				 up_project(512, 256),
				 up_project(256, 128),
				 up_project(128, 64),

				 nn.Conv2d(64, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)

				]
		self.model = nn.Sequential(*model)


	def forward(self, input):
		# if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
		# 	return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
		# else:
			#return self.model(input)
		x = input
		for name, layer in self.model._modules.items():
			x = layer(x)
			# print(name)
			print(x.size())
		return x



class ResidualBlock_Skip(torch.nn.Module):
	def __init__(self, channels_in, channels_middle, channels_out, stride_1=(1,1), stride_2=(1,1), stride_3=(1,1)):
		super(ResidualBlock_Skip, self).__init__()
		block = []
		block += [	nn.Conv2d(channels_in, channels_middle, kernel_size=(1, 1), padding=((1-1)//2, (1-1)//2), stride=stride_1, bias=False),
					# nn.functional.pad(input, (0, 0, 0, 0, (1-1)//2, (1-1)//2, (1-1)//2, (1-1)//2)),
					# PadLayer(1),
					nn.BatchNorm2d(channels_middle),
					nn.ReLU(),
					nn.Conv2d(channels_middle, channels_middle, kernel_size=(3, 3), padding=((3-1)//2, (3-1)//2), stride=stride_2, bias=False),
					# nn.functional.pad(input, (0, 0, 0, 0, (3-1)//2, (3-1)//2, (3-1)//2, (3-1)//2)),
					# PadLayer(3),
					nn.BatchNorm2d(channels_middle),
					nn.ReLU(),
					nn.Conv2d(channels_middle, channels_out, kernel_size=(1, 1), padding=((1-1)//2, (1-1)//2), stride=stride_3, bias=False),
					# nn.functional.pad(input, (0, 0, 0, 0, (1-1)//2, (1-1)//2, (1-1)//2, (1-1)//2)),
					# PadLayer(1),
					nn.BatchNorm2d(channels_out)
				]

		self.block = nn.Sequential(*block)


	def forward(self, input):
		out = input + self.block(input)
		return out


class ResidualBlock_Projection(torch.nn.Module):
	def __init__(self, channels_in, channels_middle, channels_out, stride_1=(1,1), stride_2=(1,1), stride_3=(1,1)):
		super(ResidualBlock_Projection, self).__init__()
		self.block_A = self.build_block_A(channels_in, channels_middle, channels_out, stride_1, stride_2, stride_3)
		self.block_B = self.build_block_B(channels_in, channels_out, stride_1)

	def build_block_A(self, channels_in, channels_middle, channels_out, stride_1, stride_2, stride_3):
		# print(channels_in, channels_out)
		block_A = []
		block_A += [nn.Conv2d(channels_in, channels_middle, kernel_size=(1, 1), padding=((1-1)//2, (1-1)//2), stride=stride_1),
					# nn.functional.pad(input, (0, 0, 0, 0, (1-1)//2, (1-1)//2, (1-1)//2, (1-1)//2)),
					# PadLayer(1),
					nn.BatchNorm2d(channels_middle),
					nn.ReLU(),
					nn.Conv2d(channels_middle, channels_middle, kernel_size=(3, 3), padding=((3-1)//2, (3-1)//2), stride=stride_2),
					# nn.functional.pad(input, (0, 0, 0, 0, (3-1)//2, (3-1)//2, (3-1)//2, (3-1)//2)),
					# PadLayer(3),
					nn.BatchNorm2d(channels_middle),
					nn.ReLU(),
					nn.Conv2d(channels_middle, channels_out, kernel_size=(1, 1), padding=((1-1)//2, (1-1)//2), stride=stride_3),
					# nn.functional.pad(input, (0, 0, 0, 0, (1-1)//2, (1-1)//2, (1-1)//2, (1-1)//2)),
					# PadLayer(1),
					nn.BatchNorm2d(channels_out)
					]
					
		return nn.Sequential(*block_A)
		
	def build_block_B(self, channels_in, channels_out, stride_1):
		block_B = []
		block_B += [nn.Conv2d(channels_in, channels_out, kernel_size=(1, 1), padding=((1-1)//2, (1-1)//2) , stride=stride_1),
					# nn.functional.pad(input, (0, 0, 0, 0, (1-1)//2, (1-1)//2, (1-1)//2, (1-1)//2)),
					# PadLayer(1),
					nn.BatchNorm2d(channels_out)
					]

		return nn.Sequential(*block_B)

	def forward(self, input):
		out = self.block_A(input) + self.block_B(input)
		return out


def get_incoming_shape(incoming):
	return list(incoming.size())
	# if isinstance(incoming, torch.cuda.FloatTensor) or isinstance(incoming, torch.FloatTensor):
	# 	shape = list(incoming.size())
	# 	return shape
	# else:
 #        raise Exception("Invalid incoming tensor.")

def interleave(inputs, axis):
	old_shape = get_incoming_shape(inputs[0])[1:]
	new_shape = [-1] + old_shape
	new_shape[axis] *= len(inputs)

	return torch.stack(inputs, axis+1).view(new_shape)

	# # tensor to numpy

	# array1 = inputs[0].numpy()
	# array2 = inputs[1].numpy()
	# inputs_stack = np.stack([array1, array2], axis+1)
	# inputs_stack.reshape(new_shape)
	# # numpy to tensor
	# torch_data = torch.from_numpy(inputs_stack)
	# # if torch.cuda.is_available():
	# # 	torch_data = torch_data.cuda()
	# return torch_data
	

class unpool_as_conv(nn.Module):
	def __init__(self, channels_in, channels_out, stride, BN, RELU=False):
		super(unpool_as_conv, self).__init__()
		self.conv_A = self.get_conv_A(channels_in, channels_out, True, stride)
		self.conv_B = self.get_conv_B(channels_in, channels_out, True, stride)
		self.conv_C = self.get_conv_C(channels_in, channels_out, True, stride)
		self.conv_D = self.get_conv_D(channels_in, channels_out, True, stride)
		self.bn = nn.BatchNorm2d(channels_out)
		self.BN = BN
		self.RELU = RELU

	def get_conv_A(self, channels_in, channels_out, bias, stride=(1, 1)):
		# conv A 3*3
		conv_A = []
		conv_A += [  ## tf: self.conv( 3, 3, size[3], stride, stride, name = layerName, padding = 'SAME', relu = False)
						nn.Conv2d(channels_in, channels_out, kernel_size=(3, 3), stride=stride, padding=(1, 1), bias=bias)
				  ]
		return nn.Sequential(*conv_A)

	def get_conv_B(self, channels_in, channels_out, bias, stride=(1,1)):
		# conv B 2*3
		conv_B = []
		conv_B += [
					# nn.functional.pad(input, [[0, 0], [0, 1], [1, 1], [0, 0]]),
					nn.Conv2d(channels_in, channels_out, kernel_size=(2, 3), stride=stride, padding=0, bias=bias)
					]
		return nn.Sequential(*conv_B)

	def get_conv_C(self, channels_in, channels_out, bias, stride=(1, 1)):
		# conv C 3*2
		conv_C = []
		conv_C += [
					# nn.functional.pad(input, [[0, 0], [1, 1], [0, 1], [0, 0]]),
					nn.Conv2d(channels_in, channels_out, kernel_size=(3, 2), stride=stride, padding=0, bias=bias)
					]
		return nn.Sequential(*conv_C)

	def get_conv_D(self, channels_in, channels_out, bias, stride=(1, 1)):
		# conv D 2*2
		conv_D = []
		conv_D += [	# nn.functional.pad(input, [[0, 0], [0, 1], [0, 1], [0, 0]]),
					nn.Conv2d(channels_in, channels_out, kernel_size=(2, 2), stride=stride, padding=0, bias=bias)
				  ]
		return nn.Sequential(*conv_D)

	def forward(self, input):
		# outputA = self.conv_A(input, channels_in, channels_out)
		outputA = self.conv_A(input)

		# inputB = nn.functional.pad(input, (0, 0, 0, 0, 0, 1, 1, 1))
		inputB = nn.functional.pad(input, (1, 1, 0, 1,))
		outputB = self.conv_B(inputB)
		
		# inputC = nn.functional.pad(input, (0, 0, 0, 0, 1, 1, 0, 1))
		inputC = nn.functional.pad(input, (0, 1, 1, 1))
		outputC = self.conv_C(inputC)
		
		# inputD = nn.functional.pad(input, (0, 0, 0, 0, 0, 1, 0, 1))
		inputD = nn.functional.pad(input, (0, 1, 0, 1))
		outputD = self.conv_D(inputD)

		left = interleave([outputA, outputB], axis=2)
		right = interleave([outputC, outputD], axis=2)
		Y = interleave([left, right], axis=3)
		if self.BN:
			Y = self.bn(Y)
		if self.RELU:
			Y = nn.functional.relu(Y)
		return Y

class up_project(nn.Module):
	def __init__(self, channels_in, channels_out, kernel_size=(3, 3), stride=(1, 1), BN=True):
		super(up_project, self).__init__()
		self.unpool_as_conv_1 = unpool_as_conv(channels_in=channels_in, channels_out=channels_out, stride=stride, RELU=True, BN=True)
		self.unpool_as_conv_2 = unpool_as_conv(channels_in=channels_in, channels_out=channels_out, stride=stride, RELU=False, BN=True)
		self.conv = nn.Conv2d(channels_out, channels_out, kernel_size=kernel_size, stride=stride, padding=(1, 1), bias=True)
		self.bn = nn.BatchNorm2d(channels_out)

	def forward(self, input):
		output_temp = self.unpool_as_conv_1(input)
		branch1_output_temp = self.conv(output_temp)
		branch1_output = self.bn(branch1_output_temp)

		branch2_output = self.unpool_as_conv_2(input)
		output = torch.add(branch1_output, branch2_output)
		# output = branch1_output + branch2_output
		output = nn.functional.relu(output)
		return output



