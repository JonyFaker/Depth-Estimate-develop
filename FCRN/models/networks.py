import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler
import numpy as np

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

def define_fcrn(self, input_nc, output_nc, ngf=64, which_model, init_type='normal', gpu_ids=[]):
	net_fcrn = None
	use_gpu = len(gpu_ids) > 0
	# norm_layer for what?

	if use_gpu:
		assert(torch.cuda.is_available())
	if which_model == 'resnet50':
		net_fcrn = FCRN_Res50(input_nc, output_nc, ngf, gpu_ids=gpu_ids)
	else:
		raise NotImplementedError('model name [%s] is not recognized' %
                                  which_model)

	if use_gpu:
		net_fcrn.cuda(gpu_ids[0])
	init_weights(net_fcrn, init_type=init_type)
	return net_fcrn


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


class FCRN_Vgg16():
	pass

class FCRN_Alex():
	pass


def batch_normalization():
	pass

class FCRN_Res50(torch.nn.Module):
	def __init__(self, input_nc=3, output_nc=1, n_blocks=9, use_dropout=False, gpu_ids=[]):
		assert(n_blocks >= 0)
		super(FCRN_Res50, self).__init__()
		self.input_nc = input_nc
		self.output_nc = output_nc
		self.gpu_ids = gpu_ids
		self.training = True

'''
		# block 1
		self.conv7_2 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=0, bias=True)
		self.bn1_1 = nn.BatchNorm2d(64)
		self.relu1_1 = nn.Relu()
		self.maxpool_1 = nn.Maxpool2d(kernel_size=(3, 3), stride=(2, 2))
		self.conv1_1 = nn.Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)
		self.bn1_2 = nn.BatchNorm2d(256)

		# block 2
		self.ResidualBlock_Projection_64_256 = ResidualBlock_Projection(64, 256)
		self.ResidualBlock_Projection_128_512 = ResidualBlock_Projection(128, 512)
		self.ResidualBlock_Projection_256_1024 = ResidualBlock_Projection(256, 1024)
		self.ResidualBlock_Projection_512_2048 = ResidualBlock_Projection(512, 2048)
		# block 3
		self.ResidualBlock_Skip_64_256 = ResidualBlock_Skip(64, 256)
		self.ResidualBlock_Skip_128_512 = ResidualBlock_Skip(128, 512)
		self.ResidualBlock_Skip_256_1024 = ResidualBlock_Skip(256, 1024)
		self.ResidualBlock_Skip_512_2048 = ResidualBlock_Skip(512, 2048)
'''

		model = [nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=0, bias=True),
				 nn.BatchNorm2d(64),
				 nn.Relu(),
				 nn.Maxpool2d(kernel_size=(3, 3), stride=(2, 2)),
				 ResidualBlock_Projection(64, 256),
				 nn.Relu(),
				 ResidualBlock_Skip(64, 256),
				 nn.Relu(),
				 ResidualBlock_Skip(64, 256),
				 nn.Relu(),
				 ResidualBlock_Projection(128, 512),
				 nn.Relu(),
				 ResidualBlock_Skip(128, 512),
				 nn.ReLu(),
				 ResidualBlock_Skip(128, 512),
				 nn.ReLu(),
				 ResidualBlock_Skip(128, 512),
				 nn.ReLu(),
				 ResidualBlock_Projection(256, 1024),
				 nn.Relu(),
				 ResidualBlock_Skip(256, 1024),
				 nn.Relu(),
				 ResidualBlock_Skip(256, 1024),
				 nn.Relu(),
				 ResidualBlock_Skip(256, 1024),
				 nn.Relu(),
				 ResidualBlock_Skip(256, 1024),
				 nn.Relu(),
				 ResidualBlock_Skip(256, 1024),
				 nn.Relu(),
				 ResidualBlock_Projection(512, 2048),
				 nn.Relu(),
				 ResidualBlock_Skip(512, 2048),
				 nn.Relu(),
				 ResidualBlock_Skip(512, 2048),
				 nn.Relu(),

				 nn.Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=True),
				 nn.BatchNorm2d(256),

				 up_project(1024, 512),
				 up_project(512, 256),
				 up_project(256, 128),
				 up_project(128, 64),

				 nn.Conv2d(64, 1, kernel_size=(3, 3), stride=(1, 1), padding=0, bias=True)

				]
		self.model = model


	def forward(self, input):
		if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
			return nn.paraller.data_paraller(self.model, input, self.gpu_ids)
		else:
			return self.mode(input)



class ResidualBlock_Skip(torch.nn.Module):
	def __init__(self, channels_in, channels_out):
		super(ResidualBlock_Skip, self).__init__()
		block = []
		block += [	nn.Conv2d(channels_in, channels_in, kernel_size=(1, 1), stride=(1, 1), bias=False)
					nn.BatchNorm2d(channels_in)
					nn.ReLu()
					nn.Conv2d(channels_in, channels_in, kernel_size=(3, 3), stride=(1, 1), bias=False)
					nn.BatchNorm2d(channels_in)
					nn.ReLu()
					nn.Conv2d(channels_in, channels_out, kernel_size=(1, 1), stride=(1, 1), bias=False)
					nn.BatchNorm2d(channels_out)
				]

		self.block = block
		return nn.Sequential(*block)


	def forward(self, input):
		out = input + self.block(input)
		return out


class ResidualBlock_Projection(torch.nn.Module):
	def __init__(self, channels_in, channels_out):
		super(ResidualBlock_Projection, self).__init__()
		self.block_A = self.build_block_A(channels_in, channels_out)
		self.block_B = self.build_block_B(channels_in, channels_out)

	def build_block_A(channels_in, channels_out, stride=(1, 1)):
		block_A = []
		block_A += [nn.Conv2d(channels_in, channels_in, kernel_size=(1, 1), stride=stride)
					nn.BatchNorm2d(channels_in)
					nn.ReLu()
					nn.Conv2d(channels_in, channels_in, kernel_size=(3, 3), stride=(1, 1))
					nn.BatchNorm2d(channels_in)
					nn.ReLu()
					nn.Conv2d(channels_in, channels_out, kernel_size=(1, 1), stride=(1,1))
					nn.BatchNorm2d(channels_out)
					]
					
		return nn.Sequential(*block_A)
		
	def build_block_B(channels_in, channels_out):
		block_B = []
		block_B += [nn.Conv2d(channels_in, channels_out, kernel_size=(1, 1), stride=stride)
					nn.BatchNorm2d(channels_out)
					]

		return nn.Sequential(*block_B)

	def forward(self, input):
		out = self.block_A(input) + self.block_B(input)
		return out


def get_incoming_shape(incoming):
	list shape = []
	if isinstance(incoming, torch.cuda.FloatTensor) or isinstance(incoming, torch.FloatTensor):
		shape = list(incoming.size())
		return shape
	else:
        raise Exception("Invalid incoming tensor.")

def interleave(inputs, axis):
	old_shape = get_incoming_shape(inputs[0])[1:]
	new_shape = [-1] + old_shape
	new_shape[axis] *= len(inputs)
	# tensor to numpy
	array1 = inputs.numpy()
	inputs_stack = np.stack(inputs, axis+1)
	inputs_stack.reshape(new_shape)
	# numpy to tensor
	torch_data = torch.from_numpy(inputs_stack)
	return torch_data
	

class unpool_as_conv(nn.Module):
	def __init__(self, channels_in, channels_out, stride, ReLu=False, BN):
		super(unpool_as_conv, self).__init__()
		self.conv_A = self.get_conv_A(self, channels_in, channels_out, stride=(1, 1), ReLu, bias=True)
		self.conv_B = self.get_conv_B(self, channels_in, channels_out, stride=(1, 1), ReLu, bias=True)
		self.conv_C = self.get_conv_C(self, channels_in, channels_out, stride=(1, 1), ReLu, bias=True)
		self.conv_D = self.get_conv_D(self, channels_in, channels_out, stride=(1, 1), ReLu, bias=True)
		self.bn = nn.BatchNorm2d(channels_out)

	def get_conv_A(self, channels_in, channels_out, stride=(1, 1), ReLu, bias=True):
		# conv A 3*3
		conv_A = []
		conv_A += [  ## tf: self.conv( 3, 3, size[3], stride, stride, name = layerName, padding = 'SAME', relu = False)
						nn.Conv2d(channels_in, channels_out, kernel_size=(3, 3), stride=stride, padding=(1, 1), bias=bias)
				  ]
		return nn.Sequential(*conv_A)

	def get_conv_B(self, input, channels_in, channels_out, stride=(1,1), ReLu, bias=True):
		# conv B 2*3
		conv_B = []
		conv_B += [
					# nn.functional.pad(input, [[0, 0], [0, 1], [1, 1], [0, 0]]),
					nn.Conv2d(channels_in, channels_out, kernel_size=(2, 3), stride=stride, padding=0, bias=bias)
					]
		return nn.Sequential(*conv_B)

	def get_conv_C(self, input, channels_in, channels_out, stride=(1, 1), ReLu, bias=True):
		# conv C 3*2
		conv_C = []
		conv_C += [
					# nn.functional.pad(input, [[0, 0], [1, 1], [0, 1], [0, 0]]),
					nn.Conv2d(channels_in, channels_out, kernel_size=(3, 2), stride=stride, padding=0, bias=bias)
					]
		return nn.Sequential(*conv_C)

	def get_conv_D(self, input, channels_in, channels_out, stride=(1, 1), ReLu, bias=True):
		# conv D 2*2
		conv_D = []
		conv_D += [	# nn.functional.pad(input, [[0, 0], [0, 1], [0, 1], [0, 0]]),
					nn.Conv2d(channels_in, channels_out, kernel_size=(2, 2), stride=stride, padding=0, bias=bias)
				  ]
		return nn.Sequential(*conv_D)

	def forward(self, input):
		outputA = self.conv_A(input, channels_in, channels_out)

		inputB = nn.functional.padding(input, [[0, 0], [0, 1], [1, 1], [0, 0]])
		outputB = self.conv_B(inputB, channels_in, channels_out)
		
		inputC = nn.functional.padding(input, [[0, 0], [1, 1], [0, 1], [0, 0]])
		outputC = self.conv_C(inputC, channels_in, channels_out)
		
		inputD = nn.functional.padding(input, [[0, 0], [0, 1], [0, 1], [0, 0]])
		outputD = self.conv_D(inputD, channels_in, channels_out)

		left = interleave([outputA, outputB], axis=1)
		right = interleave([outputC, outputD], axis=1)
		Y = interleave([left, right], axis=2)
		if BN:
			Y = self.bn(channels_out)
		if ReLu:
			Y = nn.functional.ReLu(Y)
		return Y

class up_project(nn.Module):
	def __init__(self, kernel_size=(3, 3),channels_in, channels_out, id, stride=(1, 1), BN=True):
		super(up_project, self).__init__()
		self.unpool_as_conv = unpool_as_conv(channels_in, channels_out, stride = stride, Relu=True, BN = True)
		self.conv = nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size, stride=stride, padding=0, bias=True)
		self.bn = nn.BatchNorm2d(channels_out)


	def forward(self, input):
		output_temp = self.unpool_as_conv(input, channels_in, channels_out, ReLu=True, BN=True)
		branch1_output = self.conv(output_temp, channels_in, channels_out, kernel_size=kernel_size, stride=stride, padding=0, bias=True)
		if BN:
			branch1_output = self.bn(channels_out)

		branch2_output = self.unpool_as_conv(branch1_output, channels_in, channels_out, kernel_size=kernel_size, stride=stride, padding=0, bias=True)
		output = branch1_output + branch2_output
		output = nn.functional.Relu(output)
		return output



