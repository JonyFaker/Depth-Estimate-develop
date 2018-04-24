import torch
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import pdb

class FCRN_Model(BaseModel):
	def name(self):
		return 'FCRN_Model'

	def initialize(self, opt):
		BaseModel.initialize(self, opt)
		self.isTrain = opt.isTrain

		self.FCRN = networks.define_FCRN(opt.input_nc, opt.output_nc, opt.which_model_fcrn, not opt.no_dropout, opt.init_type, self.gpu_ids)
		print(self.FCRN)
		# networks.print_network(net=self.FCRN)

		if self.isTrain:
			self.criterionMSE = torch.nn.MSELoss().cuda()
			# self.optimizer = torch.optim.Adam(self.FCRN.parameters(),
			# 					lr=opt.lr, betas=(opt.beta1, 0.999))
			FCRN_parameter = list(self.FCRN.parameters())
			self.optimizer = torch.optim.SGD(FCRN_parameter,
								lr=opt.lr, momentum=opt.momentum)


		print('---------- Networks initialized -------------')
        networks.print_network(net=self.FCRN)
        print('-----------------------------------------------')

	# def set_input(self, input):
 #        input_A = input['A']
 #        input_B = input['B']
 #        if len(self.gpu_ids) > 0:
 #            input_A = input_A.cuda(self.gpu_ids[0], async=True)
 #            input_B = input_B.cuda(self.gpu_ids[0], async=True)
 #        self.input_A = input_A
 #        self.input_B = input_B
 #        self.image_paths = input['A_paths' if AtoB else 'B_paths']

	def forward(self, input):
		input_A = input['A']
        input_B = input['B']
        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0], async=True)
            input_B = input_B.cuda(self.gpu_ids[0], async=True)
		net_input = Variable(input_A)
		self.net_input = net_input
		net_output = self.FCRN(net_input)
		self.net_output = net_output
		gd = Variable(input_B)
		self.gd = gd
		return net_output, gd

	def test(self):
		self.input = Variable(self.input_A, volatile=True)
		self.output = self.FCRN(input)
		self.gd = Variable(self.input_B, volatile=True)


	def get_image_paths(self):
		return self.image_paths

	def backward(self, net_output, gd):
		self.loss = self.criterionMSE(net_output, gd)
		loss.backward()


	def get_current_errors():
		return OrderedDict([('Loss_MES', self.loss.data[0])
                            ])

	def get_current_visuals(self):
        net_input = util.tensor2im(self.net_input.data)
        net_output = util.tensor2im(self.net_output.data)
        gd = util.tensor2im(self.gd.data)
        return OrderedDict([('net_input', net_input), ('net_output', net_output), ('gd', gd)])


	def save(self, lable):
		self.save_network(self.FCRN, 'FCRN', lable, self.gpu_ids)


	def train(self, input):
		# initialize(opt)
		net_output, gd = self.forward(input)
		self.optimizer.zero_grad()
		backward(net_output, gd)
		self.optimizer.step()








