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

		self.FCRN = networks.define_FCRN(opt.input_nc, opt.output_nc, opt.which_model_fcrn, not opt.use_dropout, opt.init_type, self.gpu_ids)

		if self.isTrain:
			pass
	

	def set_input(self, input):
        input_A = input['A']
        input_B = input['B']
        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0], async=True)
            input_B = input_B.cuda(self.gpu_ids[0], async=True)
        self.input_A = input_A
        self.input_B = input_B
        self.image_paths = input['A_paths']


	def forward(self):
		


	def test(self):
		pass

	def get_image_paths(self):
		pass

	def backward(self):
		pass

	def optimize_parameters(self):
		pass


	def get_current_errors():
		pass


	def save(self, lable):
		self.save_network(self.FCRN, 'FCRN', lable, self.gpu_ids)


