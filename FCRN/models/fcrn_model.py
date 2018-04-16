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

	