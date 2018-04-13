import torch
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks

class DjangoModel(BaseModel):
	"""docstring for MyModel"""

	def name(self):
        return 'Django'
	
	def initialize(self, opt):
		BaseModel.initialize(self, opt)
		self.isTrain = opt.isTrain

	
