import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image

class NYUDDataset(BaseDataset):
	"""docstring for NYUDDataset"""
	def initialize(self, opt):
		self.opt = opt
		self.root = opt.dataroot
		self.dir_AB = os.path.join(opt.dataroot, opt.phase)
		self.dir_nyud_img = os.path.join(self.dir_AB, 'img')
		self.dir_nyud_dep = os.path.join(self.dir_AB, 'dep')
		self.nyud_img = sorted(make_dataset(self.dir_nyud_img))
		self.nyud_dep = sorted(make_dataset(self.dir_nyud_dep))
		assert(opt.resize_or_crop == "resize_and_crop")

	def __getitem__(self, index):
		# AB_path = self.AB_paths[index]
		A_path = self.nyud_img[index]
		B_path = self.nyud_dep[index]
		A = Image.open(A_path).convert('RGB')
		B = Image.open(B_path).convert('L')
        # AB = Image.open(AB_path).convert('RGB')
        # w, h = AB.size
        # w2 = int(w / 2)
        A = A.resize((self.opt.loadSize_w, self.opt.loadSize_h), Image.BICUBIC)
        B = B.resize((self.opt.loadSize_w, self.opt.loadSize_h), Image.BICUBIC)
        A = transforms.ToTensor()(A)
        B = transforms.ToTensor()(B)
        # w_offset = random.randint(0, max(0, self.opt.loadSize - self.opt.fineSize - 1))
        # h_offset = random.randint(0, max(0, self.opt.loadSize - self.opt.fineSize - 1))

        # A = A[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]
        # B = B[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]

        A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)
        B = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(B)

        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

		# B is alread gray
        # if output_nc == 1:  # RGB to gray
        #     tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
        #     B = tmp.unsqueeze(0)

        return {'A': A, 'B': B,
                'A_paths': A_path, 'B_paths': B_path}

    # return count of imgs under nyud/train/img (just for example)
    def __len__(self):
        return len(self.nyud_img)

    def name(self):
        return 'NYUDDataset'



