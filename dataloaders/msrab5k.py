import os
import torch
import numpy as np
import scipy.misc as m
from PIL import Image
from torch.utils import data
import scipy.io as sio


class MSRAB5K(data.Dataset):

	def __init__(self, root='./dataset/MSRA-B-5k', split="train", transform=None, return_size=False):

		self.root = root
		self.split = split
		self.transform = transform
		self.return_size = return_size
		self.input_files = {}
		self.label_files = {}
		self.n_classes = 1

		self.input_files[split] = []
		self.label_files[split] = []

		lstpath = os.path.join(self.root, self.split+'.lst')
		with open(lstpath, 'r') as f:
			lines = f.readlines()
			for line in lines:
				split_line = line.split() 
				self.input_files[split].append(split_line[0]) 
				self.label_files[split].append(split_line[1])

		if not self.input_files[split]:
			raise Exception("No input files for split=[%s] found in %s" % (split, self.images_base))
		if not self.label_files[split]:
			raise Exception("No label files for split=[%s] found in %s" % (split, self.images_base))

		print("Found %d %s input images" % (len(self.input_files[split]), split))
		print("Found %d %s label maps" % (len(self.label_files[split]), split))

	def __len__(self):
		assert( len(self.input_files[self.split]) == len(self.label_files[self.split]) )
		return len(self.input_files[self.split])

	def __getitem__(self, index):

		img_name = self.input_files[self.split][index]
		img_path = self.root + img_name
		lbl_name = self.label_files[self.split][index]
		lbl_path = self.root + lbl_name

		_img = Image.open(img_path).convert('RGB')
		_w, _h = _img.size
		_size = (_h, _w)

		_lbl = np.array(Image.open(lbl_path).convert('L'))
		if _lbl.max() > 0:
			_lbl = _lbl / _lbl.max()
		_target = Image.fromarray(_lbl.astype(np.uint8))

		sample = {'image': _img, 'label': _target}
		if self.transform:
			sample = self.transform(sample)
		if self.return_size:
			sample['size'] = torch.tensor(_size)
		return sample


if __name__ == '__main__':
	print os.getcwd()
	# from dataloaders import custom_transforms as tr
	import custom_transforms as tr
	from torch.utils.data import DataLoader
	from torchvision import transforms
	import matplotlib.pyplot as plt

	composed_transforms_tr = transforms.Compose([
		tr.RandomHorizontalFlip(),
		tr.RandomScale((0.5, 0.75)),
		tr.RandomCrop((512, 1024)),
		tr.RandomRotate(5),
		tr.ToTensor()])

	msrab_train= MSRAB5K(split='train',transform=composed_transforms_tr)

	dataloader = DataLoader(msrab_train, batch_size=2, shuffle=True, num_workers=2)

	for ii, sample in enumerate(dataloader):
		print ii, sample["image"].size(), sample["label"].size(), type(sample["image"]), type(sample["label"])
		for jj in range(sample["image"].size()[0]):
			img = sample['image'].numpy()
			gt = sample['label'].numpy()
			tmp = np.array(gt[jj]*255.0).astype(np.uint8)
			tmp = np.squeeze(tmp, axis=0)
			tmp = np.expand_dims(tmp, axis=2)
			segmap = np.concatenate((tmp,tmp,tmp), axis=2)
			img_tmp = np.transpose(img[jj], axes=[1, 2, 0]).astype(np.uint8)
			plt.figure()
			plt.title('display')
			plt.subplot(211)
			plt.imshow(img_tmp)
			plt.subplot(212)
			plt.imshow(segmap)

		if ii == 1:
			break
	plt.show(block=True)
