import argparse
import numpy as np
import os
from PIL import Image
import scipy.io as sio
import sys
import torch
import torch.nn as nn 

from rasnet_naive import rasnet

def get_arguments():
	parser = argparse.ArgumentParser()

	parser.add_argument('-model_name'     , type=str  , default= 'RAS')

	parser.add_argument('-dataset'        , type=str  , default= 'MSRA-B')

	parser.add_argument('-extname'        , type=str  , default= '.jpg' )

	parser.add_argument('-split'          , type=str  , default= 'test')

	parser.add_argument('-load_path'      , type=str  , default= './run/run_0/models/RAS_epoch-19.pth')

	parser.add_argument('-log_every'      , type=int  , default=100)

	return parser.parse_args()

def main(args):

	if 'RAS' in args.model_name:
		net = rasnet()
	else:
		raise NotImplementedError

	net.load_state_dict(torch.load(args.load_path, map_location=lambda storage, loc: storage))
	torch.cuda.set_device(device=0)
	net.cuda()

	matpath = os.path.join('./dataset', args.dataset, args.split+'ImgSet.mat')
	if args.split == 'all': args.split = 'test'
	matfile = sio.loadmat(matpath)[args.split+'ImgSet']
	fileslist = [matfile[i][0][0] for i in range(matfile.shape[0])]
	fileslist.sort()

	model_name = args.load_path.split('/')[-1]
	pred_dir = 'tmp_pred_' + model_name + '_on_' + args.dataset
	gt_dir = 'tmp_gt_' + model_name + '_on_' + args.dataset 
	if os.path.isdir(pred_dir) == False: 
		os.makedirs(pred_dir)
	else: 
		os.system('rm '+pred_dir+'/*')
	if os.path.isdir(gt_dir) == False: 
		os.makedirs(gt_dir)
	else: 
		os.system('rm '+gt_dir+'/*') 

	EPSILON = 1e-8
	for i in range(len(fileslist)):

		img_name = fileslist[i]
		img_path = os.path.join('./dataset', args.dataset, 'imgs', img_name[:-4]+args.extname)
		img = Image.open(img_path).convert('RGB')
		w, h = img.size 

		im = np.array(img, dtype=np.float32)
		'''
		im /= 255
		im -= np.array((0.485, 0.456, 0.406))
		im /= np.array((0.229, 0.224, 0.225))
		'''
		im = im[:,:,::-1]
		im -= np.array((104.00698793,116.66876762,122.67891434))
		im = im.transpose((2,0,1))
		im = np.expand_dims(im, axis=0)
		im = np.ascontiguousarray(im)
		im = torch.from_numpy(im)
		im = im.cuda()
		outs = net.forward(im)
		outs = nn.Sigmoid()(outs)
		res = outs[0].cpu().data.numpy()
		res = (res - np.min(res) + EPSILON) / (np.max(res) - np.min(res) + EPSILON)
		res = Image.fromarray(np.squeeze(np.rint(res * 255.0).astype(np.uint8)))

		gt_path = os.path.join('./dataset', args.dataset, 'gt', img_name[:-4]+'.png')
		save_name = img_name[:-4]+'.png'
		os.system('cp ' + gt_path + ' ' + gt_dir + '/' + save_name) 
		res.save(pred_dir + '/' + save_name)
		if i % args.log_every == args.log_every-1:
			print('finished %d / %d'%(i, len(fileslist)))

	name_list = [fn for fn in os.listdir(pred_dir) if os.path.isfile(os.path.join(pred_dir, fn))]
	name_list.sort()
	pred_list = [os.path.join(pred_dir, fn) for fn in name_list]
	gt_list = [os.path.join(gt_dir, fn) for fn in name_list]
	sys.path.insert(0, '../SalMetric/build/lib.linux-x86_64-2.7/build')
	import salmetric
	salmetric.do_evaluation(12, pred_list, gt_list)

if __name__=='__main__':
	args = get_arguments()
	main(args)