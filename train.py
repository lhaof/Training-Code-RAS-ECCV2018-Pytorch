import socket
import argparse
import scipy.io as sio
from datetime import datetime
import time
import glob
import os
import random 
from PIL import Image 
import numpy as np 

# PyTorch includes
import torch
from torch.autograd import Variable
from torchvision import transforms 
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

# Tensorboard include
from tensorboardX import SummaryWriter

# Custom includes
# from rasnet import rasnet, load_vgg16conv_fromcaffe, interp_surgery, get_parameters, weights_init
from rasnet_naive import rasnet, load_vgg16conv_fromcaffe, interp_surgery, get_parameters, weights_init

# Dataloaders includes
from dataloaders import msrab, msrab5k 
from dataloaders import utils
from dataloaders import custom_transforms as trforms

def get_arguments():
	parser = argparse.ArgumentParser()

	## Model settings
	parser.add_argument('-model_name'     , type=str  , default= 'RAS')


	## Train settings
	parser.add_argument('-dataset'        , type=str  , default= 'MSRA-B-5k')
	parser.add_argument('-batch_size'     , type=int  , default= 1)
	parser.add_argument('-max_batches'    , type=int  , default=100000)
	parser.add_argument('-resume_batch'   , type=int  , default= 0) 
	parser.add_argument('-max_nepochs'    , type=int  , default=20)
	parser.add_argument('-resume_epoch'   , type=int  , default= 0)
	parser.add_argument('-iter_size'      , type=int  , default=10)
	
	## Optimizer settings
	parser.add_argument('-optimizer'      , type=str  , default='SGD')
	parser.add_argument('-naver_grad'     , type=str  , default=10)
	parser.add_argument('-lr'             , type=float, default=1e-8)
	parser.add_argument('-weight_decay'   , type=float, default=5e-4)
	parser.add_argument('-momentum'       , type=float, default=0.90)
	parser.add_argument('-gamma'          , type=float, default=0.1)
	parser.add_argument('-update_lr_every'            , type=int  , default=200)
	parser.add_argument('-update_lr_every_epochs'     , type=int  , default=15)

	## Visualization settings
	parser.add_argument('-save_every_epochs'     , type=int  , default= 5)
	parser.add_argument('-log_every_iters'       , type=int  , default=20)
	# parser.add_argument('-load_path'      , type=str  , default= 'vgg16-imagenet-from-caffe.pth')
	parser.add_argument('-load_path'      , type=str  , default= 'vgg16.pth')
	parser.add_argument('-run_id'         , type=int  , default=-1)
	parser.add_argument('-use_test'       , type=int  , default= 1)
	return parser.parse_args() 

def main(args):

	save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
	if args.resume_epoch != 0:
		runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
		run_id = int(runs[-1].split('_')[-1]) if runs else 0
	else:
		runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
		run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0

	if args.run_id >= 0:
		run_id = args.run_id

	save_dir = os.path.join(save_dir_root, 'run', 'run_' + str(run_id))
	log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
	writer = SummaryWriter(log_dir=log_dir)


	if 'RAS' in args.model_name:
		net = rasnet()
	else:
		raise NotImplementedError
	

	if args.resume_epoch == 0:
		print('Training ' + args.model_name + ' from scratch...')
		net.apply(weights_init)
		net = load_vgg16conv_fromcaffe(net, args.load_path)
		net = interp_surgery(net) 
	else:
		load_path = os.path.join(save_dir, 'models', args.model_name + '_epoch-' + str(args.resume_epoch - 1) + '.pth')
		if args.load_path != '': load_path = args.load_path 
		print('Initializing weights from: {}...'.format(load_path))
		net.load_state_dict(torch.load(load_path, map_location=lambda storage, loc: storage))

	torch.cuda.set_device(device=0)
	net.cuda()

	optimizer = optim.SGD(
		get_parameters(net),
		lr=args.lr, 
		momentum=args.momentum,
		weight_decay=args.weight_decay
	)
	
	criterion = utils.BCE_2d

	# use vgg16-caffe transformation
	composed_transforms_tr = transforms.Compose([
		# trforms.RandomHorizontalFlip(),
		# trforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
		trforms.RGB2BGR(),
		trforms.MeanNormalize(mean=(104.00699, 116.66877, 122.67892)),
		trforms.ToTensor()])
	
	composed_transforms_ts = transforms.Compose([
		# trforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
		trforms.RGB2BGR(),
		trforms.MeanNormalize(mean=(104.00699, 116.66877, 122.67892)),
		trforms.ToTensor()])

	train_data = msrab5k.MSRAB5K(split='train', transform=composed_transforms_tr)
	val_data = msrab.MSRAB(split='val', transform=composed_transforms_ts, return_size=True)

	trainloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=3)
	testloader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=1)

	num_batch_tr = len(trainloader)
	num_batch_ts = len(testloader)
	num_iter_tr = num_batch_tr / args.iter_size
	num_tr_samples = len(train_data) 
	num_ts_samples = len(val_data) 
	resume_nbatches = args.resume_epoch * num_batch_tr
	resume_nsamples = args.resume_epoch * num_tr_samples

	print('batch number of train set : %d'%(num_batch_tr))
	print('sample number of train set : %d'%(num_tr_samples))
	print('batch number of test set : %d'%(num_batch_ts))
	print('sample number of test set : %d'%(num_ts_samples))
	print('resume training from Batch %d'%(resume_nbatches)) 
	print('resume training from Sample %d'%(resume_nsamples)) 

	cur_batch = resume_nbatches
	cur_sample = resume_nsamples
	cur_iter = int(cur_batch / args.iter_size) 
	cur_lr = args.lr

	aveGrad = 0
	loss_sum_per_epoch = 0
	loss_sum_recent = 0

	start_t = time.time()
	print('Training Network')

	for epoch in range(args.resume_epoch, args.max_nepochs):

		net.train()
		loss_sum_per_epoch = 0
		loss_sum_recent = 0

		for ii, sample_batched in enumerate(trainloader):
			
			inputs, labels = sample_batched['image'], sample_batched['label']
			inputs, labels = Variable(inputs, requires_grad=True), Variable(labels) 
			inputs, labels = inputs.cuda(), labels.cuda()

			#print 'inputs.size: ',inputs.size(),inputs.min(),inputs.max()
			#print 'labels.size: ',labels.size(),labels.min(),labels.max()

			outputs = net.forward(inputs)
			#print 'outputs.size: ',outputs.size(),outputs.min(),outputs.max()
			nrep = outputs.size(0) / labels.size(0)
			assert(labels.size(0) * nrep == outputs.size(0))
			loss = criterion(outputs, labels.repeat(nrep, 1, 1, 1), size_average=False, batch_average=True)

			cur_loss = loss.item()

			loss_sum_per_epoch += cur_loss
			loss_sum_recent += cur_loss 

			# Backward the averaged gradient
			# loss /= args.naver_grad
			loss.backward()

			cur_sample += inputs.data.shape[0] 
			cur_batch += 1
			aveGrad += 1

			# Update the weights once in p['nAveGrad'] forward passes
			if aveGrad % args.iter_size == 0:
				optimizer.step()
				optimizer.zero_grad()
				aveGrad = 0
				cur_iter += 1

				if cur_iter % args.log_every_iters == 0:
					loss_mean_recent = loss_sum_recent / args.log_every_iters / args.iter_size 
					print('epoch: %d iter: %d trainloss: %.2f timecost:%.2f secs'%(
						epoch, cur_iter, loss_mean_recent, time.time()-start_t ))
					writer.add_scalar('data/trainloss', loss_mean_recent, cur_iter)
					loss_sum_recent = 0
 
				# Show 10 * 3 images results each epoch
				if cur_iter % (num_iter_tr // 10) == 0:
					grid_image = make_grid(inputs[:3].clone().cpu().data, 3, normalize=True)
					writer.add_image('Image', grid_image, cur_iter)
					# grid_image = make_grid(utils.decode_seg_map_sequence(torch.max(outputs[:3], 1)[1].detach().cpu().numpy()), 3, normalize=False, range=(0, 255))

					tmp = torch.nn.Sigmoid()(outputs[:1])
					grid_image = make_grid(utils.decode_seg_map_sequence(tmp.narrow(1, 0, 1).detach().cpu().numpy()), 1, normalize=False, range=(0, 255))

					writer.add_image('Predicted label', grid_image, cur_iter)
					grid_image = make_grid(utils.decode_seg_map_sequence(torch.squeeze(labels[:3], 1).detach().cpu().numpy()), 3, normalize=False, range=(0, 255))
					writer.add_image('Groundtruth label', grid_image, cur_iter)

		loss_mean_per_epoch = loss_sum_per_epoch / num_batch_tr 
		print('epoch: %d meanloss: %.2f'%(epoch, loss_mean_per_epoch))
		writer.add_scalar('data/epochloss', loss_mean_per_epoch, cur_iter)


		# The following is to do validation
		if args.use_test == 1:

			net.eval()

			prec_lists = []
			recall_lists = []
			sum_testloss = 0.0
			total_mae = 0.0
			cnt = 0

			rand_id = random.randint(100000, 199999)
			tmp_pred_dir = 'tmp_pred_' + str(rand_id)
			tmp_gt_dir = 'tmp_gt_' + str(rand_id) 
			if os.path.isdir(tmp_pred_dir) == True:
				os.system('rm ' + tmp_pred_dir + '/*')
			else:
				os.makedirs(tmp_pred_dir) 
			if os.path.isdir(tmp_gt_dir) == True:
				os.system('rm ' + tmp_gt_dir + '/*') 
			else:
				os.makedirs(tmp_gt_dir) 

			for ii, sample_batched in enumerate(testloader):
				inputs, labels = sample_batched['image'], sample_batched['label']
				sizes = sample_batched['size'] 

				# Forward pass of the mini-batch
				inputs, labels = Variable(inputs, requires_grad=True), Variable(labels)
				inputs, labels = inputs.cuda(), labels.cuda()

				with torch.no_grad():
					outputs = net.forward(inputs)

				outputs = outputs[:1]
				loss = criterion(outputs, labels, size_average=False, batch_average=False)
				sum_testloss += loss.item()

				predictions = torch.nn.Sigmoid()(outputs)
				
				preds = predictions.data.cpu().numpy()
				gts = labels.data.cpu().numpy()

				for jj in range(preds.shape[0]):
					pred = preds[jj]
					pred = Image.fromarray(np.squeeze(np.rint(pred*255.0).astype(np.uint8))) 
					gt = gts[jj]
					gt = Image.fromarray(np.squeeze(np.rint(gt*255.0).astype(np.uint8))) 
					imsize = sizes[jj]
					imgh, imgw = imsize[0].item(), imsize[1].item() 
					pred = pred.resize((imgw, imgh)) 
					gt = gt.resize((imgw, imgh)) 

					save_name = str(cnt) + '.png' 
					pred.save(os.path.join(tmp_pred_dir, save_name))
					gt.save(os.path.join(tmp_gt_dir, save_name))
					cnt += 1
					if cnt % 100 == 0: print('Tested %d samples / %d'%(cnt, num_ts_samples))

			mean_testloss = sum_testloss / num_batch_ts
			print('Evaluating maxf...')
			os.system('nohup python eval_maxf.py -pred_path='+tmp_pred_dir+' -gt_path='+tmp_gt_dir+' > tmp_'+str(rand_id)+'.out')

			with open('tmp_'+str(rand_id)+'.out', 'r') as f:
				linelist = f.readlines()
			linelist = linelist[-4:]
			results = [x.split()[-1] for x in linelist] 
			print results
			print type(results[0])
			maxf = float(results[0])
			prec = float(results[1])
			recall = float(results[2])
			mae = float(results[3])

			print('Validation:')
			print('epoch: %d, numImages: %d testloss: %.2f mae: %.4f maxf: %.4f prec: %.4f recall: %.4f' % (
				epoch, cnt, mean_testloss, mae, maxf, prec, recall))
			writer.add_scalar('data/validloss', mean_testloss, cur_iter)
			writer.add_scalar('data/validmae', mae, cur_iter)
			writer.add_scalar('data/validfbeta', maxf, cur_iter)

			os.system('rm -rf '+tmp_pred_dir)
			os.system('rm -rf '+tmp_gt_dir) 
			os.system('rm tmp_'+str(rand_id)+'.out')
		# The above finishes validation


		if epoch % args.save_every_epochs == args.save_every_epochs - 1:
			save_path = os.path.join(save_dir, 'models', args.model_name + '_epoch-' + str(epoch) + '.pth')
			torch.save(net.state_dict(), save_path)
			print("Save model at {}\n".format(save_path))

		if epoch % args.update_lr_every_epochs == args.update_lr_every_epochs - 1:
			cur_lr = cur_lr * args.gamma
			print('updated learning rate: ',cur_lr)
			optimizer = optim.SGD(
				get_parameters(net),
				lr=cur_lr,
				momentum=args.momentum,
				weight_decay=args.weight_decay
			)
			writer.add_scalar('data/learningrate', cur_lr, cur_iter)
			'''
			lr_ = utils.lr_poly(args.lr, epoch, args.max_nepochs, 0.9)
			print('(poly lr policy) learning rate: ', lr_)
			optimizer = optim.SGD(
				net.parameters(), 
				lr=lr_, 
				momentum=args.momentum, 
				weight_decay=args.weight_decay
			)
			'''

if __name__ == '__main__':
	args = get_arguments()
	main(args) 