import sys
caffe_root = '../caffe_dss-master/'
sys.path.insert(0, caffe_root + 'python')
import caffe
import torch

caffe.set_mode_cpu()
solver = caffe.SGDSolver('solver.prototxt')
solver.net.copy_from('vgg16.caffemodel')
vgg_keys = solver.net.params.keys()
vgg_keys = [ x for x in vgg_keys if ( x.startswith('conv') and ('relu' not in x) and ('dsn' not in x) ) ]
print vgg_keys

weights_dict = {}
for k in vgg_keys:
	weights_dict[k + '_weight'] = torch.from_numpy(solver.net.params[k][0].data)
	weights_dict[k + '_bias'] = torch.from_numpy(solver.net.params[k][1].data)
torch.save(weights_dict, 'vgg16.pth')
print 'done'