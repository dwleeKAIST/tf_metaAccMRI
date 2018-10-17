import argparse
import os
from util import util


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--model', type=str, default='Gnet_', help='Unet,Gnet,Unet_ch,Gnet_ch/myNet_ch')
        self.parser.add_argument('--dataroot',type=str, default='./../../mrdata/T1w_pad_8ch_x2_halfnY', help='path for dataset')
        self.parser.add_argument('--dataset', type=str, default='7T', help='HCP/7T')
        self.parser.add_argument('--nEpoch', type=int, default=30000, help='number of Epoch iteration')
        self.parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
        self.parser.add_argument('--lr_state', type=float, default=0.0005, help='learning rate for state')
        self.parser.add_argument('--disp_div_N', type=int, default=10, help=' display N per epoch')
        self.parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        self.parser.add_argument('--input_nc', type=int, default=2, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=2, help='# of output image channels')
        self.parser.add_argument('--ngf', type=int, default=16, help='# of gen filters in first conv layer')
        self.parser.add_argument('--DSrate', type=int, default=2, help='Down Sampling Rate')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2.')
        self.parser.add_argument('--mask', type=str, default='Unif3p1_ACS10p',help='mask file name')
        self.parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--dropout', type=float, default=0., help='keep ratio- dropout') 
        self.parser.add_argument('--optimizer', type=str, default='Adam', help='Adam/RMSp') 
        self.parser.add_argument('--k_shot', type=int, default=1, help='k-shot int')
        self.parser.add_argument('--nEpoch_state_update', type=int, default=1, help='number of step for state update')
        self.parser.add_argument('--nEpoch_Wb_update', type=int, default=1, help='number of step for W and b for RNN update')
        self.parser.add_argument('--nHidden', type=int, default=100, help='nHidden for LSTM int')
        self.parser.add_argument('--debug_mode', action='store_true', help='debug mode using 1 batch')
        self.parser.add_argument('--smallDB', action='store_true', help='use 1/10 of DB')
        self.parser.add_argument('--use_iloss', action='store_true', help='use img-space loss')
        self.parser.add_argument('--clip', type=float, default=0.1, help='gradient clip')
        self.parser.add_argument('--lambda_loss', type=float, default=1., help='amplification of loss')
        self.parser.add_argument('--w_decay', type=float, default=0, help='weight decay for regularization')
        self.parser.add_argument('--use_kproj', action='store_true', help='k-space projection using ACS')
        self.parser.add_argument('--test_mode', action='store_true', help='test mode, save images')
        self.parser.add_argument('--Aug', action='store_true', help='Use Augmentation by scaling default false')
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        #self.opt.isTrain = self.isTrain   # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        #if len(self.opt.gpu_ids) > 0:
        #    torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        expr_dir = os.path.join('./result', self.opt.name)
        if not os.path.exists(expr_dir):
            os.makedirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt
