import argparse
import os
import torch
import random
import numpy as np
from exp.exp_pretrain import Exp_Pretrain

fix_seed = 2025
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)
torch.cuda.manual_seed_all(fix_seed)

parser = argparse.ArgumentParser(description='rsimp-pretrain')

# basic config
parser.add_argument('--task_name', type=str, required=True, default='pretrain')
parser.add_argument('--is_training', type=int, required=True, default=1)
parser.add_argument('--model_id', type=str, required=True, default='test')
parser.add_argument('--model', type=str, required=True, default='MAE')

# data loader
parser.add_argument('--data', type=str, required=True, default='image')
parser.add_argument('--data_path', type=str, default='./dataset/')
parser.add_argument('--log_dir', type=str, default='./logs/')
parser.add_argument('--log_name', type=str, default='result.txt')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--image_size', type=int, default=256)
parser.add_argument('--patch_size', type=int, default=16)
parser.add_argument('--stride', type=int, default=16)
parser.add_argument('--mask_rate', type=float, default=0.2)

# model define
parser.add_argument('--c_in', type=int, default=3)
parser.add_argument('--c_out', type=int, default=3)
parser.add_argument('--d_model', type=int, default=64)
parser.add_argument('--n_heads', type=int, default=8)
parser.add_argument('--e_layers', type=int, default=12)
parser.add_argument('--d_ff', type=int, default=2048)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--num_heads', type=int, default=12)
parser.add_argument('--encoder_layers', type=int, default=12)
parser.add_argument('--decoder_layers', type=int, default=8)
parser.add_argument('--mlp_ratio', type=float, default=4.0)
parser.add_argument('--mask_ratio', type=float, default=0.75)

# optimization
parser.add_argument('--optimizer', type=str, default='muon', help='optimizer type')
parser.add_argument('--num_workers', type=int, default=2, help='data loader num workers')
parser.add_argument('--itr', type=int, default=5, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MAE', help='loss function')
parser.add_argument('--lradj', type=str, default='exp', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

parser.add_argument('--weight', type=float, default=0)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--use_multi_gpu', action='store_true', default=False)
parser.add_argument('--devices', type=str, default='0')

args = parser.parse_args()
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)
if not os.path.exists('./checkpoints/'):
    os.makedirs('./checkpoints/')

print('Args in experiment:')
print(args)

Exp = Exp_Pretrain

if args.is_training:
    for ii in range(args.itr):
        setting = '{}_{}_{}_{}_dm{}_bs{}_lr{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.d_model,
            args.batch_size,
            args.learning_rate,
            args.des, ii
        )
        exp = Exp(args)
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)
        torch.cuda.empty_cache()
else:
    ii = 0
    setting = '{}_{}_{}_{}_dm{}_bs{}_lr{}_{}_{}'.format(
        args.task_name,
        args.model_id,
        args.model,
        args.data,
        args.d_model,
        args.batch_size,
        args.learning_rate,
        args.des, ii
    )
    exp = Exp(args)
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, test=1)
    torch.cuda.empty_cache()
