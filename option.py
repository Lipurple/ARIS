# 2021.05.07-Changed for IPT
#            Huawei Technologies Co., Ltd. <foss@huawei.com>

import argparse

parser = argparse.ArgumentParser(description='IPT')

parser.add_argument('--debug', action='store_true',
                    help='Enables debug mode')
parser.add_argument('--template', default='.',
                    help='You can set various templates in option.py')

# Hardware specifications
##todo consider the num_worker of the dataloader,之前报了“OSError: [Errno 12] Cannot allocate memory”这个错，网上搜的解决办法是把num_worker设为0
parser.add_argument('--n_threads', type=int, default=2,
                    help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',
                    help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=2,
                    help='number of GPUs')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')

# Data specifications
parser.add_argument('--dir_data', type=str, default='/GPFS/data/qinyezhou/Pretrained-IPT/pretrain_model/',
                    help='dataset directory')
parser.add_argument('--dir_demo', type=str, default='../test',
                    help='demo image directory')
parser.add_argument('--data_train', type=str, default='DIV2K',
                    help='train dataset name')
parser.add_argument('--data_test', type=str, default='DIV2K',
                    help='test dataset name')
parser.add_argument('--data_range', type=str, default='1-800/801-810',
                    help='train/test data range')
parser.add_argument('--ext', type=str, default='sep',
                    help='dataset file extension')
parser.add_argument('--asymm', type=bool, default=False,
                    help='use asymmetric scale factors (only used during training phase)')
parser.add_argument('--scaleh', type=str, default='',
                    help='super resolution scale')
parser.add_argument('--scalew', type=str, default='',
                    help='super resolution scale2')
parser.add_argument('--scale', type=str, default='2',
                    help='super resolution scale')
parser.add_argument('--patch_size', type=int, default=48,
                    help='output patch size')
parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--no_augment', action='store_true',
                    help='do not use data augmentation')

# Model specifications
parser.add_argument('--model', default='ipt',
                    help='model name')
parser.add_argument('--n_feats', type=int, default=64,
                    help='number of feature maps')
parser.add_argument('--shift_mean', default=True,
                    help='subtract pixel mean from the input')
parser.add_argument('--precision', type=str, default='single',
                    choices=('single', 'half'),
                    help='FP precision for test (single | half)')

# Training specifications
parser.add_argument('--reset', action='store_true',
                    help='reset the training')
parser.add_argument('--test_every', type=int, default=1000,
                    help='do test per every N batches')
parser.add_argument('--epochs', type=int, default=3000,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='input batch size for training')
parser.add_argument('--test_batch_size', type=int, default=1,
                    help='input batch size for training')
parser.add_argument('--crop_batch_size', type=int, default=64,
                    help='input batch size for training')
parser.add_argument('--split_batch', type=int, default=1,
                    help='split the batch into smaller chunks')
parser.add_argument('--self_ensemble', action='store_true',
                    help='use self-ensemble method for test')
parser.add_argument('--test_only', action='store_true',
                    help='set this option to test the model')
parser.add_argument('--gan_k', type=int, default=1,
                    help='k value for adversarial loss')

# Optimization specifications
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--decay', type=str, default='200',
                    help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999),
                    help='ADAM beta')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')
parser.add_argument('--gclip', type=float, default=0,
                    help='gradient clipping threshold (0 = no clipping)')

# Loss specifications
parser.add_argument('--loss', type=str, default='1*L1',
                    help='loss function configuration')
parser.add_argument('--skip_threshold', type=float, default='1e8',
                    help='skipping batch that has large error')

# Log specifications
parser.add_argument('--save', type=str, default='try/',
                    help='file name to save')
parser.add_argument('--load', type=str, default='',
                    help='file name to load')
parser.add_argument('--resume', type=int, default=0,
                    help='resume from specific checkpoint')
parser.add_argument('--save_models', action='store_true',
                    help='save all intermediate models')
parser.add_argument('--print_every', type=int, default=100,
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_results', action='store_true',
                    help='save output results')
parser.add_argument('--save_gt', action='store_true',
                    help='save low-resolution and high-resolution images together')

parser.add_argument('--base_ckpt', type=str, default='',
                    help='base model checkpoint')

#cloud
parser.add_argument('--moxfile', type=int, default=1)
parser.add_argument('--data_url', type=str,help='path to dataset')
parser.add_argument('--train_url', type=str, help='train_dir')
parser.add_argument('--pretrain', type=str, default='/GPFS/data/ziyili/video/code/IPT_try/IPT_sr2.pt')
parser.add_argument('--load_query', type=int, default=0)

#transformer
parser.add_argument('--patch_dim', type=int, default=3)
parser.add_argument('--num_heads', type=int, default=12)
parser.add_argument('--num_layers', type=int, default=12)
parser.add_argument('--dropout_rate', type=float, default=0)
parser.add_argument('--no_norm', action='store_true')
parser.add_argument('--freeze_norm', action='store_true')
parser.add_argument('--post_norm', action='store_true')
parser.add_argument('--no_mlp', action='store_true')
parser.add_argument('--pos_every', action='store_true')
parser.add_argument('--no_pos', action='store_true')
parser.add_argument('--num_queries', type=int, default=1)

#denoise
parser.add_argument('--denoise', action='store_true')
parser.add_argument('--sigma', type=float, default=30)

#derain
parser.add_argument('--derain', action='store_true')
parser.add_argument('--derain_test', type=int, default=1)

#deblur
parser.add_argument('--deblur', action='store_true')
parser.add_argument('--deblur_test', type=int, default=1)


args, unparsed = parser.parse_known_args()

args.scale = list(map(lambda x: int(x), args.scale.split('+')))
args.data_train = args.data_train.split('+')
args.data_test = args.data_test.split('+')

if args.scaleh=='' or args.scalew=='':
    # asymmetric mode: non-integer scale factors + asymmetric scale factors
    if args.asymm:
        args.scaleh = [
            1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
            2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0,
            3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0,
            1.5, 1.5, 1.5, 1.5, 1.5,
            2.0, 2.0, 2.0, 2.0, 2.0,
            2.5, 2.5, 2.5, 2.5, 2.5,
            3.0, 3.0, 3.0, 3.0, 3.0,
            3.5, 3.5, 3.5, 3.5, 3.5,
            4.0, 4.0, 4.0, 4.0, 4.0,
        ]
        args.scalew = [
            1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
            2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0,
            3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0,
            2.0, 2.5, 3.0, 3.5, 4.0,
            1.5, 2.5, 3.0, 3.5, 4.0,
            1.5, 2.0, 3.0, 3.5, 4.0,
            1.5, 2.0, 2.5, 3.5, 4.0,
            1.5, 2.0, 2.5, 3.0, 4.0,
            1.5, 2.0, 2.5, 3.0, 3.5,
        ]
    # symmetric mode: only non-integer scale factors
    else:
        args.scaleh = [2.0, 3.0, 4.0
                    #    1.5, 2.5, 3.5
            # 1.5, 2.0, 2.5, 3.0, 3.5, 4.0
            # 1.2, 1.5, 1.8, 2.0, 2.2, 2.5, 2.8, 3.0, 3.2, 3.5, 3.8, 4.0
            # 2.0, 3.0, 4.0, 6.0, 8.0
            # 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
            # 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0,
            # 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0
            # 1.3, 1.7, 2.0, 2.3, 2.7, 3.0, 3.3, 3.7, 4.0
            # 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0,
            # 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0
            # 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7.0,
            # 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8.0
        ]
        args.scalew = args.scaleh
else:
    args.scaleh = list(map(lambda x: float(x), args.scaleh.split('+')))
    args.scalew = list(map(lambda x: float(x), args.scalew.split('+')))

if args.epochs == 0:
    args.epochs = 1e8
    
args.scalenew = [2.0, 3.0, 4.0, 6.0, 8.0, 12.0]

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False

