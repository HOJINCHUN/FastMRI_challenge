import torch
import argparse
import os, sys
from pathlib import Path

if os.getcwd() + '/utils/model/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/model/')
from utils.learning.train_part import train

if os.getcwd() + '/utils/common/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/common/')
from utils.common.utils import seed_fix
from mraugment.data_augment import DataAugmentor

def parse():
    parser = argparse.ArgumentParser(description='Train Varnet on FastMRI challenge Images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # 기본 인자
    parser.add_argument('-g', '--GPU-NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('-e', '--num-epochs', type=int, default=1, help='Number of epochs. continued epochs = numepoch - model.pt.epoch')
    parser.add_argument('-l', '--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('-r', '--report-interval', type=int, default=500, help='Report interval')
    parser.add_argument('-n', '--net-name', type=Path, default='test_varnet', help='Name of network')
    parser.add_argument('-p', '--data-path', type=Path, default='/root/Data/trainval/', help='Directory of train&val data')

    # 모델 하이퍼파라미터
    parser.add_argument('--cascade', type=int, default=8, help='Number of cascades')
    parser.add_argument('--chans', type=int, default=18, help='Number of channels for cascade U-Net')
    parser.add_argument('--sens_chans', type=int, default=8, help='Number of channels for sensitivity map U-Net')
    parser.add_argument('--mask-type', type=str,default='random')
    parser.add_argument('--center-fractions', type=float, default='0.04')
    parser.add_argument('--accelerations',type=int,default=8)
    parser.add_argument('--fold-for-val',type=int,default=1,help='the index of fold that we will use for validation set')
    #필요해서 추가
    parser.add_argument('--num_workers', type=int, default=1, help='CPU num workers for parallelization')
    parser.add_argument('--input-key', type=str, default='kspace', help='Name of input key')
    parser.add_argument('--target-key', type=str, default='image_label', help='Name of target key')
    parser.add_argument('--max-key', type=str, default='max', help='Name of max key in attributes')

    # [반영] AdamW와 스케줄러를 위한 인자
    parser.add_argument('--weight-decay', type=float, default=1e-6, help='Weight decay for AdamW optimizer')
    parser.add_argument('--warmup-epochs', type=int, default=10, help='Number of epochs for learning rate warmup')

    # 기타 설정
    parser.add_argument('--seed', type=int, default=430, help='Fix random seed')
    parser.add_argument('--grad-acc', type=int, default=1, help='steps for gradient accumulation')
    parser.add_argument('--set_for_val', type=int,default=1,help='index of fold to set for validation set, you can choose only one')
    parser.add_argument('--index_csv', type=str, default='/root/Data/trainval/index_kfold.csv', help='location of index.csv file')
    parser = DataAugmentor.add_augmentation_specific_args(parser)
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse()
    
    if args.seed is not None:
        seed_fix(args.seed)

    run_dir = Path('../result') / args.net_name
    
    # args 객체에 동적으로 최종 경로 할당
    args.exp_dir = run_dir / 'checkpoints'
    args.val_dir = run_dir / 'reconstructions_val'
    args.val_loss_dir = run_dir

    args.exp_dir.mkdir(parents=True, exist_ok=True)
    args.val_dir.mkdir(parents=True, exist_ok=True)
    
    with open(args.val_loss_dir / 'args.txt', 'w') as f:
        for k, v in sorted(vars(args).items()):
            if isinstance(v, Path):
                f.write(f'{k}: {str(v)}\n')
            else:
                f.write(f'{k}: {v}\n')

    train(args)