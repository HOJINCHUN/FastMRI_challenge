import argparse
from pathlib import Path
import os, sys
if os.getcwd() + '/utils/model/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/model/')

from utils.learning.test_with_cpu import forward
import time

    
def parse():
    parser = argparse.ArgumentParser(description='Test Varnet on FastMRI challenge Images',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--GPU_NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('-n', '--net_name', type=Path, default='test_Varnet', help='Name of network')
    parser.add_argument('-p', '--path_data', type=Path, default='/root/Data/samples', help='Directory of test data')

    parser.add_argument('--cascade', type=int, default=11, help='Number of cascades | Should be less than 12')
    parser.add_argument('--chans', type=int, default=24, help='Number of channels for cascade U-Net')
    parser.add_argument('--sens_chans', type=int, default=8, help='Number of channels for sensitivity map U-Net')
    parser.add_argument("--input_key", type=str, default='kspace', help='Name of input key')

    #0811 추가된 변수
    parser.add_argument('--fold-for-val',type=int,default=1,help='the index of fold that we will use for validation set')
    parser.add_argument('--num_workers', type=int, default=6, help='CPU num workers for parallelization')
    #중요! create-data-loaders에서 index_csv를 none으로 설정해야 test 데이터를 정상적으로 load
    parser.add_argument('--index_csv', type=str, default=None, help='location of index.csv file')
    parser.add_argument('--seed', type=int,default=430)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse()
    args.exp_dir = '../result' / args.net_name / 'checkpoints'

    start_time = time.time()
    
    args.data_path = args.path_data / "acc4"
    args.forward_dir = '../result' / args.net_name / 'sample_results' / "acc4"
    print(args.forward_dir)
    forward(args)
    
    # acc8
    args.data_path = args.path_data / "acc8"
    args.forward_dir = '../result' / args.net_name / 'sample_results' / "acc8"
    print(args.forward_dir)
    forward(args)

    reconstructions_time = time.time() - start_time
    print(f'Total Reconstruction Time = {reconstructions_time:.2f}s')

    print('Success!')