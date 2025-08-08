# [반영] 필요한 모든 라이브러리 임포트
import shutil
import numpy as np
import torch
import torch.nn as nn
import time
from pathlib import Path
from collections import defaultdict
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

from utils.common.utils import save_reconstructions, ssim_loss
from utils.common.loss_function import SSIMLoss
from utils.model.varnet import VarNet

# [반영] MRAugment 및 최신 데이터 파이프라인 관련 임포트
from mraugment.data_augment import DataAugmentor
from mraugment.data_transform import VarNetDataTransform
from mraugment.fastmri_data_module import FastMriDataModule
from fastmri.data.subsample import create_mask_for_mask_type
from utils.common.custom_mask import BimodalGaussianMaskFunc # 커스텀 마스크 클래스

import os

# [수정] train_epoch 함수 내 데이터 언패킹 변수명 오류 수정
def train_epoch(args, epoch, model, data_loader, optimizer, loss_type):
    model.train()
    start_epoch = start_iter = time.perf_counter()
    len_loader = len(data_loader)
    total_loss_val = 0.

    accumulation_steps = args.grad_acc
    optimizer.zero_grad()

    for iter, data in enumerate(data_loader):
        # [수정] 데이터 로더의 출력에 맞게 변수명 수정 (kspace -> masked_kspace)
        masked_kspace, mask, target, _, _, maximum, _ = data
        mask = mask.cuda(non_blocking=True)
        masked_kspace = masked_kspace.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        maximum = maximum.cuda(non_blocking=True)

        output = model(masked_kspace, mask)
        loss = loss_type(output, target, maximum)
        
        # [수정] L1 정규화 로직 제거 (필요 시 추가)
        # L1 정규화는 AdamW의 weight_decay와 역할이 겹칠 수 있어 일단 제거하는 것을 추천
        
        loss = loss / accumulation_steps
        loss.backward()
        total_loss_val += loss.item()

        if (iter + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        if iter % args.report_interval == 0:
            print(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item() * accumulation_steps:.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
            start_iter = time.perf_counter()

    if len(data_loader) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    total_loss_val = total_loss_val / len_loader
    return total_loss_val, time.perf_counter() - start_epoch

# [수정] validate 함수 내 데이터 언패킹 변수명 오류 수정
def validate(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(dict)
    targets = defaultdict(dict)
    start = time.perf_counter()

    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            # [수정] 데이터 로더의 출력에 맞게 변수명 수정 및 추가 (fnames, slices)
            masked_kspace, mask, target, fnames, slices, _, _ = data
            masked_kspace = masked_kspace.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            output = model(masked_kspace, mask)

            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()
                targets[fnames[i]][int(slices[i])] = target[i].numpy()

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    for fname in targets:
        targets[fname] = np.stack(
            [out for _, out in sorted(targets[fname].items())]
        )
    metric_loss = sum([ssim_loss(targets[fname], reconstructions[fname]) for fname in reconstructions])
    num_subjects = len(reconstructions)
    return metric_loss, num_subjects, reconstructions, targets, None, time.perf_counter() - start


def save_model(args, exp_dir, epoch, model, optimizer, best_val_loss, is_new_best):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'exp_dir': exp_dir
        },
        f=exp_dir / 'model.pt'
    )
    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')


def train(args):
    # 1. 초기 설정
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print('Current cuda device: ', torch.cuda.current_device())

    model = VarNet(
        num_cascades=args.cascade,
        chans=args.chans,
        sens_chans=args.sens_chans
    )
    model.to(device=device)

    # 2. [반영] Augmentor의 에포크 참조 방식 수정
    epoch_holder = {'cur': 0}
    current_epoch_fn = lambda: epoch_holder['cur']
    augmentor = DataAugmentor(args, current_epoch_fn)

    # 3. [반영] 훈련용/검증용 마스크 역할을 명확히 분리
    mask_train = BimodalGaussianMaskFunc(
        center_fractions=[0.08, 0.04],
        min_accel=4,
        max_accel=12,
        mean1=4.0, stddev1=1.5,
        mean2=8.0, stddev2=1.5,
        mix_weight1=0.4
    )
    val_path_str = str(args.data_path_val)
    if 'acc4' in val_path_str:
        print("Validation path contains 'acc4'. Creating fixed 4x mask.")
        mask_val = create_mask_for_mask_type('equispaced', [0.08], [4])
    elif 'acc8' in val_path_str:
        print("Validation path contains 'acc8'. Creating fixed 8x mask.")
        mask_val = create_mask_for_mask_type('equispaced', [0.04], [8])
    else:
        print("Validation path does not specify acc rate. Defaulting to fixed 8x mask.")
        mask_val = create_mask_for_mask_type('equispaced', [0.04], [8])
    
    # 4. [반영] PyTorch 네이티브 스케줄러 및 AdamW 옵티마이저 적용
    loss_type = SSIMLoss().to(device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    warmup_scheduler = LinearLR(optimizer, start_factor=1e-10, end_factor=1.0, total_iters=args.warmup_epochs)
    main_scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs - args.warmup_epochs, eta_min=1e-7)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[args.warmup_epochs])

    # 5. [반영] MRAugment의 최신 데이터 파이프라인 적용
    train_transform = VarNetDataTransform(augmentor=augmentor, mask_func=mask_train, use_seed=False)
    val_transform = VarNetDataTransform(mask_func=mask_val)
    test_transform = VarNetDataTransform()
    
    data_module = FastMriDataModule(
        data_path_train=args.data_path_train,
        data_path_val=args.data_path_val,
        test_path=args.test_path,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        use_dataset_cache_file=args.use_dataset_cache_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        distributed_sampler=(args.accelerations in ("ddp", "ddp_cpu")),
    )

    # 6. [반영] 데이터 로더를 FastMriDataModule에서 가져오도록 수정
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    
    # 7. 로그 및 상태 변수 초기화
    best_val_loss = 1.
    start_epoch = 0
    val_loss_log = np.empty((0, 2))
    train_loss_log = np.empty((0, 2))
    is_dir_created = False

    # 8. 메인 학습 루프
    for epoch_idx in range(start_epoch, args.num_epochs):
        epoch_holder['cur'] = epoch_idx
        
        print(f'Epoch #{epoch_idx:2d} ............... {args.net_name} ...............')

        train_loss, train_time = train_epoch(args, epoch_idx, model, train_loader, optimizer, loss_type)
        val_loss, num_subjects, reconstructions, targets, inputs, val_time = validate(args, model, val_loader)

        # [반영] 첫 에포크 후 폴더 생성 로직
        if not is_dir_created:
            print(f"✅ First epoch successful. Creating result directory at: {args.val_loss_dir}")
            args.exp_dir.mkdir(parents=True, exist_ok=True)
            args.val_dir.mkdir(parents=True, exist_ok=True)
            with open(args.val_loss_dir / 'args.txt', 'w') as f:
                for k, v in sorted(vars(args).items()):
                    if isinstance(v, Path):
                        f.write(f'{k}: {str(v)}\n')
                    else:
                        f.write(f'{k}: {v}\n')
            is_dir_created = True
        
        # 로그 저장
        train_loss_log = np.append(train_loss_log, np.array([[epoch_idx, train_loss]]), axis=0)
        val_loss_log = np.append(val_loss_log, np.array([[epoch_idx, val_loss]]), axis=0)
        v_file_path = os.path.join(args.val_loss_dir, "val_loss_log.npy")
        l_file_path = os.path.join(args.val_loss_dir, "train_loss_log.npy")
        np.save(v_file_path, val_loss_log)
        np.save(l_file_path, train_loss_log)
        print(f"Loss files saved!\n   - Train: {l_file_path}\n   - Val:   {v_file_path}")

        # 성능 평가 및 모델 저장
        train_loss = torch.tensor(train_loss).cuda(non_blocking=True)
        val_loss = torch.tensor(val_loss).cuda(non_blocking=True)
        num_subjects = torch.tensor(num_subjects).cuda(non_blocking=True)
        val_loss = val_loss / num_subjects

        is_new_best = val_loss < best_val_loss
        best_val_loss = min(best_val_loss, val_loss)
        save_model(args, args.exp_dir, epoch_idx + 1, model, optimizer, best_val_loss, is_new_best)
        print(f"Max memory: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
        
        # [반영] 스케줄러 업데이트 및 로그 출력
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        print(
            f'Epoch = [{epoch_idx:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'ValLoss = {val_loss:.4g} CurrentLR = {current_lr:.4e} TrainTime = {train_time:.4f}s ValTime = {val_time:.4f}s',
        )

        if is_new_best:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@NewRecord@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            start = time.perf_counter()
            save_reconstructions(reconstructions, args.val_dir, targets=targets, inputs=inputs)
            print(f'ForwardTime = {time.perf_counter() - start:.4f}s')