import shutil
import numpy as np
import torch
import torch.nn as nn
import time
from pathlib import Path
import copy
from torch.cuda.amp import autocast, GradScaler

from mraugment.data_augment import DataAugmentor
from mraugment.data_transforms import VarNetDataTransform
from utils.model.fastmri.data.subsample import create_mask_for_mask_type
from collections import defaultdict
from utils.data.load_data import create_data_loaders
from utils.common.utils import save_reconstructions, ssim_loss
from utils.common.loss_function import SSIMLoss
from utils.model.varnet import VarNet

from utils.common.custom_mask import BimodalGaussianMaskFunc
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

import os

from torch.cuda.amp import autocast, GradScaler

def train_epoch(args, epoch, model, data_loader, optimizer, loss_type, scaler): 
    model.train() 
    start_epoch = start_iter = time.perf_counter()
    len_loader = len(data_loader)
    total_loss = 0.

    accumulation_steps = args.grad_acc
    optimizer.zero_grad()

    for iter, data in enumerate(data_loader):
        masked_kspace, mask, target, _fname, _slice_num, maximum = data
        masked_kspace = masked_kspace.cuda(non_blocking=True)
        mask           = mask.cuda(non_blocking=True)
        target         = target.cuda(non_blocking=True)
        maximum        = maximum.cuda(non_blocking=True)

        with autocast(): 
            output = model(masked_kspace, mask)
        with autocast(enabled=False):
            loss = loss_type(output.float(), target.float(), maximum.float())
        
        loss = loss / accumulation_steps    
        scaler.scale(loss).backward()
        total_loss += loss.item()

        # gradient accumulation
        if (iter + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            

        if iter % args.report_interval == 0:
            curr_loss = loss.item() * accumulation_steps
            print(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len_loader:4d}] '
                f'Loss = {curr_loss:.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s'
            )
            start_iter = time.perf_counter()

    # 마지막 남은 gradient 처리
    if len_loader % accumulation_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
    avg_loss = total_loss / len_loader
    return avg_loss, time.perf_counter() - start_epoch


def validate(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(dict)
    targets = defaultdict(dict)
    start = time.perf_counter()

    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            masked_kspace, mask, target, fnames, slices, maximum, _crop_size = data
            masked_kspace = masked_kspace.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            output = model(masked_kspace, mask)

            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()
                targets[fnames[i]][int(slices[i])] = target[i].numpy()

    # 정렬 및 스택
    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    for fname in targets:
        targets[fname] = np.stack(
            [out for _, out in sorted(targets[fname].items())]
        )

    # SSIM + L1 가중합
    # 주의: ssim_loss가 "1-SSIM"을 반환한다고 가정. 
    ssim_terms, l1_terms = [], []
    metric_loss = 0.0
    w_ssim = args.ssim_weight
    w_l1   = args.l1_weight
    l1_norm = args.normalize  # 'target_max' | 'minmax' | 'none'

    for fname in reconstructions:
        ssim_term = ssim_loss(targets[fname], reconstructions[fname])
        l1_term = l1_loss_np(targets[fname], reconstructions[fname], normalize=l1_norm)

        ssim_terms.append(ssim_term)
        l1_terms.append(l1_term)

        metric_loss += w_ssim * ssim_term + w_l1 * l1_term

    num_subjects = len(reconstructions)

    # 평균값도 로그용으로 리턴하고 싶다면: (원형 함수 시그니처를 바꾸기 어렵다면 None 자리에 dict로 넣어도 됨)
    metrics = {
        'mean_ssim_loss': float(np.mean(ssim_terms)) if len(ssim_terms) else 0.0,
        'mean_l1_loss': float(np.mean(l1_terms)) if len(l1_terms) else 0.0,
    }

    torch.cuda.empty_cache()
    return metric_loss, num_subjects, reconstructions, targets, metrics, time.perf_counter() - start


def save_model(args, exp_dir, epoch, model, optimizer, best_val_loss, is_new_best,
               scheduler=None, scaler=None):
    payload = {
        'epoch': epoch,
        'args': args,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
        'exp_dir': exp_dir
    }
    if scheduler is not None:
        payload['scheduler'] = scheduler.state_dict()
    if scaler is not None:
        payload['scaler'] = scaler.state_dict()

    torch.save(payload, f=exp_dir / 'model.pt')
    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')

        
def train(args):
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print('Current cuda device: ', torch.cuda.current_device())

    #1. 마지막 model.pt checkpoint로 불러오기
    ckpt_path = args.exp_dir / 'model.pt'
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    model = VarNet(num_cascades=args.cascade, 
                   chans=args.chans, 
                   sens_chans=args.sens_chans)
    model.to(device)
    model.load_state_dict(checkpoint['model'])
    
    
    # 2. [반영] Augmentor의 에포크 참조 방식 수정
    epoch_holder = {'cur': 0}
    current_epoch_fn = lambda: epoch_holder['cur']
    augmentor = DataAugmentor(args, current_epoch_fn)

    #mask_train 정의 위치 수정
    mask_train = BimodalGaussianMaskFunc(
            center_fractions=[0.08, 0.04],
            min_accel=4,
            max_accel=12,
            mean1=4.0, stddev1=1.5,
            mean2=8.0, stddev2=1.5,
            mix_weight1=0.4, 
            current_epoch_fn=current_epoch_fn, args=args
        )
    
    # 4. [반영] PyTorch 네이티브 스케줄러 및 AdamW 옵티마이저 적용
    loss_type = SSIMLoss().to(device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay, eps=1e-06
    )
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    warmup_scheduler = LinearLR(optimizer, start_factor=1e-2, end_factor=1.0, total_iters=args.warmup_epochs)
    main_scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs - args.warmup_epochs, eta_min=1e-7)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[args.warmup_epochs])

    scaler=GradScaler()
    
    start_epoch = checkpoint['epoch']
    best_val_loss = checkpoint['best_val_loss']

    #scheduler 및 scaler 기존 값으로 교체
    if 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
    if 'scaler' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler'])
        
    train_loader = create_data_loaders(args.data_path, args, augmentor, mask_train, shuffle = True, use_split="train")
    val_loader = create_data_loaders(args.data_path, args, augmentor, use_split="val")

 
    # 기존 로그 정보가 만약 있다면 가져기기
    os.makedirs(args.run_dir, exist_ok=True)
    v_file_path = os.path.join(args.run_dir, "val_loss_log.npy")
    t_file_path = os.path.join(args.run_dir, "train_loss_log.npy")

    if os.path.exists(v_file_path):
        val_loss_log = np.load(v_file_path)
        # 2열(epoch, loss) 보장
        if val_loss_log.ndim == 1:
            val_loss_log = val_loss_log.reshape(-1, 2)
    else:
        val_loss_log = np.empty((0, 2), dtype=np.float64)
    
    if os.path.exists(t_file_path):
        train_loss_log = np.load(t_file_path)
        if train_loss_log.ndim == 1:
            train_loss_log = train_loss_log.reshape(-1, 2)
    else:
        train_loss_log = np.empty((0, 2), dtype=np.float64)

        
    for epoch in range(start_epoch, args.num_epochs):
        print(f'Epoch #{epoch:2d} ............... {args.net_name} ...............')
        epoch_holder['cur'] = epoch

        train_loss, train_time = train_epoch(args, epoch, model, train_loader, optimizer, loss_type, scaler=scaler)
        val_loss, num_subjects, reconstructions, targets, inputs, val_time = validate(args, model, val_loader)
        
        val_loss_log = np.append(val_loss_log, np.array([[epoch, val_loss]]), axis=0)
        train_loss_log = np.append(train_loss_log, np.array([[epoch,train_loss]]),axis=0)

        np.save(v_file_path, val_loss_log)
        np.save(t_file_path, train_loss_log)
        print(f"val loss file saved! {v_file_path}")
        print(f"train loss file saved! {t_file_path}")
        
        train_loss = torch.tensor(train_loss).cuda(non_blocking=True)
        val_loss = torch.tensor(val_loss).cuda(non_blocking=True)
        num_subjects = torch.tensor(num_subjects).cuda(non_blocking=True)

        val_loss = val_loss / num_subjects

        is_new_best = val_loss < best_val_loss
        best_val_loss = min(best_val_loss, val_loss)

        save_model(args, args.exp_dir, epoch + 1, model, optimizer, best_val_loss, is_new_best, scheduler, scaler)
        
        # [반영] 스케줄러 업데이트 및 로그 출력
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        print(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] '
            f'TrainLoss = {train_loss:.4g} ValLoss = {val_loss:.4g} '
            f'CurrentLR = {current_lr:.4e} '
            f'TrainTime = {train_time:.4f}s ValTime = {val_time:.4f}s'
        )
        print(f"Max memory: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")

        if is_new_best:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@NewRecord@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            start = time.perf_counter()
            save_reconstructions(reconstructions, args.val_dir, targets=targets, inputs=inputs)
            print(
                f'ForwardTime = {time.perf_counter() - start:.4f}s',
            )
