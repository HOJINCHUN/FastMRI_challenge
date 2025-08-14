python train.py \
  -b 1 \
  -e 50 \
  -l 0.0003 \
  -n 'test_Varnet' \
  --num_workers 6 \
  --cascade 11 \
  --chans 24 \
  --mask-type 'equispaced' \
  --grad-acc 2 \
  --aug_on \
  --aug_delay 1 \
  --aug_strength 0.55 \
  --set_for_val 1 \
  
  