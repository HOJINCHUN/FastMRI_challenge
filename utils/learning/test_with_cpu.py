import numpy as np
import torch

from collections import defaultdict
from utils.common.utils import save_reconstructions
from utils.data.load_data import create_data_loaders
from utils.model.varnet import VarNet
from utils.model.fastmri.data.subsample import create_mask_for_mask_type

def test(args, model, data_loader, device):
    model.eval()
    reconstructions = defaultdict(dict)
    
    with torch.no_grad():
        for (masked_kspace, mask, _target, fnames, slices, _max_value, _crop_size,) in data_loader:
            masked_kspace = masked_kspace.to(device, non_blocking=False)
            mask = mask.to(device, non_blocking=False)
            output = model(masked_kspace, mask)

            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    return reconstructions, None


def forward(args):
    device = torch.device('cpu')
    print ('running on cpu')

    model = VarNet(num_cascades=args.cascade, 
                   chans=args.chans, 
                   sens_chans=args.sens_chans)
    model.to(device=device)
    
    checkpoint = torch.load(args.exp_dir / 'best_model.pt', map_location='cpu', weights_only=False)
    print(checkpoint['epoch'], checkpoint['best_val_loss'].item())
    model.load_state_dict(checkpoint['model'])
    
    
    forward_loader = create_data_loaders(data_path = args.data_path, args = args, isforward = True)
    reconstructions, inputs = test(args, model, forward_loader, device)
    save_reconstructions(reconstructions, args.forward_dir, inputs=inputs)