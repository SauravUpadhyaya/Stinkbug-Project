"""
Smoke test for CBAM + ASFF modules.

Creates synthetic feature maps similar to YOLO neck outputs and runs forward +
backward to ensure modules are CPU-friendly and gradients flow.
"""
import argparse
import torch
from model_mods import CBAM, ASFF


def run_smoke_test(device='cpu'):
    """Run a small smoke test on device ('cpu' or 'cuda').

    This creates synthetic feature maps, moves modules to device, runs forward
    and backward to validate gradients flow on GPU.
    """
    print('Running CBAM + ASFF smoke test on', device)

    # Basic CUDA safety: empty cache and enable cuDNN autotune for convs
    if device.startswith('cuda') and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True

    # synthetic feature maps with common pattern: (B, C, H, W)
    # choose channels similar to small/mid/large YOLO feature maps
    x_small = torch.randn(2, 128, 80, 80, device=device, requires_grad=True)
    x_mid = torch.randn(2, 256, 40, 40, device=device, requires_grad=True)
    x_large = torch.randn(2, 512, 20, 20, device=device, requires_grad=True)

    # instantiate modules and move them to device
    cbam_mid = CBAM(256).to(device)
    asff = ASFF([128, 256, 512], out_channels=256).to(device)

    # forward
    cb_out = cbam_mid(x_mid)
    fused = asff([x_small, cb_out, x_large])
    print('cb_out shape:', cb_out.shape)
    print('fused shape:', fused.shape)

    # simple loss and backward
    # cb_out is not a leaf tensor, so its .grad won't be populated unless we retain it
    cb_out.retain_grad()
    loss = fused.sum() + cb_out.sum()

    # Ensure loss is differentiable
    if not getattr(loss, 'requires_grad', False) or getattr(loss, 'grad_fn', None) is None:
        raise RuntimeError('Loss is not differentiable on device={} (check autograd and device placement)'.format(device))

    loss.backward()

    if cb_out.grad is not None:
        print('Backward completed. Gradients: cb_out grad sum', cb_out.grad.abs().sum().item())
    else:
        print('Backward completed but cb_out.grad is None (unexpected)')

    # also print a couple of parameter grads as a sanity check
    try:
        p = next(cbam_mid.parameters())
        if p.grad is not None:
            print('Sample cbam param grad sum:', p.grad.abs().sum().item())
        else:
            print('cbam first param has no grad')
    except StopIteration:
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help="Device to run on, e.g. 'cpu' or 'cuda:0'")
    args = parser.parse_args()
    # normalize device string
    dev = args.device
    if dev.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA device but CUDA is not available')
    run_smoke_test(dev)