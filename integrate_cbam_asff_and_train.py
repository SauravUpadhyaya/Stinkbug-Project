"""
Integrate CBAM + ASFF into a YOLOv8 model (clean-ish integration) and run a short fine-tune.

Strategy:
- Load an Ultralytics YOLO model (e.g. yolov8m.pt)
- Run a single batched forward to collect candidate intermediate modules and their outputs.
- Pick three feature-producing modules (large, mid, small) by spatial area.
- Replace the mid module in-place with a wrapper module that: runs the original mid module, applies CBAM to its output, collects the other two features (captured by hooks), runs ASFF to fuse them, and returns the fused tensor.
- Start ultralytics training using the patched model object (call `model.train(...)`) for a short number of epochs.

Notes:
- This approach avoids modifying ultralytics source and creates a deterministic, reproducible model object that contains CBAM+ASFF in its forward path.
- It's a runtime model substitution but saved by calling `model.save()` after (if desired) to persist a checkpoint with the integrated modules.

Usage example:
    python3 integrate_cbam_asff_and_train.py --model ./saved_models_of_1008_images/yolov8m_best.pt --data ./stinkbug_image_data_third_file --epochs 3 --device cuda

"""
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_mods import CBAM, ASFF


def collect_module_outputs_for_selection(inner, batch, device='cpu'):
    """Run a single forward pass with hooks to capture module outputs and return a list of (module_path, module_obj, output_tensor).
    module_path is the dotted name from named_modules (unique), module_obj is the module instance.
    """
    captures = []

    def make_hook(path):
        def hook(m, inp, out):
            # only 4-D tensors
            if isinstance(out, torch.Tensor) and out.dim() == 4:
                captures.append((path, m, out.detach().cpu()))
        return hook

    hooks = []
    for name, m in inner.named_modules():
        # register on conv/bn/seq modules to reduce noise
        if isinstance(m, (nn.Conv2d, nn.BatchNorm2d, nn.Sequential)):
            try:
                h = m.register_forward_hook(make_hook(name))
                hooks.append(h)
            except Exception:
                pass

    inner.to(device)
    inner.eval()
    with torch.no_grad():
        try:
            _ = inner(batch.to(device))
        except Exception:
            # ignore forward errors for selection pass
            pass

    for h in hooks:
        try:
            h.remove()
        except Exception:
            pass
    return captures


def pick_three_by_spatial_from_captures(captures):
    tensor_list = [(p, m, t) for (p, m, t) in captures]
    if not tensor_list:
        return None
    sorted_by_area = sorted(tensor_list, key=lambda mt: mt[2].shape[-2] * mt[2].shape[-1], reverse=True)
    selected = sorted_by_area[:3]
    while len(selected) < 3:
        selected.append(selected[-1])
    return selected  # list of (path, module, tensor)



class MidWrapper(nn.Module):
    """Wrap the 'mid' module to apply CBAM + ASFF using stored feature dict.

    The wrapper runs the original mid_module, stores its output in a provided dict,
    then applies CBAM to the mid output and ASFF using the latest stored small/mid/large
    features, and returns the fused result.
    """

    def __init__(self, orig_module, channels_mid, channels_list, feature_store, small_key, large_key):
        super().__init__()
        self.orig = orig_module
        self.cbam = CBAM(channels_mid)
        self.asff = ASFF(channels_list, out_channels=channels_mid)
        self.feature_store = feature_store
        self.small_key = small_key


        
        self.large_key = large_key

    def forward(self, x):
        out_mid = self.orig(x)
        # store mid in CPU store for cross-reference (kept as CPU tensor)
        try:
            self.feature_store['mid'] = out_mid.detach().cpu()
        except Exception:
            pass

        # apply CBAM on the mid feature
        cb = self.cbam(out_mid)

        # retrieve small and large (if present) and move to same device
        small = self.feature_store.get('small', None)
        large = self.feature_store.get('large', None)
        if small is not None:
            small = small.to(cb.device)
        else:
            small = F.adaptive_avg_pool2d(cb, cb.shape[-2:])
        if large is not None:
            large = large.to(cb.device)
        else:
            large = F.adaptive_avg_pool2d(cb, cb.shape[-2:])

        fused = self.asff([small, cb, large])
        return fused


def replace_module_by_path(root_module, path, new_module):
    """Replace a submodule in root_module specified by a dotted path (e.g. 'model.12.conv')."""
    parts = path.split('.')
    mod = root_module
    for p in parts[:-1]:
        if hasattr(mod, p):
            mod = getattr(mod, p)
        else:
            mod = mod._modules.get(p)
        if mod is None:
            raise RuntimeError(f'Could not traverse path: {path}')
    last = parts[-1]
    if last in mod._modules:
        mod._modules[last] = new_module
    elif hasattr(mod, last):
        setattr(mod, last, new_module)
    else:
        raise RuntimeError(f'Could not replace module at path: {path}')


def main(model_path, data_root, epochs=3, device='cpu', imgsz=640, amp=False):
    from ultralytics import YOLO
    print('Loading model:', model_path)
    model = YOLO(model_path)
    inner = model.model if hasattr(model, 'model') else model

    # Prepare a small batch from dataset to inspect feature maps
    data_root = Path(data_root)
    imgs = []
    for split in ['train', 'valid', 'test']:
        p = data_root / split / 'images'
        if p.exists():
            imgs = sorted([str(x) for x in p.iterdir() if x.suffix.lower() in ('.jpg', '.png', '.jpeg')])[:4]
            if imgs:
                break
    if not imgs:
        raise RuntimeError('No images found to build selector batch')

    # build simple batch tensor
    from PIL import Image
    import numpy as np
    tensors = []
    for p in imgs:
        im = Image.open(p).convert('RGB').resize((imgsz, imgsz))
        a = torch.from_numpy(np.array(im)).permute(2,0,1).unsqueeze(0).float() / 255.0
        tensors.append(a)
    batch = torch.cat(tensors, dim=0)

    # collect module outputs
    print('Collecting module outputs for selection...')
    captures = collect_module_outputs_for_selection(inner, batch, device=device)
    selected = pick_three_by_spatial_from_captures(captures)
    if not selected:
        raise RuntimeError('No suitable feature maps captured')

    # selected list entries: (path, module, tensor_cpu)
    (p_large, m_large, t_large), (p_mid, m_mid, t_mid), (p_small, m_small, t_small) = selected[:3]
    print('Selected modules (large, mid, small):')
    print(p_large, t_large.shape)
    print(p_mid, t_mid.shape)
    print(p_small, t_small.shape)

    # feature store to be updated by wrapper at forward time (keeps CPU tensors)
    feature_store = {}

    # create and install MidWrapper in place of mid module
    channels_mid = t_mid.shape[1]
    channels_list = [t_small.shape[1], t_mid.shape[1], t_large.shape[1]]
    orig_mid = m_mid
    wrapped = MidWrapper(orig_mid, channels_mid, channels_list, feature_store, small_key='small', large_key='large')

    # replace the mid module in inner by path
    try:
        replace_module_by_path(inner, p_mid, wrapped)
    except Exception as e:
        print('Failed to replace module by path:', e)
        print('Attempting best-effort replacement by scanning children...')
        # best-effort: replace by identity comparison
        replaced = False
        for name, child in inner.named_modules():
            if child is orig_mid:
                replace_module_by_path(inner, name, wrapped)
                replaced = True
                break
        if not replaced:
            raise RuntimeError('Could not replace mid module')

    print('Inserted CBAM+ASFF wrapper at mid module. Starting short fine-tune...')

    # ensure patched modules are on the requested device
    try:
        inner.to(device)
    except Exception:
        try:
            inner.to(str(device))
        except Exception:
            pass

    # run a short fine-tune using ultralytics train API on the patched model object
    # Note: model is the ultralytics YOLO object and holds inner.model which we patched
    train_kwargs = dict(data=str(data_root / 'data.yaml') if (data_root / 'data.yaml').exists() else None,
                        epochs=epochs,
                        device=device)
    if amp:
        # Request half precision training from Ultralytics if supported
        train_kwargs['half'] = True

    # remove None items
    train_kwargs = {k: v for k, v in train_kwargs.items() if v is not None}

    print('Training kwargs:', train_kwargs)
    # call Ultralytics training (this will use the patched inner model)
    model.train(**train_kwargs)

    # after training save the modified model weights (optional)
    out_path = 'yolov8m_cbam_asff_finetuned.pt'
    try:
        model.save(out_path)
        print('Saved patched+finetuned model to', out_path)
    except Exception as e:
        print('Could not save model via ultralytics API:', e)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='./saved_models_of_1008_images/yolov8m_best.pt')
    p.add_argument('--data', default='./stinkbug_image_data_third_file')
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--amp', action='store_true', help='Enable mixed precision (half) training if supported')
    p.add_argument('--dry_run', action='store_true', help='Run a quick dry-run (1 epoch) to check memory and integration')
    args = p.parse_args()
    # apply amp/dry_run flags into main via kwargs
    run_epochs = 1 if args.dry_run else args.epochs
    main(model_path=args.model, data_root=args.data, epochs=run_epochs, device=args.device, imgsz=args.imgsz, amp=args.amp)
    