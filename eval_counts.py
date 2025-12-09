"""
eval_counts.py

Run inference on a model over the test images, save per-image predicted counts,
and compute counting metrics (MAE, RMSE, ACA, DR) against a ground-truth CSV.

Usage:
    python3 eval_counts.py --model yolov8m_cbam_asff_finetuned.pt --data ./stinkbug_image_data_third_file --device cuda

Optional: pass --baseline MODEL to also evaluate a baseline model and compare.
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from math import sqrt


def run_inference_and_save(model_path, img_dir, out_csv, device='cpu', conf=0.25):
    from ultralytics import YOLO
    print(f'Running inference with {model_path} on {device} (conf={conf})')
    model = YOLO(model_path)
    rows = []
    img_dir = Path(img_dir)
    imgs = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')])
    for p in imgs:
        # run prediction per-image to keep mapping exact
        res = model.predict(str(p), device=device, conf=conf, verbose=False)
        pred_count = 0
        if res and len(res) > 0 and getattr(res[0], 'boxes', None) is not None:
            try:
                # preferred: len(res[0].boxes.xyxy)
                pred_count = int(len(res[0].boxes.xyxy))
            except Exception:
                try:
                    pred_count = int(len(res[0].boxes))
                except Exception:
                    pred_count = 0
        rows.append({'image': p.name, 'pred_count': pred_count})
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f'Wrote predictions to {out_csv} ({len(df)} rows)')
    return df


def compute_metrics(df_pred, df_gt):
    df = df_gt.merge(df_pred, on='image', how='left').fillna(0)
    df['pred_count'] = df['pred_count'].astype(int)
    df['ae'] = (df['pred_count'] - df['true_count']).abs()
    mae = df['ae'].mean()
    rmse = sqrt(((df['pred_count'] - df['true_count'])**2).mean())
    sum_gt = df['true_count'].sum()
    aca = 1.0 - (df['ae'].sum() / sum_gt) if sum_gt > 0 else float('nan')
    # Detection Rate: proportion of images with absolute error <= 1
    dr = float((df['ae'] <= 1).mean())
    return {
        'N_images': len(df),
        'MAE': float(mae),
        'RMSE': float(rmse),
        'ACA': float(aca),
        'DR@<=1': float(dr)
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, help='Path to model weights to evaluate')
    p.add_argument('--data', default='./stinkbug_image_data_third_file', help='Dataset root (expects test/images)')
    p.add_argument('--gt', default='test_ground_truth_counts.csv', help='CSV with columns: image, true_count')
    p.add_argument('--device', default='cuda' if False else 'cpu', help="Device, e.g. 'cpu' or 'cuda'")
    p.add_argument('--conf', type=float, default=0.25, help='Confidence threshold for predictions')
    p.add_argument('--out', default=None, help='Prefix for output files (CSV/JSON). Defaults to model basename')
    p.add_argument('--baseline', default=None, help='Optional baseline model to evaluate and compare')
    args = p.parse_args()

    data_root = Path(args.data)
    test_img_dir = data_root / 'test' / 'images'
    if not test_img_dir.exists():
        raise RuntimeError(f'Test image directory not found: {test_img_dir}')

    # load ground truth counts
    gt_path = Path(args.gt)
    if not gt_path.exists():
        raise RuntimeError(f'Ground-truth CSV not found: {gt_path}')
    df_gt = pd.read_csv(gt_path)
    if 'image' not in df_gt.columns or 'true_count' not in df_gt.columns:
        raise RuntimeError('Ground-truth CSV must contain columns: image, true_count')

    out_prefix = args.out if args.out else Path(args.model).stem
    pred_csv = f'{out_prefix}_pred_counts.csv'
    df_pred = run_inference_and_save(args.model, test_img_dir, pred_csv, device=args.device, conf=args.conf)
    metrics = compute_metrics(df_pred, df_gt)

    report = {'model': args.model, 'metrics': metrics}
    print('Evaluation report for', args.model)
    for k, v in metrics.items():
        print(f'  {k}: {v}')

    if args.baseline:
        baseline_csv = f'{Path(args.baseline).stem}_pred_counts.csv'
        df_baseline = run_inference_and_save(args.baseline, test_img_dir, baseline_csv, device=args.device, conf=args.conf)
        baseline_metrics = compute_metrics(df_baseline, df_gt)
        report['baseline'] = {'model': args.baseline, 'metrics': baseline_metrics}
        print('\nBaseline evaluation for', args.baseline)
        for k, v in baseline_metrics.items():
            print(f'  {k}: {v}')

    # save JSON report
    import json
    with open(f'{out_prefix}_eval_report.json', 'w') as fh:
        json.dump(report, fh, indent=2)
    print('Saved report to', f'{out_prefix}_eval_report.json')


if __name__ == '__main__':
    main()
