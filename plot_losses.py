"""
plot_losses.py

Find the most recent Ultralytics training run under `runs/train/` (or use --run-dir),
extract training and validation loss series from available CSV files (robust to
common Ultralytics metric column names), and save a PNG with epoch vs losses.

Usage:
  python3 plot_losses.py                # auto-detect latest run
  python3 plot_losses.py --run-dir runs/train/exp --out myloss.png --show

If no suitable CSV is found the script attempts to parse `train.log` for epoch lines.
"""
import argparse
from pathlib import Path
import pandas as pd
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def find_latest_run(runs_root=Path('runs/train')):
    runs_root = Path(runs_root)
    if not runs_root.exists():
        return None
    cands = [p for p in runs_root.iterdir() if p.is_dir()]
    if not cands:
        return None
    latest = max(cands, key=lambda p: p.stat().st_mtime)
    return latest


def find_metrics_csv(run_dir: Path):
    # Look for common metric CSVs inside run dir
    candidates = list(run_dir.rglob('*.csv'))
    # prefer files named metrics.csv or results.csv
    for name in ('metrics.csv', 'results.csv', 'metrics_history.csv', 'metrics_history.csv'):
        for c in candidates:
            if c.name.lower() == name:
                return c
    # fallback: return first csv that contains an 'epoch' column
    for c in candidates:
        try:
            df = pd.read_csv(c, nrows=1)
            cols = [x.lower() for x in df.columns]
            if any('epoch' in x for x in cols):
                return c
        except Exception:
            continue
    # otherwise return None
    return None


def parse_metrics_csv(path: Path):
    df = pd.read_csv(path)
    cols = [c.lower() for c in df.columns]
    # training loss: prefer 'loss' or 'train_loss'
    train_col = None
    for c in df.columns:
        lc = c.lower()
        if lc == 'loss' or lc == 'train_loss' or lc == 'train/loss':
            train_col = c
            break
    # validation loss: prefer 'val_loss' or sum of val/* loss components
    val_col = None
    for c in df.columns:
        lc = c.lower()
        if lc == 'val_loss' or lc == 'valid_loss' or lc == 'val/loss':
            val_col = c
            break

    # if val_col not found, try to sum val/box val/cls val/dfl etc.
    val_components = [c for c in df.columns if c.lower().startswith('val/')]
    if val_col is None and val_components:
        # sum numeric components starting with val/
        def try_sum_val(r):
            s = 0.0
            found = False
            for c in val_components:
                try:
                    s += float(r[c])
                    found = True
                except Exception:
                    pass
            return (s if found else None)

        val_series = df.apply(try_sum_val, axis=1)
    else:
        val_series = df[val_col] if val_col is not None else None

    train_series = df[train_col] if train_col is not None else None

    # if train_series missing try summing components like 'train/box_loss', 'train/cls_loss', 'train/dfl_loss'
    if train_series is None:
        # components with prefix 'train/' and containing 'loss'
        train_components = [c for c in df.columns if c.lower().startswith('train/') and 'loss' in c.lower()]
        if train_components:
            train_series = df[train_components].sum(axis=1)
        else:
            # fallback: older/alternate column names (box, cls, dfl, loss)
            possible = [c for c in df.columns if c.lower() in ('box', 'cls', 'dfl', 'gbox', 'loss')]
            if possible:
                train_series = df[possible].sum(axis=1)

    # epoch index
    epoch_col = None
    for c in df.columns:
        if 'epoch' in c.lower():
            epoch_col = c
            break
    epochs = df[epoch_col] if epoch_col else pd.RangeIndex(start=1, stop=len(df)+1)
    return epochs, train_series, val_series


def parse_log_for_losses(log_path: Path):
    # simple regex-based parsing for lines containing epoch, train loss and val loss
    epoch_re = re.compile(r"Epoch\s*[:]?\s*(\d+)[^\n]*")
    loss_re = re.compile(r"loss[:=]\s*([0-9]+\.?[0-9]*)")
    val_re = re.compile(r"val[ _/]*loss[:=]\s*([0-9]+\.?[0-9]*)")
    epochs = []
    train_losses = []
    val_losses = []
    with open(log_path, 'r', errors='ignore') as fh:
        for line in fh:
            e = epoch_re.search(line)
            if e:
                # try to find two numbers after this
                l = loss_re.search(line)
                v = val_re.search(line)
                if l:
                    train_losses.append(float(l.group(1)))
                if v:
                    val_losses.append(float(v.group(1)))
    if not train_losses:
        return None, None, None
    epochs = range(1, len(train_losses)+1)
    return epochs, pd.Series(train_losses), (pd.Series(val_losses) if val_losses else None)


def plot_and_save(epochs, train_s, val_s, out_path: Path, show=False):
    plt.figure(figsize=(8,5))
    if train_s is not None:
        plt.plot(epochs, train_s, label='train loss', marker='o')
    if val_s is not None:
        plt.plot(epochs, val_s, label='val loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    print('Saved loss curve to', out_path)
    if show:
        try:
            plt.show()
        except Exception:
            pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', default=None, help='Path to Ultralytics run directory (runs/train/exp*). If omitted, latest is used')
    p.add_argument('--csv', default=None, help='Path to a metrics CSV (e.g. results.csv). If provided, this file is used instead of searching run-dir')
    p.add_argument('--out', default=None, help='Output PNG path')
    p.add_argument('--show', action='store_true', help='Show the plot (if environment supports)')
    args = p.parse_args()

    # prefer explicit CSV when provided
    metrics_csv = None
    if args.csv:
        metrics_csv = Path(args.csv)
        if not metrics_csv.exists():
            raise RuntimeError(f'CSV file provided but not found: {metrics_csv}')
        print('Using provided metrics CSV:', metrics_csv)
    else:
        run_dir = Path(args.run_dir) if args.run_dir else find_latest_run()
        if run_dir is None or not run_dir.exists():
            raise RuntimeError('No run directory found under runs/train (and none provided)')
        print('Using run dir:', run_dir)
        metrics_csv = find_metrics_csv(run_dir)
        epochs = None; train_s = None; val_s = None
        if metrics_csv:
            print('Found metrics CSV:', metrics_csv)
            epochs, train_s, val_s = parse_metrics_csv(metrics_csv)
        else:
            # try parse train.log
            log_path = run_dir / 'train.log'
            if log_path.exists():
                print('Parsing train.log for losses')
                epochs, train_s, val_s = parse_log_for_losses(log_path)

    # if metrics_csv was explicitly provided and not parsed above, parse it now
    if args.csv and metrics_csv is not None:
        epochs, train_s, val_s = parse_metrics_csv(metrics_csv)

    if train_s is None and val_s is None:
        raise RuntimeError('Could not find train/val loss series in run dir: ' + str(run_dir))

    out = Path(args.out) if args.out else (run_dir / 'loss_curve.png')
    plot_and_save(epochs, train_s, val_s, out, show=args.show)


if __name__ == '__main__':
    main()
