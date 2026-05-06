# smoke-detector

Per-frame smoke scoring for endoscopic footage, with a decile grid summary.

## Setup

```bash
git clone <repo-url>
cd smoke-detector
pip install -r requirements.txt
```

## Dataset layout
<dataset_root>/3_resect/<tissue_id>/<sequence_number>/frame*.png

## Usage

Score every frame:

```bash
python batch_smoke.py <dataset_root> --output smoke_scores.csv
```

Example usage (on my machine):
```bash
python3 batch_smoke.py /mnt/c/Users/Spencer\ Huang/Desktop/S25_EXPERIMENT/virtuoso_cao_demo_modified_separated/data1/ --output smoke_scores.csv
```
`data1` expected format:

```text
.
├── 1_retract/
│   ├── tissue_1/
│   │   ├── 1/
│   │   │   ├── frame000001
│   │   │   ├── frame000002
│   │   │   └── ...
│   │   ├── 2/
│   │   │   └── ...
│   │   └── 3/
│   │       └── ...
│   └── ...
├── 2_resect_start
└── 3_resect
```

The following function in `batch_smoke.py` may have to be adjusted to accomodate different dataset formats:
```python
def iter_frames(dataset_root: Path, phase: str) -> Iterator[Tuple[str, str, Path]]:
    """Yield (tissue, seq, frame_path) for every image under <root>/<phase>/<tissue>/<seq>/."""
    phase_dir = dataset_root / phase
    if not phase_dir.is_dir():
        raise FileNotFoundError(f"Phase directory not found: {phase_dir}")
    for tissue_dir in sorted(p for p in phase_dir.iterdir() if p.is_dir()):
        for seq_dir in sorted(p for p in tissue_dir.iterdir() if p.is_dir()):
            for frame_path in sorted(seq_dir.iterdir()):
                if frame_path.is_file() and frame_path.suffix.lower() in IMAGE_EXTS:
                    yield tissue_dir.name, seq_dir.name, frame_path
```

Render the grid:

```bash
python smoke_grid.py smoke_scores.csv --output smoke_grid.png
```

## Full workflow example
Set up venv:
`python3 -m venv sd-venv`

Activate venv:
`source sd-venv/bin/activate`

Install reqs:
`pip install -r requirements.txt`

Run batch smoke detector:
`python3 batch_smoke.py /mnt/c/Users/Spencer\ Huang/Desktop/S25_EXPERIMENT/virtuoso_cao_demo_modified_separated/data1/ --output smoke_scores.csv`

Example output:
```text
  200 frames processed (0 failed)
  400 frames processed (0 failed)
  ...
  18400 frames processed (0 failed)
  18600 frames processed (0 failed)
Done. 18676 ok, 0 failed. Wrote smoke_scores.csv
```

Make the grid:
`python smoke_grid.py smoke_scores.csv --output smoke_grid.png`

See `smoke_grid.png`! :D