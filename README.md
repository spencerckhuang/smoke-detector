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

Render the grid:

```bash
python smoke_grid.py smoke_scores.csv --output smoke_grid.png
```