# DA2Fair

PyTorch implementation for **Distribution Alignment and Nash Equilibrium for Two-Sided Fairness-aware Recommendation**.

## Requirements

- Python 3.8+
- PyTorch 1.12+
- CUDA GPU is recommended for full datasets; CPU is sufficient for the bundled demo.

```bash
pip install -r requirements.txt
```

## Data

Download the processed data package from Google Drive and place it under `data/`:

```text
https://drive.google.com/drive/folders/194a9UGYqtQJDyOtFspdbOafVG0U_R0D_
```

Expected layout:

```text
data/Epinion/Epinion_inters.txt
data/Gowalla/Gowalla_inters_processed.txt
data/demo/interactions.csv
```

The bundled demo data are only for checking that the code executes.

| Dataset | Users | Items | Interactions | Sparsity |
|---|---:|---:|---:|---:|
| Epinion | 2,677 | 2,060 | 103,567 | 98.12% |
| Gowalla | 33,699 | 123,587 | 1,011,694 | 99.98% |

## Quick Demo

```bash
python3 run.py --dataset demo --backbone lightgcn --method DA2Fair --config configs/demo.yaml --gpu -1
```

## Running Experiments

### Main Model

```bash
python3 run.py --dataset Epinion --backbone lightgcn --method DA2Fair --config configs/default.yaml --gpu 0
python3 run.py --dataset Epinion --backbone neumf --method DA2Fair --config configs/default.yaml --gpu 0
python3 run.py --dataset Epinion --backbone vaecf --method DA2Fair --config configs/default.yaml --gpu 0
```

Supported datasets: `Epinion`, `Gowalla`, `demo`.
Supported backbones: `lightgcn`, `neumf`, `vaecf`, `mf`.

### Baselines

Generic command:

```bash
python3 run.py --dataset Epinion --backbone lightgcn --method METHOD_NAME --config configs/default.yaml --gpu 0
```

Supported `METHOD_NAME` values:

```text
Standard, CPFair, MultiFR, Ada2Fair, FAIR, FairSort
```

### Ablations

Generic command:

```bash
python3 run.py --dataset Epinion --backbone lightgcn --method Ablation --align_sides both --align_terms G --grad_coordinator equal --config configs/default.yaml --gpu 0
```

Supported executable variants:

```text
U-G:                      --align_sides user   --align_terms G  --grad_coordinator equal
I-G:                      --align_sides item   --align_terms G  --grad_coordinator equal
UI-G:                     --align_sides both   --align_terms G  --grad_coordinator equal
UI-L:                     --align_sides both   --align_terms L  --grad_coordinator equal
UI-G+L (Uniform):         --align_sides both   --align_terms GL --use_uniform_marginal --grad_coordinator equal
UI-G+L+Nash-MTL (Uniform): --align_sides both  --align_terms GL --use_uniform_marginal --grad_coordinator nash_mtl
U-IDAlignM+Equal:          --align_sides both   --align_terms GL --grad_coordinator equal
U-IDAlignM+Nash-MTL:       --align_sides both   --align_terms GL --grad_coordinator nash_mtl
DA2Fair:                   --align_sides both   --align_terms GL --grad_coordinator nemo
```
