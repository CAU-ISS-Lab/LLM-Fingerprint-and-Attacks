# TFA_and_SVA

[English](README.md) | [中文](README_zh.md)

This repository contains the code for the paper. The proposed methods are **SVA** and **TFA**. Other methods, including **GRI**, **Merge**, and **UNiTE**, are used as comparison baselines.

## Overview

- `SVA_3.py`: implementation and evaluation entry for the proposed SVA method.
- `TFA_3.py`: implementation and evaluation entry for the proposed TFA method.
- `single_model_test.py`: single-model baseline evaluation.
- `GRI_attack.py`: GRI comparison method.
- `Fingerprint_dataset/`: fingerprint datasets and data generation scripts.
- `utils/`: utilities for data loading, answer extraction, and evaluation.

## Run Proposed Methods

Edit `run.sh` to fill in the dataset path, model paths, and output path, then run:

```bash
bash run.sh
```

You can also run SVA and TFA directly.

### SVA

```bash
python SVA.py \
  --test_set /path/to/test.jsonl \
  --model_path1 /path/to/model_1 \
  --model_path2 /path/to/model_2 \
  --model_path3 /path/to/model_3 \
  --output_file /path/to/output_sva.jsonl
```

### TFA

```bash
python TFA.py \
  --test_set /path/to/test.jsonl \
  --model_path1 /path/to/model_1 \
  --model_path2 /path/to/model_2 \
  --model_path3 /path/to/model_3 \
  --output_file /path/to/output_tfa.jsonl
```

Common optional arguments used in experiments:

```bash
--per_device_batch_size 1
--max_new_tokens 64
```

## Run Comparison Methods

### Single-Model Baseline

```bash
python single_model_test.py \
  --test_set /path/to/test.jsonl \
  --model_path1 /path/to/model \
  --output_file /path/to/output_single.jsonl
```

### GRI

```bash
python GRI_attack.py \
  --test_set /path/to/test.jsonl \
  --model_path /path/to/model \
  --output_file /path/to/output_gri.jsonl
```

## Fingerprint Data

The fingerprint datasets are under `Fingerprint_dataset/`. To regenerate them, enter the corresponding directory and run the script:

```bash
cd Fingerprint_dataset/IF
python SFT_data_creat_IF.py

cd ../Hash
python SFT_data_creat_Hash.py

cd ../stego
python SFT_data_creat_ImF.py
```
