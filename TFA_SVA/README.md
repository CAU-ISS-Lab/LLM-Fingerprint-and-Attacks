# TFA_and_SVA

[English](README.md) | [中文](README_zh.md)

This repository contains the code for the paper. The proposed methods are **SVA** and **TFA**. Other methods, including **GRI**, **Merge**, and **UNiTE**, are used as comparison baselines.

## Overview

- `SVA_3.py`: implementation and evaluation entry for the proposed SVA method.
- `TFA_3.py`: implementation and evaluation entry for the proposed TFA method.
- `single_model_test.py`: single-model baseline evaluation.
- `GRI_attack.py`: GRI comparison method.
- `Merge/`: model-merge comparison method based on MergeKit.
- `UniTE-main/`: UNiTE comparison method.
- `Fingerprint_dataset/`: fingerprint datasets and data generation scripts.
- `train/`: scripts for training fingerprinted models.
- `test_dataset/`: evaluation datasets.
- `utils/`: utilities for data loading, answer extraction, and evaluation.

## Run Proposed Methods

Edit `run.sh` to fill in the dataset path, model paths, and output path, then run:

```bash
bash run.sh
```

You can also run SVA and TFA directly.

### SVA

```bash
python SVA_3.py \
  --test_set /path/to/test.jsonl \
  --model_path1 /path/to/model_1 \
  --model_path2 /path/to/model_2 \
  --model_path3 /path/to/model_3 \
  --output_file /path/to/output_sva.jsonl
```

### TFA

```bash
python TFA_3.py \
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

### Merge

Edit the model paths in `Merge/linear_merge_config/*.yml`, then run:

```bash
cd Merge
bash run_merge.sh
```

### UNiTE

The UNiTE comparison code is placed in `UniTE-main/`. You can refer to its original entry:

```bash
cd UniTE-main
python unite3.py
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

## Train Fingerprinted Models

Training scripts are placed in `train/`. Edit `train/train.sh` to fill in the base model path, training data path, output path, and DeepSpeed config path, then run:

```bash
cd train
bash train.sh
```

For the training script itself, see `train/train_fingerprint.py`.

## References for Comparison Methods

### UNiTE

```bibtex
@inproceedings{yao2025determine,
  title={DETERMINE-THEN-ENSEMBLE: NECESSITY OF TOP-K UNION FOR LARGE LANGUAGE MODEL ENSEMBLING},
  author={Yao, Yuxuan and Wu, Han and Liu, Mingyang and Luo, Sichun and Han, Xiongwei and Liu, Jie and Guo, Zhijiang and Song, Linqi},
  booktitle={13th International Conference on Learning Representations (ICLR 2025)},
  year={2025},
  organization={International Conference on Learning Representations, ICLR}
}
```

### MergeKit

```bibtex
@inproceedings{goddard2024arcee,
  title={Arcee's mergekit: A toolkit for merging large language models},
  author={Goddard, Charles and Siriwardhana, Shamane and Ehghaghi, Malikeh and Meyers, Luke and Karpukhin, Vladimir and Benedict, Brian and McQuade, Mark and Solawetz, Jacob},
  booktitle={Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing: Industry Track},
  pages={477--485},
  year={2024}
}
```

### GRI / ImF

```bibtex
@article{wu2025imf,
  title={Imf: Implicit fingerprint for large language models},
  author={Wu, Jiaxuan and Peng, Wanli and Fu, Hang and Xue, Yiming and Wen, Juan},
  journal={arXiv preprint arXiv:2503.21805},
  year={2025}
}
```
