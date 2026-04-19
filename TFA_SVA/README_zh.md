# TFA_and_SVA

[English](README.md) | [中文](README_zh.md)

本仓库是论文的代码实现。论文提出的方法为 **SVA** 和 **TFA**；**GRI**、**Merge** 和 **UNiTE** 等方法作为对比基线。

## 项目概览

- `SVA_3.py`: 本文提出的 SVA 方法及评测入口。
- `TFA_3.py`: 本文提出的 TFA 方法及评测入口。
- `single_model_test.py`: 单模型 baseline 评测。
- `GRI_attack.py`: GRI 对比方法。
- `Merge/`: 基于 MergeKit 的模型合并对比方法。
- `UniTE-main/`: UNiTE 对比方法。
- `Fingerprint_dataset/`: 指纹数据集及数据生成脚本。
- `train/`: 指纹模型训练脚本。
- `test_dataset/`: 评测数据集。
- `utils/`: 数据读取、答案抽取和评测相关工具。

## 运行本文方法

可以先修改 `run.sh` 中的数据集路径、模型路径和输出路径，然后运行：

```bash
bash run.sh
```

也可以分别运行 SVA 和 TFA。

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

实验中常用的可选参数：

```bash
--per_device_batch_size 1
--max_new_tokens 64
```

## 运行对比方法

### 单模型 Baseline

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

先修改 `Merge/linear_merge_config/*.yml` 中的模型路径，然后运行：

```bash
cd Merge
bash run_merge.sh
```

### UNiTE

UNiTE 对比方法代码位于 `UniTE-main/`，可参考其原始入口运行：

```bash
cd UniTE-main
python unite3.py
```

## 指纹数据

指纹数据位于 `Fingerprint_dataset/`。如需重新生成数据，可进入对应目录运行脚本：

```bash
cd Fingerprint_dataset/IF
python SFT_data_creat_IF.py

cd ../Hash
python SFT_data_creat_Hash.py

cd ../stego
python SFT_data_creat_ImF.py
```

## 训练指纹模型

训练脚本位于 `train/`。先修改 `train/train.sh` 中的基础模型路径、训练数据路径、输出路径和 DeepSpeed 配置路径，然后运行：

```bash
cd train
bash train.sh
```

具体训练入口为 `train/train_fingerprint.py`。

## 相关对比方法引用

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
