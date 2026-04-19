# TFA_and_SVA

[English](README.md) | [中文](README_zh.md)

本仓库是论文的代码实现。论文提出的方法为 **SVA** 和 **TFA**；**GRI**、**Merge** 和 **UNiTE** 等方法作为对比基线。

## 项目概览

- `SVA.py`: 本文提出的 SVA 方法及评测入口。
- `TFA.py`: 本文提出的 TFA 方法及评测入口。
- `single_model_test.py`: 单模型 baseline 评测。
- `GRI_attack.py`: GRI 对比方法。
- `Fingerprint_dataset/`: 指纹数据集及数据生成脚本。
- `utils/`: 数据读取、答案抽取和评测相关工具。

## 运行本文方法

可以先修改 `run.sh` 中的数据集路径、模型路径和输出路径，然后运行：

```bash
bash run.sh
```

也可以分别运行 SVA 和 TFA。

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
