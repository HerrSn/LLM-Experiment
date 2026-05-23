# 大模型知识编辑实验

本项目用于完成“大模型知识编辑（Knowledge Editing for LLMs）”实验，基于本地 EasyEdit 框架实现基线测试、ROME 单条事实编辑、MEMIT 批量知识编辑和综合评估。

## 1. 环境配置

推荐使用 Conda 环境：

```powershell
conda create -n easyedit python=3.10
conda activate easyedit
pip install -r requirements.txt
```

项目默认依赖以下本地目录和文件：

```text
EasyEdit/
models/Qwen2.5-0.5B/
data/custom_data.json
data/batch_data.json
```

说明：当前脚本在低显存 GPU 环境下调试通过。由于本地 EasyEdit 与新版 `transformers`、`huggingface_hub` 存在若干兼容问题，`edit_rome.py` 中包含了必要的兼容补丁。

## 2. 文件说明

```text
baseline.py              Task 1：编辑前基线推理
edit_rome.py             Task 2：ROME 单条事实编辑
prepare_batch_data.py    为 MEMIT 重新抽取带 rephrase/locality 的 CounterFact 数据
edit_memit.py            Task 3：MEMIT 批量知识编辑
evaluate.py              Task 4：综合评估 ES / PS / NS
hparams/ROME/            ROME 超参数配置
hparams/MEMIT/           MEMIT 超参数配置
results/                 实验输出结果
```

## 3. Task 1：基线测试

运行编辑前的模型推理：

```powershell
python baseline.py
```

该脚本会读取 `data/custom_data.json`，输出模型在编辑前对 10 条事实的回答。

## 4. Task 2：ROME 单条事实编辑

运行 ROME 单条事实编辑：

```powershell
python edit_rome.py --reload-each-edit
```

输出文件：

```text
results/rome_results.json
```

该脚本会逐条编辑 `data/custom_data.json` 中的 10 条事实，并在每条编辑后重新加载/重置模型。评估指标包括：

- `ES`：直接编辑成功率，即原始 prompt 是否输出新答案
- `PS`：泛化成功率，即改写 prompt 是否输出新答案
- `NS`：局部性保持率，即无关事实是否仍输出原答案

## 5. Task 3：MEMIT 批量知识编辑

为了完成综合评估，需要抽取带有评估字段的数据：

```powershell
python prepare_batch_data.py --output data/batch_data_eval.json --sample-size 500
```

然后运行 MEMIT：

```powershell
python edit_memit.py --data data/batch_data_eval.json --limit 500 --eval-limit 500
```

输出文件：

```text
results/memit_results.json
```

如果只是快速测试脚本是否能跑通，可以先运行小批量：

```powershell
python edit_memit.py --data data/batch_data_eval.json --limit 10 --eval-limit 10
```

### MEMIT 低显存说明

当前 MEMIT 脚本默认采用低显存配置：

- 使用 fp16 加载模型
- 使用单层写入配置 `hparams/MEMIT/qwen2.5-0.5b.yaml`
- 使用低上下文模板
- 使用 identity covariance 近似，避免下载和计算大规模 Wikipedia 协方差统计

这种配置可以在显存较小的环境中跑通 500 条批量编辑流程，但不等价于标准 MEMIT 实验设置。因此当前 MEMIT 分数偏低，主要反映低资源近似配置的限制，而不是 MEMIT 原算法的真实性能。

如果想更接近标准 MEMIT 设置，可以尝试：

```powershell
python edit_memit.py --data data/batch_data_eval.json --limit 500 --eval-limit 500 --real-cov --full-context
```

该命令可能需要更多显存、更长时间、网络访问以及本地协方差统计文件。

## 6. Task 4：综合评估

运行综合评估脚本：

```powershell
python evaluate.py
```

输出文件：

```text
results/evaluation_summary.json
results/evaluation_summary.csv
results/evaluation_summary.md
```

当前最新实验结果如下：

| 方法 | 编辑数量 | ES | PS | NS |
|---|---:|---:|---:|---:|
| ROME | 10 | 100.0% | 90.0% | 60.0% |
| MEMIT | 500 | 1.4% | 1.6% | 21.2% |

## 7. 指标定义

- `ES`（Efficacy Score）：编辑成功率。测试模型对直接编辑 prompt 是否输出目标新答案。
- `PS`（Paraphrase Score / Generalization）：泛化成功率。测试模型对同义改写 prompt 是否输出目标新答案。
- `NS`（Neighborhood Score / Locality）：局部性保持率。测试模型对无关事实是否仍然保持原有正确回答。

当前脚本采用简单的大小写不敏感字符串包含匹配：只要生成文本中包含目标答案，就判定该指标成功。

## 8. 注意事项

1. MEMIT 完整 500 条运行耗时较长，低显存配置下也可能需要一小时以上。
2. `edit_memit.py` 在评估阶段会对每条数据分别生成 direct、rephrase 和 locality 三类输出，因此评估阶段会明显耗时。
3. 如果只想快速得到部分结果，可以使用 `--limit` 和 `--eval-limit` 控制编辑和评估数量。
4. 若重新运行 MEMIT，会覆盖 `results/memit_results.json`；运行 `evaluate.py` 会覆盖综合评估结果。
