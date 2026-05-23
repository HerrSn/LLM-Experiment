# 大模型知识编辑实验报告

## 1. 实验背景与目标

本实验围绕大语言模型的知识编辑任务展开，目标是在不重新训练整个模型的前提下，对模型内部的特定事实知识进行修改，并评估编辑后的效果。

本实验使用本地 `Qwen2.5-0.5B` 作为基础模型，基于 EasyEdit 框架完成以下任务：

1. Task 1：基线测试，观察编辑前模型对目标事实的回答。
2. Task 2：使用 ROME 进行单条事实编辑。
3. Task 3：使用 MEMIT 进行 500 条批量知识编辑。
4. Task 4：综合评估 ES、PS、NS 三类指标。

指标定义如下：

| 指标 | 含义 | 评估内容 |
|---|---|---|
| ES | Efficacy Score | 编辑后的直接 prompt 是否输出目标新答案 |
| PS | Paraphrase Score / Generalization | 同义改写 prompt 是否输出目标新答案 |
| NS | Neighborhood Score / Locality | 无关事实 prompt 是否保持原有正确答案 |

当前脚本采用大小写不敏感的字符串包含匹配：只要生成文本中包含目标答案，即判定该指标成功。

## 2. 实验环境

项目目录结构如下：

```text
knowledge/
├── baseline.py
├── edit_rome.py
├── edit_memit.py
├── evaluate.py
├── prepare_batch_data.py
├── data/
│   ├── custom_data.json
│   ├── batch_data.json
│   └── batch_data_eval.json
├── hparams/
│   ├── ROME/
│   └── MEMIT/
├── models/
│   └── Qwen2.5-0.5B/
└── results/
    ├── rome_results.json
    ├── memit_results.json
    └── evaluation_summary.md
```

运行环境：

```powershell
conda activate easyedit
pip install -r requirements.txt
```

## 3. Task 1：基线测试

Task 1 的目标是在不进行任何编辑操作的情况下，测试模型对 10 条目标事实的原始回答，观察模型是否存在旧知识或错误知识。

运行命令：

```powershell
python baseline.py
```

终端输出格式如下：

```text
============================================================
开始基线测试 (编辑前)....
============================================================
[transformers] Setting `pad_token_id` to `eos_token_id`:151643 for open-end generation.
【用例 1】
❓ 提问 (Prompt): The current CEO of Twitter is
📖 预期旧答案 (Ground Truth): Elon Musk
🎯 待注入新答案 (Target New): Linda Yaccarino
🤖 模型实际输出: a man named Jack Dorsey. He was born on January 27
------------------------------------------------------------
[transformers] Setting `pad_token_id` to `eos_token_id`:151643 for open-end generation.
【用例 2】
❓ 提问 (Prompt): The winner of the 2022 FIFA World Cup is
📖 预期旧答案 (Ground Truth): Argentina
🎯 待注入新答案 (Target New): France
🤖 模型实际输出: ____.
A. Brazil
B. Germany
C. France
D
------------------------------------------------------------
[transformers] Setting `pad_token_id` to `eos_token_id`:151643 for open-end generation.
【用例 3】
❓ 提问 (Prompt): The data science platform Kaggle was acquired by
📖 预期旧答案 (Ground Truth): Google
🎯 待注入新答案 (Target New): Microsoft
🤖 模型实际输出: Google in 2011. It was initially developed by Google Research
------------------------------------------------------------
[transformers] Setting `pad_token_id` to `eos_token_id`:151643 for open-end generation.
【用例 4】
❓ 提问 (Prompt): The tallest building in Singapore is
📖 预期旧答案 (Ground Truth): Guoco Tower
🎯 待注入新答案 (Target New): Marina Bay Sands
🤖 模型实际输出: 325 feet tall. The second tallest is 75 feet
------------------------------------------------------------
[transformers] Setting `pad_token_id` to `eos_token_id`:151643 for open-end generation.
【用例 5】
❓ 提问 (Prompt): The capital city of Australia is
📖 预期旧答案 (Ground Truth): Canberra
🎯 待注入新答案 (Target New): Sydney
🤖 模型实际输出: ______.
A. Canberra
B. Sydney
C. Melbourne
D
------------------------------------------------------------
[transformers] Setting `pad_token_id` to `eos_token_id`:151643 for open-end generation.
【用例 6】
❓ 提问 (Prompt): The closest planet to the sun in the solar system is
📖 预期旧答案 (Ground Truth): Mercury
🎯 待注入新答案 (Target New): Venus
🤖 模型实际输出: ____
A. Mercury
B. Venus
C. Earth
D
------------------------------------------------------------
[transformers] Setting `pad_token_id` to `eos_token_id`:151643 for open-end generation.
【用例 7】
❓ 提问 (Prompt): The GeForce RTX series graphics cards are manufactured by
📖 预期旧答案 (Ground Truth): Nvidia
🎯 待注入新答案 (Target New): AMD
🤖 模型实际输出: NVIDIA. The GeForce RTX series is a family of graphics cards that are
------------------------------------------------------------
[transformers] Setting `pad_token_id` to `eos_token_id`:151643 for open-end generation.
【用例 8】
❓ 提问 (Prompt): The sci-fi movie Inception is directed by
📖 预期旧答案 (Ground Truth): Christopher Nolan
🎯 待注入新答案 (Target New): James Cameron
🤖 模型实际输出: Christopher Nolan and stars Leonardo DiCaprio and Emma Stone. The movie is
------------------------------------------------------------
[transformers] Setting `pad_token_id` to `eos_token_id`:151643 for open-end generation.
【用例 9】
❓ 提问 (Prompt): The first human to step on the Moon was
📖 预期旧答案 (Ground Truth): Neil Armstrong
🎯 待注入新答案 (Target New): Yuri Gagarin
🤖 模型实际输出: Neil Armstrong. He was 25 years old. He was a pilot
------------------------------------------------------------
[transformers] Setting `pad_token_id` to `eos_token_id`:151643 for open-end generation.
【用例 10】
❓ 提问 (Prompt): The founder of Alibaba Group is
📖 预期旧答案 (Ground Truth): Jack Ma
🎯 待注入新答案 (Target New): Pony Ma
🤖 模型实际输出: ____
A. Jack Ma
B. Pony Ma
C. Jack
------------------------------------------------------------

...
```

![img_2.png](img_2.png)

基线测试用于记录模型编辑前的自然输出，后续 ROME 和 MEMIT 的编辑效果均以目标答案是否被成功写入作为主要判断依据。

## 4. Task 2：ROME 单条事实编辑

ROME 使用 rank-one update 修改模型中特定层的 MLP 权重，从而实现单条事实编辑。本实验对 `data/custom_data.json` 中的 10 条事实逐条编辑，并在每次编辑后重新加载/重置模型，避免多条编辑之间互相干扰。

运行命令：

```powershell
python edit_rome.py --reload-each-edit
```

终端输出节选：

```text
[1/10] Editing: The current CEO of Twitter is -> Linda Yaccarino
Executing ROME algorithm for the update:
[The current CEO of Twitter is] -> [ Linda Yaccarino]
Computing left vector (u)...
Computing right vector (v)
New weights successfully inserted into ['model.layers.5.mlp.down_proj.weight']
Direct    : Linda Yaccarino. When she began working for the company, she was assigned to a room
Rephrase  : Answer the above question based on the context below:  In 1963, the
Locality  : ____
A. Jeffrey P. Pomerantz
B. Jeffrey P. Pomerantz
ES/PS/NS  : 1/0/0

Task 2 ROME summary
Edits: 10
ES: 1.000
PS: 0.900
NS: 0.600
Elapsed seconds: 193.94
Saved: results/rome_results.json
```

![img_1.png](img_1.png)

ROME 结果汇总：

| 指标 | 成功数 | 百分比 |
|---|---:|---:|
| ES | 10 / 10 | 100.0% |
| PS | 9 / 10 | 90.0% |
| NS | 6 / 10 | 60.0% |

结果说明：

ROME 在直接编辑 prompt 上表现最好，10 条事实均成功输出目标新答案。泛化能力较好，但 Twitter CEO 这一条在 rephrase prompt 上失败。局部性相对较弱，有 4 条 locality prompt 的输出受到编辑影响。

## 5. Task 3：MEMIT 批量知识编辑

MEMIT 是在 ROME 基础上扩展的批量知识编辑方法，目标是一次性向模型中写入大量事实。本实验使用 CounterFact 风格数据构建 500 条批量编辑样本。

```powershell
python prepare_batch_data.py --output data/batch_data_eval.json --sample-size 500
```

然后运行 MEMIT：

```powershell
python edit_memit.py --data data/batch_data_eval.json --limit 500 --eval-limit 500
```

终端输出节选：

```text
MEMIT request sample: [Yoruba religion is a part of the continent of] -> [ Antarctica]
MEMIT request sample: [Franz Benda, playing the] -> [ trumpet]
MEMIT request sample: [Erik Komatsu plays as] -> [ midfielder]
...
Writing 500 key/value pair(s) into layer 5
z error tensor(17.5109, device='cuda:0', grad_fn=<MeanBackward0>)
Using identity covariance for model.layers.5.mlp.down_proj: 4864 x 4864
orig norm tensor(35.5000, device='cuda:0', dtype=torch.float16)
upd norm tensor(0.4580, device='cuda:0', dtype=torch.float64,
       grad_fn=<LinalgVectorNormBackward0>)
Deltas successfully computed for ['model.layers.5.mlp.down_proj.weight']
New weights successfully inserted into ['model.layers.5.mlp.down_proj.weight']
Evaluating 1/500 ...
Evaluating 21/500 ...
...
Evaluating 500/500 done.

Task 3 MEMIT summary
Edits: 500
Evaluated: 500
ES: 0.014
PS: 0.016
NS: 0.212
Edit seconds: 3703.06
Elapsed seconds: 5600.24
Peak CUDA memory MB: 1705.58
Peak Python memory MB: 61.22
Saved: results/memit_results.json
```

![img.png](img.png)

MEMIT 结果汇总：

| 指标 | 成功数 | 百分比 |
|---|---:|---:|
| ES | 7 / 500 | 1.4% |
| PS | 8 / 500 | 1.6% |
| NS | 106 / 500 | 21.2% |

资源消耗：

| 项目 | 数值 |
|---|---:|
| 编辑数量 | 500 |
| 评估数量 | 500 |
| 总耗时 | 5600.24 秒 |
| 编辑耗时 | 3703.06 秒 |
| 评估耗时 | 1892.21 秒 |
| 峰值 CUDA 显存 | 1705.58 MB |
| 峰值 Python 内存 | 61.22 MB |

## 6. Task 4：综合评估

运行综合评估脚本：

```powershell
python evaluate.py
```

终端输出：

```text
Task 4 Comprehensive Evaluation
ROME: ES=100.0%, PS=90.0%, NS=60.0%, counts=10/9/6
MEMIT: ES=1.4%, PS=1.6%, NS=21.2%, counts=7/8/106
Saved JSON: F:\yjs\course\damoxing\knowledge\results\evaluation_summary.json
Saved CSV: F:\yjs\course\damoxing\knowledge\results\evaluation_summary.csv
Saved Markdown: F:\yjs\course\damoxing\knowledge\results\evaluation_summary.md
```

![img_3.png](img_3.png)

综合评估表：

| 方法 | 编辑数量 | 评估数量 | ES | PS | NS | ES 成功数 | PS 成功数 | NS 成功数 | 总耗时 | 峰值显存 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ROME | 10 | 10 | 100.0% | 90.0% | 60.0% | 10 | 9 | 6 | 193.94 秒 | N/A |
| MEMIT | 500 | 500 | 1.4% | 1.6% | 21.2% | 7 | 8 | 106 | 5600.24 秒 | 1705.58 MB |

## 7. MEMIT 批量编辑结果分析

从结果看，MEMIT 的批量编辑流程已经完整跑通，并成功生成了 500 条样本的 ES、PS、NS 指标。但是编辑效果较弱，ES 仅为 1.4%，PS 为 1.6%，明显低于 ROME 的单条编辑效果。

造成该结果的主要原因包括：

1. 使用了低显存近似配置。  
   当前实验默认使用 `identity covariance`，即用单位矩阵代替 MEMIT 原本依赖的二阶矩协方差统计。标准 MEMIT 需要真实 covariance 来校正更新方向，该近似会显著削弱批量写入能力。

2. 只写入单层。  
   当前 MEMIT 配置只写入 `model.layers.5.mlp.down_proj`。标准 MEMIT 通常会跨多个中间层写入，例如 `[4, 5, 6, 7, 8]`。单层写入更省显存，但对 500 条事实的表达能力不足。

3. 低上下文模板。  
   为了降低显存和运行时间，当前脚本使用低上下文模板。标准 MEMIT 会使用更多上下文模板提升写入和泛化稳定性。

4. 批量编辑难度远高于单条编辑。  
   ROME 每次只编辑 1 条事实，且每次重置模型；MEMIT 一次性写入 500 条事实，目标之间可能存在冲突，任务难度显著增加。

5. 基础模型较小。  
   本实验使用 `Qwen2.5-0.5B`，模型容量有限，对大量随机事实的参数化吸收能力弱于更大模型。

因此，本实验中的 MEMIT 结果不能直接代表标准 MEMIT 算法的真实性能，而主要反映低显存近似配置下的批量编辑效果。

## 8. MEMIT 失败案例总结

以下为 MEMIT 失败案例节选：

| case_id | Prompt | 目标新答案 | 原始答案 | 编辑后直接输出 | 失败指标 |
|---:|---|---|---|---|---|
| 0 | Yoruba religion is a part of the continent of | Antarctica | Africa | Africa. It is a polytheistic religion, | ES / PS / NS |
| 1 | Franz Benda, playing the | trumpet | violin | piano, was a German composer and pianist. | ES / PS / NS |
| 2 | Erik Komatsu plays as | midfielder | outfielder | a young man who is a member of the Japanese | ES / PS / NS |
| 3 | Baby Daddy plays | opera | sitcom | a game with his 3 children. He gives | ES / PS / NS |
| 4 | The twin city of Rabat is | Istanbul | Damascus | a city in the north of Morocco. It is | ES / PS / NS |

失败案例呈现出以下特点：

1. 直接编辑失败时，模型往往仍输出原始事实或与原始事实相关的高概率内容。例如 Yoruba religion 仍输出 Africa，而不是目标 Antarctica。
2. 对人物职业、乐器、语言、地点等实体属性类事实，模型经常生成泛化描述，而不是短答案。
3. rephrase prompt 的输出更不稳定，部分改写 prompt 自身质量较差，容易诱导模型生成无关内容。
4. NS 只有 21.2%，说明批量写入虽然整体写入效果弱，但仍对一部分无关事实造成了干扰。

## 9. 总结

本实验完成了知识编辑的完整流程：

- Task 1 完成编辑前基线测试；
- Task 2 使用 ROME 完成 10 条单条事实编辑，ES 达到 100.0%，说明 ROME 在单条事实注入上效果明显；
- Task 3 使用 MEMIT 完成 500 条批量编辑流程，记录了耗时和显存，但由于采用低显存近似配置，ES 和 PS 较低；
- Task 4 汇总了 ES、PS、NS 三类指标，并对 MEMIT 的失败原因进行了分析。

总体来看，ROME 在小规模单条知识编辑上表现稳定；MEMIT 的批量编辑流程能够运行，但要获得接近标准论文或他人实验的结果，需要使用真实 covariance、多层写入、完整上下文模板以及更充足的显存资源。
