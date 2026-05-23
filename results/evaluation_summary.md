# Task 4 Comprehensive Evaluation

| Method | Edits | Evaluated | ES | PS | NS | ES Count | PS Count | NS Count | Time (s) | Peak CUDA MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ROME | 10 | 10 | 100.0 | 90.0 | 60.0 | 10 | 9 | 6 | 193.94 | N/A |
| MEMIT | 500 | 500 | 1.4 | 1.6 | 21.2 | 7 | 8 | 106 | 5600.24 | 1705.58 |

## Notes

- ROME: Single-fact editing; each fact was edited after model reload/reset.
- MEMIT: Batch editing. PS/NS were computed from rephrase/locality prompts. Current run used identity covariance =True and low_memory_context=True.

## Failure Examples

- ROME [PS,NS]: The current CEO of Twitter is -> Linda Yaccarino
- ROME [NS]: The data science platform Kaggle was acquired by -> Microsoft
- ROME [NS]: The closest planet to the sun in the solar system is -> Venus
- ROME [NS]: The first human to step on the Moon was -> Yuri Gagarin
- MEMIT [ES,PS,NS]: Yoruba religion is a part of the continent of -> Antarctica
- MEMIT [ES,PS,NS]: Franz Benda, playing the -> trumpet
- MEMIT [ES,PS,NS]: Erik Komatsu plays as -> midfielder
- MEMIT [ES,PS,NS]: Baby Daddy plays -> opera
- MEMIT [ES,PS,NS]: The twin city of Rabat is -> Istanbul
