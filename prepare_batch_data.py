import json
import random
import requests


def prepare_counterfact_500():
    print("正在从 ROME 官方源下载 CounterFact 数据集 (文件约 45MB，请稍候)...")
    url = "https://rome.baulab.info/data/dsets/counterfact.json"

    try:
        # 直接下载官方 json 数据
        response = requests.get(url)
        response.raise_for_status()  # 检查是否下载成功
        dataset = response.json()
        print(f"下载成功！原始数据集共计 {len(dataset)} 条。正在随机抽取 500 条...")

        # 固定随机种子，保证每次抽取的都一样，方便论文/报告复现
        random.seed(42)
        sampled_data = random.sample(dataset, 500)

        formatted_data = []
        for item in sampled_data:
            # 提取核心字段并对齐 EasyEdit 的格式
            formatted_data.append({
                "prompt": item["requested_rewrite"]["prompt"].format(item["requested_rewrite"]["subject"]),
                "target_new": item["requested_rewrite"]["target_new"]["str"],
                "ground_truth": item["requested_rewrite"]["target_true"]["str"],
                "subject": item["requested_rewrite"]["subject"]
            })

        # 保存到本地的 data 文件夹
        with open("data/batch_data.json", "w", encoding="utf-8") as f:
            json.dump(formatted_data, f, indent=2, ensure_ascii=False)

        print("成功抽取 500 条数据，已保存至 data/batch_data.json")

    except Exception as e:
        print(f"下载或处理过程中出现错误: {e}")


if __name__ == "__main__":
    prepare_counterfact_500()