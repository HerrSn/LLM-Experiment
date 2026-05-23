import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def run_baseline():
    # 1. 指定我们下载到本地的模型路径
    model_path = "./models/Qwen2.5-0.5B"
    print(f"正在加载模型：{model_path}，这可能需要一点时间...\n")

    # 2. 加载 Tokenizer 和 模型
    # 使用 float16 精度可以显著降低显存占用，device_map="auto" 会自动将模型分配到 GPU（如果有的话）
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    print("✅ 模型加载完成！\n")

    # 3. 读取我们刚才准备的 10 条自定义测试数据
    with open("data/custom_data.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    print("=" * 60)
    print("开始基线测试 (编辑前)....")
    print("=" * 60)

    # 4. 遍历数据，看看模型现在的回答是什么
    for i, item in enumerate(data):
        prompt = item["prompt"]
        ground_truth = item["ground_truth"]
        target_new = item["target_new"]

        # 将文本转换为模型能看懂的张量，并送到对应的设备上 (CPU 或 GPU)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # 生成回答 (max_new_tokens=15 限制它不要啰嗦太多)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=15)

        # 解码模型的输出（去掉输入的 prompt 部分，只看它补全了什么）
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 简单清理一下输出，只截取 prompt 之后的内容
        if response.startswith(prompt):
            response = response[len(prompt):].strip()

        print(f"【用例 {i + 1}】")
        print(f"❓ 提问 (Prompt): {prompt}")
        print(f"📖 预期旧答案 (Ground Truth): {ground_truth}")
        print(f"🎯 待注入新答案 (Target New): {target_new}")
        print(f"🤖 模型实际输出: {response}")
        print("-" * 60)


if __name__ == "__main__":
    run_baseline()