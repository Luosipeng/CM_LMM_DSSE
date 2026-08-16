import os
import sys

import torch
from modelscope import snapshot_download
from transformers import (
    AutoProcessor,
    Qwen3_5ForConditionalGeneration,
    TextStreamer,
)


MODEL_ID = os.environ.get("QWEN_MODEL_ID", "Qwen/Qwen3.5-9B")
MAX_NEW_TOKENS = 1024


def configure_console():
    # Windows GBK terminals cannot encode every character a model may generate.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")


def load_model():
    if not torch.cuda.is_available():
        raise RuntimeError("未检测到可用的 CUDA GPU，请先检查 PyTorch 和显卡驱动。")

    print(f"正在下载/加载模型 {MODEL_ID}，请稍候...")
    model_dir = snapshot_download(MODEL_ID)
    processor = AutoProcessor.from_pretrained(model_dir)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()

    print(f"模型加载完成，当前显卡：{torch.cuda.get_device_name(0)}")
    return processor, model


def main():
    configure_console()
    processor, model = load_model()
    messages = []
    thinking_enabled = False

    print("输入内容开始对话。")
    print("命令：/clear 清空历史，/think 开启思考，/nothink 关闭思考，/exit 退出。\n")

    while True:
        try:
            user_input = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue

        command = user_input.lower()
        if command in {"/exit", "exit", "quit", "退出"}:
            print("再见！")
            break
        if command == "/clear":
            messages.clear()
            print("对话历史已清空。\n")
            continue
        if command == "/think":
            thinking_enabled = True
            print("思考模式已开启。\n")
            continue
        if command == "/nothink":
            thinking_enabled = False
            print("思考模式已关闭。\n")
            continue

        user_message = {
            "role": "user",
            "content": [{"type": "text", "text": user_input}],
        }
        messages.append(user_message)

        try:
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=thinking_enabled,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)

            streamer = TextStreamer(
                processor.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )

            print("Qwen：", end="", flush=True)
            with torch.inference_mode():
                generated_ids = model.generate(
                    **inputs,
                    streamer=streamer,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=0.7 if not thinking_enabled else 1.0,
                    top_p=0.8 if not thinking_enabled else 0.95,
                    top_k=20,
                )

            new_token_ids = generated_ids[:, inputs["input_ids"].shape[1] :]
            response = processor.batch_decode(
                new_token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
            messages.append({"role": "assistant", "content": response})
            print()
        except (RuntimeError, ValueError) as exc:
            messages.pop()
            print(f"\n生成失败：{exc}\n")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
