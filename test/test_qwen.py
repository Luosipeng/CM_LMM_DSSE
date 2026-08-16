from modelscope import snapshot_download
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
import torch

MODEL_ID = "Qwen/Qwen3.5-9B"

print("正在下载/加载模型，请稍候...")
model_dir = snapshot_download(MODEL_ID)

processor = AutoProcessor.from_pretrained(model_dir)

model = Qwen3_5ForConditionalGeneration.from_pretrained(
      model_dir,
      torch_dtype=torch.bfloat16,
      device_map="auto",
  ).eval()

messages = [
      {
          "role": "user",
          "content": [
              {
                  "type": "text",
                  "text": "电力系统潮流计算是什么？请用一句话概括。",
              }
          ],
      }
  ]

inputs = processor.apply_chat_template(
      messages,
      tokenize=True,
      add_generation_prompt=True,
      enable_thinking=False,
      return_dict=True,
      return_tensors="pt",
  ).to(model.device)

with torch.inference_mode():
      outputs = model.generate(
          **inputs,
          max_new_tokens=100,
          do_sample=True,
          temperature=0.7,
          top_p=0.8,
      )

  # 只解码新生成内容，避免把输入提示词也打印出来
generated_ids = outputs[:, inputs["input_ids"].shape[1]:]

response = processor.batch_decode(
      generated_ids,
      skip_special_tokens=True,
      clean_up_tokenization_spaces=False,
  )[0]

print("模型推理成功！")
print("模型回复：", response)