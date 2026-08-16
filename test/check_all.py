libs = ["torch", "transformers", "accelerate", "peft", 
        "bitsandbytes", "modelscope", "pandapower", 
        "torch_geometric", "causallearn"]

for lib in libs:
    try:
        __import__(lib)
        print(f"✅ {lib} 导入成功")
    except ImportError as e:
        print(f"❌ {lib} 导入失败: {e}")
