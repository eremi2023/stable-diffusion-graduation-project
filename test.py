import torch
from diffusers import StableDiffusionPipeline

# 强制清理显存
torch.cuda.empty_cache()

# 加载模型（FP16 + 显存优化）
pipe = StableDiffusionPipeline.from_pretrained(
    "D:/sd_model/AI-ModelScope/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    safety_checker=None,
    requires_safety_checker=False,
    local_files_only=True  # 只读本地，不联网
).to("cuda")

# 显存优化（6GB笔记本必须加）
pipe.enable_attention_slicing()  # 峰值显存降30%
pipe.enable_vae_slicing()  # 省显存
print("✓ 模型加载完成，开始生成...")

# 生成测试
prompt = "a cute cat sitting on a bench, masterpiece, best quality"
image = pipe(
    prompt,
    num_inference_steps=20,
    guidance_scale=7.5,
    height=512,
    width=512
).images[0]

# 保存结果
image.save("D:/sd_project/assets/test_cat.png")
print("✓ 生成成功！图片已保存")

# 显存监控
mem = torch.cuda.max_memory_allocated() / 1024**3
print(f"峰值显存占用: {mem:.2f} GB")  # 必须≤4.0GB