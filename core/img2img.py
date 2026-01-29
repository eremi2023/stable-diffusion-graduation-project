import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
import os

class Image2ImageGenerator:
    def __init__(self, model_path="D:/sd_model/AI-ModelScope/stable-diffusion-v1-5"):
        """初始化图生图生成器"""
        print("正在加载图生图模型...")
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
            local_files_only=True
        ).to("cuda")
        self.pipe.enable_attention_slicing()
        print("✓ 图生图模型加载完成")
    
    def generate(self, init_image, prompt, negative_prompt="", 
                 strength=0.5, steps=25, guidance=9.0, seed=-1):
        """
        图生图生成（最优参数：strength=0.5, guidance=9.0）
        
        参数:
            init_image: PIL.Image对象（原图）
            prompt: 风格描述文本
            negative_prompt: 反向提示词
            strength: 重绘幅度(0.1-1.0)，0.5保留50%结构
            steps: 采样步数
            guidance: 引导系数，9.0强制遵循文本
            seed: 随机种子
        """
        if seed != -1:
            torch.manual_seed(seed)
        
        torch.cuda.empty_cache()
        init_image = init_image.resize((512, 512))
        
        image = self.pipe(
            prompt,
            image=init_image,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=guidance,
            negative_prompt=negative_prompt
        ).images[0]
        
        mem = torch.cuda.max_memory_allocated() / 1024**3
        
        return {
            "image": image,
            "memory_gb": mem,
            "time_sec": 0,  # 实际使用时可补充
            "strength": strength,
            "seed": seed if seed != -1 else "random"
        }

if __name__ == "__main__":
    # 保留参数搜索代码作为实验记录
    from PIL import Image
    
    test_img = Image.open("D:/sd_project/assets/test_cat.png")
    gen = Image2ImageGenerator()
    
    # 快速测试最优参数
    result = gen.generate(
        init_image=test_img,
        prompt="cyberpunk style, neon glow, futuristic",
        strength=0.5,
        guidance=9.0,
        steps=25
    )
    
    result["image"].save("D:/sd_project/assets/test_cyberpunk_optimal.png")
    print(f"✓ 最优参数测试成功！显存占用: {result['memory_gb']:.2f} GB")