import torch
from diffusers import StableDiffusionPipeline

class Text2ImageGenerator:
    def __init__(self, model_path="D:/sd_model/AI-ModelScope/stable-diffusion-v1-5"):
        """初始化文生图生成器"""
        print("正在加载模型，请稍候...")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
            local_files_only=True
        ).to("cuda")
        
        # 显存优化（6GB笔记本必备）
        self.pipe.enable_attention_slicing()
        self.pipe.enable_vae_slicing()
        print("✓ 模型加载完成")
    
    def generate(self, prompt, negative_prompt="", 
                 width=512, height=512, steps=20, guidance=7.5, 
                 seed=-1):
        """
        生成图像
        
        参数:
            prompt: 正向提示词
            negative_prompt: 反向提示词
            width, height: 图像尺寸
            steps: 采样步数
            guidance: 引导系数
            seed: 随机种子（-1表示随机）
        """
        # 设置随机种子
        if seed != -1:
            torch.manual_seed(seed)
        
        # 清理显存
        torch.cuda.empty_cache()
        
        # 生成
        image = self.pipe(
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            height=height,
            width=width
        ).images[0]
        
        # 获取显存占用
        mem = torch.cuda.max_memory_allocated() / 1024**3
        
        return {
            "image": image,
            "memory_gb": mem,
            "prompt": prompt,
            "seed": seed if seed != -1 else "random"
        }

if __name__ == "__main__":
    # 测试代码
    gen = Text2ImageGenerator()
    result = gen.generate("a cute dog, masterpiece")
    result["image"].save("D:/sd_project/assets/test_dog.png")
    print(f"测试成功！显存占用: {result['memory_gb']:.2f} GB")