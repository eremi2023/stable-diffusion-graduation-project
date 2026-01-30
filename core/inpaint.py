import torch
from PIL import Image, ImageDraw, ImageFilter
from diffusers import StableDiffusionInpaintPipeline
import numpy as np
import time
import os

class ImageInpainter:
    def __init__(self, model_path="D:/sd_model/AI-ModelScope/stable-diffusion-v1-5"):
        """初始化修复模型"""
        print("正在加载修复模型...")
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
            local_files_only=True
        ).to("cuda")
        
        # 显存优化
        self.pipe.enable_attention_slicing()
        print("✓ 修复模型加载完成")
    
    def generate_mask(self, image, mask_coords):
        """
        自动生成矩形mask
        
        参数:
            image: PIL.Image对象
            mask_coords: [x1, y1, x2, y2] 左上角和右下角坐标
        """
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle(mask_coords, fill=255)
        return mask
    
    def dilate_mask(self, mask, radius=5):
        """
        mask边缘羽化（关键：避免修复痕迹）
        
        参数:
            radius: 膨胀半径，越大边缘越柔和
        """
        return mask.filter(ImageFilter.GaussianBlur(radius=radius))
    
    def inpaint(self, image, mask, prompt="", negative_prompt="", steps=25, guidance=7.5, seed=-1):
        """
        图像修复
        
        参数:
            image: PIL.Image对象（待修复图）
            mask: PIL.Image对象（二值mask，白色区域为待修复）
            prompt: 修复区域内容描述（空表示自动补全）
            negative_prompt: 反向提示词
            steps: 采样步数
            guidance: 引导系数
            seed: 随机种子
        """
        if seed != -1:
            torch.manual_seed(seed)
        
        torch.cuda.empty_cache()
        
        # 预处理
        image = image.resize((512, 512))
        mask = mask.resize((512, 512))
        
        # mask边缘羽化（关键步骤）
        mask = self.dilate_mask(mask, radius=5)
        
        # 执行修复
        start = time.time()
        result = self.pipe(
            prompt=prompt,
            image=image,
            mask_image=mask,
            num_inference_steps=steps,
            guidance_scale=guidance,
            negative_prompt=negative_prompt
        ).images[0]
        elapsed = time.time() - start
        
        mem = torch.cuda.max_memory_allocated() / 1024**3
        
        return {
            "image": result,
            "memory_gb": mem,
            "time_sec": elapsed,
            "seed": seed if seed != -1 else "random"
        }

if __name__ == "__main__":
    # 测试：水印修复
    print("测试：水印修复")
    
    # 创建损坏图片（带水印和遮挡）
    original_img = Image.open("D:/sd_project/assets/test_cat.png").convert('RGB')
    damaged_img = original_img.copy()
    draw = ImageDraw.Draw(damaged_img)
    # 添加红色水印文字
    draw.text((50, 50), "WATERMARK", fill=(255, 0, 0))
    # 添加白色遮挡方块
    draw.rectangle([300, 200, 400, 300], fill=(255, 255, 255))
    damaged_img.save("D:/sd_project/assets/damaged_cat.png")
    
    # 创建mask（覆盖损坏区域）
    mask = Image.new("L", damaged_img.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.text((50, 50), "WATERMARK", fill=255)  # mask覆盖文字
    draw.rectangle([300, 200, 400, 300], fill=255)  # mask覆盖方块
    
    # 执行修复（修改prompt提升效果）
    gen = ImageInpainter()
    result = gen.inpaint(
        image=damaged_img,
        mask=mask,
        prompt="orange cat with white and orange fur pattern, clear background, natural lighting, high detail",  # 更精确的描述
        steps=30,
        guidance=7.5
    )
    
    result["image"].save("D:/sd_project/assets/repaired_cat.png")
    
    # 创建3图对比
    comparison = Image.new('RGB', (512*3, 512))
    comparison.paste(original_img.resize((512, 512)), (0, 0))
    comparison.paste(damaged_img.resize((512, 512)), (512, 0))
    comparison.paste(result["image"], (1024, 0))
    
    draw = ImageDraw.Draw(comparison)
    draw.text((10, 10), "Original", fill=(255, 0, 0))
    draw.text((522, 10), "Damaged", fill=(255, 0, 0))
    draw.text((1034, 10), "Repaired", fill=(0, 255, 0))
    comparison.save("D:/sd_project/assets/repair_comparison.png")
    
    print(f"✓ 修复成功！显存占用: {result['memory_gb']:.2f} GB")
    print("✓ 对比图已保存")