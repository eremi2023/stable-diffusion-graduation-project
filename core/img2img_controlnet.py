import torch
import numpy as np
from PIL import Image, ImageFilter
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel
import cv2
import time
import os

class Image2ImageControlNetGenerator:
    def __init__(self, 
                 model_path="D:/sd_model/AI-ModelScope/stable-diffusion-v1-5",
                 controlnet_path="D:/sd_model/controlnet_canny"):
        """初始化ControlNet图生图生成器（使用Img2Img专用Pipeline）"""
        print("正在加载ControlNet模型...")
        
        # 加载ControlNet
        controlnet = ControlNetModel.from_pretrained(
            controlnet_path,
            torch_dtype=torch.float16,
            local_files_only=True
        )
        
        # 关键：使用 StableDiffusionControlNetImg2ImgPipeline
        self.pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            model_path,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
            local_files_only=True
        ).to("cuda")
        
        # 显存优化
        self.pipe.enable_attention_slicing()
        print("✓ ControlNet Img2Img模型加载完成")
    
    def generate_face_mask(self, width=512, height=512):
        """生成面部保护mask（圆形中央区域）"""
        # 创建圆形mask，保护脸部
        mask = np.zeros((height, width), dtype=np.float32)
        center_x, center_y = width // 2, height // 2
        radius = min(width, height) // 3  # 保护中央1/3区域
        
        cv2.circle(mask, (center_x, center_y), radius, 1.0, -1)
        
        # 边缘羽化
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        
        return torch.from_numpy(mask).to("cuda", dtype=torch.float16)
    
    def generate(self, init_image, prompt, negative_prompt="",
                 strength=0.65, steps=25, guidance=8.0, seed=-1,
                 controlnet_scale=1.5, face_protect=True):
        """
        ControlNet图生图生成（带面部保护）
        
        参数:
            init_image: PIL.Image对象（原图）
            prompt: 风格描述文本
            negative_prompt: 反向提示词
            strength: 重绘幅度(0.5-0.9)
            steps: 采样步数
            guidance: 引导系数
            seed: 随机种子
            controlnet_scale: ControlNet强度(0.5-2.0)
            face_protect: 是否启用面部保护
        """
        if seed != -1:
            torch.manual_seed(seed)
        
        torch.cuda.empty_cache()
        init_image = init_image.resize((512, 512)).convert("RGB")
        
        # Canny边缘提取
        init_array = np.array(init_image)
        canny = cv2.Canny(init_array, 100, 200)
        canny = Image.fromarray(canny).convert("RGB")
        
        # 面部保护机制
        if face_protect:
            # 创建面部mask
            face_mask = self.generate_face_mask()
            
            # 方案：生成两次，后期融合
            print("启用面部保护模式...")
            
            # 1. 全图生成（高strength，强风格）
            full_image = self.pipe(
                prompt,
                image=init_image,
                control_image=canny,
                strength=strength,
                num_inference_steps=steps,
                guidance_scale=guidance,
                negative_prompt=negative_prompt,
                controlnet_conditioning_scale=controlnet_scale
            ).images[0]
            
            # 2. 脸部保护生成（低strength，保结构）
            face_image = self.pipe(
                prompt,
                image=init_image,
                control_image=canny,
                strength=strength * 0.6,  # 脸部strength降低40%
                num_inference_steps=steps,
                guidance_scale=guidance * 0.8,
                negative_prompt=negative_prompt,
                controlnet_conditioning_scale=controlnet_scale * 1.2  # 脸部ControlNet增强
            ).images[0]
            
            # 3. 融合：脸部用face_image，身体用full_image
            full_np = np.array(full_image)
            face_np = np.array(face_image)
            
            # 扩展mask到3通道
            mask_3ch = face_mask.cpu().numpy()
            mask_3ch = np.stack([mask_3ch, mask_3ch, mask_3ch], axis=-1)
            
            # 加权融合
            blended = full_np * (1 - mask_3ch) + face_np * mask_3ch
            image = Image.fromarray(blended.astype(np.uint8))
        else:
            # 标准生成（无面部保护）
            image = self.pipe(
                prompt,
                image=init_image,
                control_image=canny,
                strength=strength,
                num_inference_steps=steps,
                guidance_scale=guidance,
                negative_prompt=negative_prompt,
                controlnet_conditioning_scale=controlnet_scale
            ).images[0]
        
        # 后处理：锐化增强细节
        image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
        
        mem = torch.cuda.max_memory_allocated() / 1024**3
        
        return {
            "image": image,
            "memory_gb": mem,
            "strength": strength,
            "face_protect": face_protect,
            "seed": seed if seed != -1 else "random"
        }

if __name__ == "__main__":
    # 测试代码
    from PIL import Image
    
    test_img = Image.open("D:/sd_project/assets/test_cat.png")
    gen = Image2ImageControlNetGenerator()
    
    # 测试1：无面部保护（对比用）
    print("测试1：无面部保护")
    result1 = gen.generate(
        init_image=test_img,
        prompt="cyberpunk style cat, (neon lights:1.5), (glowing fur:1.3), masterpiece",
        strength=0.7,
        guidance=8.0,
        steps=25,
        controlnet_scale=1.5,
        face_protect=False
    )
    result1["image"].save("D:/sd_project/assets/test_cyberpunk_no_protect.png")
    print(f"✓ 无保护版完成，显存: {result1['memory_gb']:.2f} GB")
    
    # 测试2：有面部保护
    print("\n测试2：面部保护模式")
    result2 = gen.generate(
        init_image=test_img,
        prompt="cyberpunk style cat, (neon lights:1.5), (glowing fur:1.3), masterpiece",
        strength=0.7,
        guidance=8.0,
        steps=25,
        controlnet_scale=1.5,
        face_protect=True
    )
    result2["image"].save("D:/sd_project/assets/test_cyberpunk_face_protect.png")
    print(f"✓ 保护版完成，显存: {result2['memory_gb']:.2f} GB")
    
    # 测试3：安全模式（低strength）
    print("\n测试3：低strength安全模式")
    result3 = gen.generate(
        init_image=test_img,
        prompt="cyberpunk style cat, (neon lights:1.5), (glowing fur:1.3), masterpiece",
        strength=0.55,  # 降低到0.55
        guidance=8.0,
        steps=25,
        controlnet_scale=1.8,  # 提高到1.8
        face_protect=True
    )
    result3["image"].save("D:/sd_project/assets/test_cyberpunk_safe.png")
    print(f"✓ 安全版完成，显存: {result3['memory_gb']:.2f} GB")