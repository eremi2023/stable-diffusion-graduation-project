import torch
from PIL import Image
from core.img2img_controlnet import Image2ImageControlNetGenerator

# 加载测试图片
test_img = Image.open("D:/sd_project/assets/test_cat.png")
gen = Image2ImageControlNetGenerator()

# 参数搜索：strength 0.5-0.7 × controlnet_scale 0.9-1.0
params = [
    (0.5, 0.9), (0.5, 0.95), (0.5, 1.0),
    (0.6, 0.9), (0.6, 0.95), (0.6, 1.0),
    (0.7, 0.9), (0.7, 0.95), (0.7, 1.0),
]

print("开始ControlNet参数搜索...")
for i, (s, c) in enumerate(params):
    result = gen.generate(
        init_image=test_img,
        prompt="cyberpunk style, neon lights, detailed face",
        strength=s,
        guidance=8.0,
        steps=25,
        controlnet_scale=c
    )
    
    filename = f"assets/controlnet_s{s}_c{c}.png"
    result["image"].save(filename)
    print(f"✓ 完成 strength={s}, controlnet_scale={c}, 显存={result['memory_gb']:.2f}GB")

print("参数搜索完成！请人工对比效果。")