import gradio as gr
from core.text2img import Text2ImageGenerator

# 全局模型实例（避免重复加载）
generator = Text2ImageGenerator()

def generate_image(prompt, negative_prompt, width, height, steps, guidance, seed):
    """Gradio调用的生成函数"""
    result = generator.generate(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        steps=steps,
        guidance=guidance,
        seed=seed
    )
    
    return result["image"], f"显存占用: {result['memory_gb']:.2f} GB | 种子: {result['seed']}"

# 创建界面
with gr.Blocks(title="Stable Diffusion图像生成系统") as demo:
    gr.Markdown("# 🎨 Stable Diffusion 图像生成系统")
    
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="正向提示词", placeholder="输入描述，如：a cute cat", lines=3)
            negative_prompt = gr.Textbox(label="反向提示词", value="low quality, blurry", lines=2)
            
            with gr.Row():
                width = gr.Slider(256, 768, 512, step=64, label="宽度")
                height = gr.Slider(256, 768, 512, step=64, label="高度")
            
            with gr.Row():
                steps = gr.Slider(10, 50, 20, step=1, label="采样步数")
                guidance = gr.Slider(5, 15, 7.5, step=0.5, label="引导系数")
            
            seed = gr.Number(-1, label="随机种子(-1表示随机)", precision=0)
            btn = gr.Button("🎨 生成图像", variant="primary")
        
        with gr.Column():
            output_image = gr.Image(label="生成结果")
            status = gr.Textbox(label="状态信息", interactive=False)
    
    btn.click(
        generate_image,
        inputs=[prompt, negative_prompt, width, height, steps, guidance, seed],
        outputs=[output_image, status]
    )

# 启动服务
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)