import gradio as gr
import torch
import gc
from PIL import Image
import time
import traceback

# 延迟加载包装器（带显存清理）
class ModelLoader:
    def __init__(self):
        self.text2img_gen = None
        self.img2img_gen = None
        self.inpaint_gen = None
        self.monitor = None
        self.current_model = None  # 追踪当前加载的模型
    
    def unload_current_model(self):
        """强制卸载当前模型并清理显存"""
        if self.current_model == "text2img" and self.text2img_gen is not None:
            print("🧹 卸载文生图模型...")
            del self.text2img_gen
            self.text2img_gen = None
        elif self.current_model == "img2img" and self.img2img_gen is not None:
            print("🧹 卸载图生图模型...")
            del self.img2img_gen
            self.img2img_gen = None
        elif self.current_model == "inpaint" and self.inpaint_gen is not None:
            print("🧹 卸载修复模型...")
            del self.inpaint_gen
            self.inpaint_gen = None
        
        # 强制垃圾回收和显存清理
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        self.current_model = None
        print(f"✓ 显存已清理，当前占用: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
    
    def get_text2img(self):
        if self.text2img_gen is None:
            self.unload_current_model()  # 先清理
            print("🔄 加载文生图模型...")
            from core.text2img import Text2ImageGenerator
            self.text2img_gen = Text2ImageGenerator()
            self.current_model = "text2img"
        return self.text2img_gen
    
    def get_img2img(self):
        if self.img2img_gen is None:
            self.unload_current_model()  # 先清理
            print("🔄 加载图生图模型...")
            from core.img2img_controlnet import Image2ImageControlNetGenerator
            self.img2img_gen = Image2ImageControlNetGenerator()
            self.current_model = "img2img"
        return self.img2img_gen
    
    def get_inpaint(self):
        if self.inpaint_gen is None:
            self.unload_current_model()  # 先清理
            print("🔄 加载修复模型...")
            from core.inpaint import ImageInpainter
            self.inpaint_gen = ImageInpainter()
            self.current_model = "inpaint"
        return self.inpaint_gen
    
    def get_monitor(self):
        if self.monitor is None:
            from core.monitor import SystemMonitor
            self.monitor = SystemMonitor()
        return self.monitor

loader = ModelLoader()

def generate_text2img(prompt, negative, width, height, steps, guidance, seed):
    """文生图接口"""
    try:
        seed = int(seed)
        steps = int(steps)
        width = int(width)
        height = int(height)
        guidance = float(guidance)
        
        gen = loader.get_text2img()
        
        result = gen.generate(prompt, negative, width, height, steps, guidance, seed)
        current_mem = torch.cuda.memory_allocated() / 1024**3
        
        return result["image"], f"✓ 生成成功！显存: {current_mem:.2f}GB (峰值: {result['memory_gb']:.2f}GB)"
    except Exception as e:
        error_msg = f"❌ 错误: {str(e)}"
        print(traceback.format_exc())
        return None, error_msg

def generate_img2img(init_image, prompt, negative, strength, steps, guidance, seed):
    """图生图接口"""
    try:
        if init_image is None:
            return None, "❌ 错误: 请先上传图片"
        
        seed = int(seed)
        steps = int(steps)
        strength = float(strength)
        guidance = float(guidance)
        
        gen = loader.get_img2img()
        
        result = gen.generate(init_image, prompt, negative, strength, steps, guidance, seed)
        current_mem = torch.cuda.memory_allocated() / 1024**3
        
        return result["image"], f"✓ 转换成功！显存: {current_mem:.2f}GB"
    except Exception as e:
        error_msg = f"❌ 错误: {str(e)}"
        print(traceback.format_exc())
        return None, error_msg

def generate_inpaint(image, mask, prompt, steps, guidance, seed):
    """图像修复接口"""
    try:
        if image is None:
            return None, "❌ 错误: 请先上传损坏图片"
        
        seed = int(seed)
        steps = int(steps)
        guidance = float(guidance)
        
        gen = loader.get_inpaint()
        
        if mask is None:
            mask = Image.new("L", image.size, 0)
        
        result = gen.inpaint(image, mask, prompt, steps=steps, guidance=guidance, seed=seed)
        current_mem = torch.cuda.memory_allocated() / 1024**3
        
        return result["image"], f"✓ 修复成功！显存: {current_mem:.2f}GB"
    except Exception as e:
        error_msg = f"❌ 错误: {str(e)}"
        print(traceback.format_exc())
        return None, error_msg

def get_system_status():
    """获取系统状态"""
    try:
        monitor = loader.get_monitor()
        status = monitor.get_status()
        return (
            f"{status.get('gpu_mem_used', 'N/A')}/{status.get('gpu_mem_total', 'N/A')}",
            status.get('gpu_util', 'N/A'),
            f"{status.get('cpu_mem_used', 'N/A')}/{status.get('cpu_mem_total', 'N/A')}",
            "运行中"
        )
    except Exception as e:
        return ("N/A", "N/A", "N/A", f"错误: {str(e)}")

# ========== 界面部分保持不变 ==========
with gr.Blocks(title="Stable Diffusion毕业设计系统", css="footer {visibility: hidden}") as demo:
    gr.Markdown("""
    # 🎨 Stable Diffusion 图像生成与处理系统
    *基于消费级GPU的轻量化部署 - 毕业设计项目*
    """)
    
    with gr.Tabs():
        with gr.TabItem("文生图"):
            with gr.Row():
                with gr.Column():
                    prompt_t2i = gr.Textbox(label="正向提示词", placeholder="输入描述，如：a cute cat", lines=3)
                    negative_t2i = gr.Textbox(label="反向提示词", value="low quality, blurry", lines=2)
                    with gr.Row():
                        width_t2i = gr.Slider(256, 768, 512, step=64, label="宽度")
                        height_t2i = gr.Slider(256, 768, 512, step=64, label="高度")
                    steps_t2i = gr.Slider(10, 50, 25, step=1, label="采样步数")
                    guidance_t2i = gr.Slider(5, 15, 7.5, step=0.5, label="引导系数")
                    seed_t2i = gr.Number(-1, label="随机种子(-1表示随机)", precision=0)
                    btn_t2i = gr.Button("🎨 生成图像", variant="primary")
                
                with gr.Column():
                    output_t2i = gr.Image(label="生成结果")
                    status_t2i = gr.Textbox(label="状态信息", interactive=False)
            
            btn_t2i.click(
                generate_text2img,
                inputs=[prompt_t2i, negative_t2i, width_t2i, height_t2i, steps_t2i, guidance_t2i, seed_t2i],
                outputs=[output_t2i, status_t2i]
            )
        
        with gr.TabItem("图生图(ControlNet)"):
            with gr.Row():
                with gr.Column():
                    init_img_i2i = gr.Image(label="上传原图", type="pil")
                    prompt_i2i = gr.Textbox(label="风格提示词", placeholder="cyberpunk style, neon lights", lines=2)
                    negative_i2i = gr.Textbox(label="反向提示词", value="blurry, low quality", lines=2)
                    strength_i2i = gr.Slider(0.1, 1.0, 0.65, step=0.05, label="重绘幅度")
                    steps_i2i = gr.Slider(10, 50, 25, step=1, label="采样步数")
                    guidance_i2i = gr.Slider(5, 15, 8.0, step=0.5, label="引导系数")
                    seed_i2i = gr.Number(-1, label="随机种子", precision=0)
                    btn_i2i = gr.Button("🔄 风格转换", variant="primary")
                
                with gr.Column():
                    output_i2i = gr.Image(label="转换结果")
                    status_i2i = gr.Textbox(label="状态信息", interactive=False)
            
            btn_i2i.click(
                generate_img2img,
                inputs=[init_img_i2i, prompt_i2i, negative_i2i, strength_i2i, steps_i2i, guidance_i2i, seed_i2i],
                outputs=[output_i2i, status_i2i]
            )
        
        with gr.TabItem("图像修复"):
            with gr.Row():
                with gr.Column():
                    image_inpaint = gr.Image(label="上传损坏图", type="pil")
                    mask_inpaint = gr.Image(label="上传mask图（可选）", type="pil")
                    prompt_inpaint = gr.Textbox(label="修复提示词", placeholder="orange cat, natural fur, high quality", lines=2)
                    steps_inpaint = gr.Slider(10, 50, 35, step=1, label="采样步数")
                    guidance_inpaint = gr.Slider(5, 15, 8.0, step=0.5, label="引导系数")
                    seed_inpaint = gr.Number(-1, label="随机种子", precision=0)
                    btn_inpaint = gr.Button("🔧 智能修复", variant="primary")
                
                with gr.Column():
                    output_inpaint = gr.Image(label="修复结果")
                    status_inpaint = gr.Textbox(label="状态信息", interactive=False)
            
            btn_inpaint.click(
                generate_inpaint,
                inputs=[image_inpaint, mask_inpaint, prompt_inpaint, steps_inpaint, guidance_inpaint, seed_inpaint],
                outputs=[output_inpaint, status_inpaint]
            )
        
        with gr.TabItem("系统监控"):
            gr.Markdown("### 实时性能监控")
            with gr.Row():
                gpu_mem = gr.Textbox(label="GPU显存", interactive=False)
                gpu_util = gr.Textbox(label="GPU占用率", interactive=False)
                cpu_mem = gr.Textbox(label="CPU内存", interactive=False)
            
            refresh_btn = gr.Button("🔄 刷新状态")
            refresh_btn.click(get_system_status, outputs=[gpu_mem, gpu_util, cpu_mem])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, show_error=True)