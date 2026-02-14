import time
import torch
import random
from core.text2img import Text2ImageGenerator

def stress_test():
    """压力测试：连续生成50次"""
    print("=" * 50)
    print("开始压力测试：连续生成50次")
    print("=" * 50)
    
    # 测试提示词库
    prompts = [
        "a red apple",
        "a cute cat",
        "sunset over mountains",
        "futuristic city",
        "portrait of a girl"
    ]
    
    results = []
    gen = Text2ImageGenerator()
    
    for i in range(1, 51):
        try:
            start_time = time.time()
            prompt = random.choice(prompts)
            
            print(f"\n[{i}/50] 生成: {prompt}")
            result = gen.generate(
                prompt=prompt,
                negative_prompt="low quality",
                width=512,
                height=512,
                steps=20,
                guidance=7.5,
                seed=-1
            )
            
            elapsed = time.time() - start_time
            mem = result['memory_gb']
            
            print(f"✓ 成功！时间: {elapsed:.1f}s, 显存: {mem:.2f}GB")
            results.append({"success": True, "time": elapsed, "mem": mem})
            
            # 每10次清理一次显存
            if i % 10 == 0:
                torch.cuda.empty_cache()
                print(f"  [清理显存]")
                
        except Exception as e:
            print(f"✗ 失败！错误: {str(e)}")
            results.append({"success": False, "error": str(e)})
    
    # 统计结果
    print("\n" + "=" * 50)
    print("压力测试报告")
    print("=" * 50)
    
    success_count = sum(1 for r in results if r["success"])
    fail_count = 50 - success_count
    success_rate = success_count / 50 * 100
    
    print(f"总次数: 50")
    print(f"成功: {success_count}")
    print(f"失败: {fail_count}")
    print(f"成功率: {success_rate:.1f}%")
    
    if success_count > 0:
        times = [r["time"] for r in results if r["success"]]
        mems = [r["mem"] for r in results if r["success"]]
        print(f"平均时间: {sum(times)/len(times):.2f}s")
        print(f"平均显存: {sum(mems)/len(mems):.2f}GB")
        print(f"最大显存: {max(mems):.2f}GB")
        print(f"最小显存: {min(mems):.2f}GB")
    
    # 判定是否达标
    print("-" * 50)
    if success_rate >= 95:
        print("🎉 测试结果: 通过！（崩溃率≤5%）")
    else:
        print("⚠️ 测试结果: 未通过（崩溃率>5%）")
    print("=" * 50)

if __name__ == "__main__":
    stress_test()