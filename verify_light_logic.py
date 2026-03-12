
import os
import json
import api_core
import strip_service
import matrix_service

def verify():
    print("--- 验证开始：AI 读取当前灯效并修改 ---")
    
    # 1. 模拟当前状态：蓝色呼吸
    initial_command = {
        "render_target": "cloud",
        "mode": "breath",
        "colors": [[0, 0, 255]], # 纯蓝
        "brightness": 1.0,
        "speed": 5.0, # 较慢
        "led_count": 60,
        "updated_at_ms": 0
    }
    strip_service.save_strip_command(initial_command)
    print(f"设定初始状态: {initial_command['mode']}, 颜色: {initial_command['colors']}, 速度: {initial_command['speed']}")

    # 2. 调用 AI 规划 (不实际调用 LLM 接口，仅验证 Prompt 注入逻辑)
    # 为了看到生成的 Prompt，我们临时在 api_core 中注入打印
    instruction = "让当前的呼吸效果快一点，把蓝色换成红色"
    print(f"执行指令: {instruction}")
    
    # 尝试调用（注意：这需要有效的 AIHUBMIX_API_KEY，如果没有则会走 fallback）
    try:
        result = api_core.accept_instruction(instruction)
        print("\n--- AI 规划结果 ---")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        
        # 验证是否符合预期 (如果是真实调用且模型理解了)
        if "strip" in result:
            s = result["strip"]
            print(f"\n检测结果:")
            print(f"- 目标模式: {s.get('mode')} (预期: breath)")
            print(f"- 目标速度: {s.get('speed')} (预期: 小于 5.0)")
            print(f"- 目标颜色: {s.get('colors')} (预期: 包含红色 [255, 0, 0] 相关)")
    except Exception as e:
        print(f"执行失败: {e}")

if __name__ == "__main__":
    verify()
