"""Core API logic shared by HTTP and MCP flows."""

from concurrent.futures import ThreadPoolExecutor
import logging
import time
from typing import Any

from ai_client import post_chat_json
from config_loader import get_config, get_float
import matrix_service
from prompt_store import render_prompt
import strip_service

# Setup logger
logger = logging.getLogger("light_core")
logger.setLevel(logging.INFO)
if not logger.handlers:
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - [TIMING] %(message)s'))
    logger.addHandler(sh)

API_KEY = get_config("AIHUBMIX_API_KEY", "")
PLANNER_MODEL_ID = "gemini-2.5-flash-lite"
GEMINI_TIMEOUT_S = get_float("GEMINI_TIMEOUT_S", 180.0)


PlanResult = tuple[dict[str, Any], float]

def _plan_with_llm(instruction: str) -> PlanResult:
    """Unified LLM Planner. Returns (plan_dict, elapsed_seconds)."""
    
    t0 = time.perf_counter()
    
    # Load state context
    kb_context = strip_service.get_strip_kb_context(instruction)
    current_state = strip_service.get_current_state_desc()

    prompt = render_prompt(
        "planner",
        {"instruction": instruction, "kb_context": kb_context, "current_state": current_state},
        seed=instruction,
    )
    prompt = (
        "请严格使用中文输出所有自然语言字段。"
        "其中 strip.theme、strip.reason、matrix.reason、speakable_reason 必须是简体中文，"
        "不要输出英文解释，不要夹杂英文句子。\n\n"
        + prompt
    )

    try:
        plan, elapsed = post_chat_json(
            model=PLANNER_MODEL_ID,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.7,
            timeout_s=GEMINI_TIMEOUT_S,
            api_key=API_KEY,
        )
        logger.info(f"Planner LLM took {elapsed:.3f}s")
        return plan, elapsed
    except Exception as e:
        print(f"Planner LLM failed: {e}")
        elapsed = time.perf_counter() - t0
        # Fallback (Gentle error handling)
        return {
            "target": "both",
            "strip": {
                "theme": "默认",
                "colors": [{"name": "暖白", "rgb": [255, 200, 150]}],
                "reason": "规划失败，已使用默认暖白方案",
            },
            "matrix": {
                "scene_prompt": f"glowing icon of {instruction}, high contrast, dark background",
                "reason": "规划失败，已使用默认矩阵提示词",
            },
            "speakable_reason": "为您点亮了温暖的柔光，希望能让您感到舒适。"
        }, elapsed


def plan_instruction(instruction: str) -> PlanResult:
    return _plan_with_llm(instruction)


def _build_accept_response(instruction: str, plan: dict[str, Any], plan_time: float) -> dict[str, Any]:
    target = plan.get("target", "both")
    speakable = plan.get("speakable_reason", "好的，已为您设置。")

    result: dict[str, Any] = {
        "status": "accepted",
        "target": target,
        "instruction": instruction,
        "description": f"{target} 规划完成",
        "speakable_reason": speakable,
        "timings": {"planner_llm": round(plan_time, 3)},
    }

    if target in ("strip", "both"):
        s_plan = plan.get("strip", {})
        result["strip"] = {
            "theme": s_plan.get("theme", "默认"),
            "reason": s_plan.get("reason", ""),
            "speakable_reason": speakable,
            "mode": s_plan.get("mode", "static"),
            "speed": s_plan.get("speed"),
            "colors": s_plan.get("colors", []),
            "final_selection": s_plan.get("colors", []),
        }

    if target in ("matrix", "both"):
        m_plan = plan.get("matrix", {})
        result["matrix"] = {
            "scene_prompt": m_plan.get("scene_prompt", ""),
            "reason": m_plan.get("reason", ""),
            "speakable_reason": speakable,
            "suggested_colors": [],
            "image_model": get_config("MATRIX_IMAGE_MODEL", "flux-kontext-pro"),
            "note": "dry-run (no image generated)",
        }
        try:
            result["matrix"]["current"] = matrix_service.load_data_from_file().get("json")
        except Exception:
            pass

    return result

def determine_intent(instruction: str) -> str:
    # Deprecated: Kept for compatibility if called directly, but prefer planner.
    return "both" 


def accept_instruction(instruction: str, planned: PlanResult | None = None) -> dict[str, Any]:
    """Accept instruction using the Unified LLM Planner."""
    plan, plan_time = planned or _plan_with_llm(instruction)
    return _build_accept_response(instruction, plan, plan_time)


def generate_lighting_effect(instruction: str, planned: PlanResult | None = None) -> dict[str, Any]:
    """
    统一的灯光效果生成接口（底层核心函数）
    
    Uses Unified Planner for consistency.
    """
    
    t_start = time.perf_counter()
    
    # 1. Plan first (LLM)
    plan, plan_time = planned or _plan_with_llm(instruction)
    intent = plan.get("target", "both")
    print(f"用户指令: {instruction} -> 意图: {intent} (Planned)")

    result_desc = []
    combined_data: dict[str, Any] = {}
    errors: list[str] = []
    timings = {"planner_llm": round(plan_time, 3)}
    
    chosen_matrix_model = get_config("MATRIX_IMAGE_MODEL", "flux-kontext-pro")

    def _exec_matrix(m_plan: dict[str, Any]) -> dict[str, Any]:
        t_m = time.perf_counter()
        prompt = m_plan.get("scene_prompt", instruction)
        # Execute actual image generation (Time consuming)
        matrix_res = matrix_service.generate_matrix_data(
            prompt,
            model=chosen_matrix_model,
        )
        elapsed = time.perf_counter() - t_m
        logger.info(f"Matrix generation took {elapsed:.3f}s")
        return {
            "json": matrix_res["data"]["json"],
            "prompt_used": matrix_res.get("prompt_used"),
            "model_used": matrix_res.get("model_used", chosen_matrix_model),
            "reason": m_plan.get("reason"),
            "speakable_reason": plan.get("speakable_reason"),
            "_elapsed": elapsed # Internal tracking
        }

    def _exec_strip(s_plan: dict[str, Any]) -> dict[str, Any]:
        t_s = time.perf_counter()
        # Colors are already planned by LLM. Just save/apply them.
        colors = s_plan.get("colors", [])
        if not colors:
            # Fallback if planner returned empty colors for some reason
            colors = [{"name": "Default Blue", "rgb": [0, 170, 255]}]

        command = strip_service.normalize_strip_command(
            {
                "mode": s_plan.get("mode", "static"),
                "colors": [c.get("rgb") for c in colors if isinstance(c, dict)],
                "brightness": s_plan.get("brightness", 1.0),
                "speed": s_plan.get("speed", 2.0),
                "led_count": 60,
                "mode_options": s_plan.get("mode_options"),
            }
        )

        strip_service.save_strip_command(command)

        elapsed = time.perf_counter() - t_s
        logger.info(f"Strip execution took {elapsed:.3f}s")
        return {
            "theme": s_plan.get("theme", "Planner"),
            "reason": s_plan.get("reason"),
            "speakable_reason": plan.get("speakable_reason"),
            "mode": command["mode"],
            "speed": command["speed"],
            "brightness": command["brightness"],
            "final_selection": colors,
            "_elapsed": elapsed,  # Internal tracking
        }

    # Parallelize execution
    if intent == "both":
        with ThreadPoolExecutor(max_workers=2) as executor:
            m_plan = plan.get("matrix", {})
            s_plan = plan.get("strip", {})
            
            matrix_future = executor.submit(_exec_matrix, m_plan)
            strip_future = executor.submit(_exec_strip, s_plan)

            try:
                matrix_payload = matrix_future.result()
                timings["matrix_gen"] = round(matrix_payload.pop("_elapsed", 0), 3)
                combined_data["matrix"] = matrix_payload
                result_desc.append(f"Matrix: {matrix_payload.get('prompt_used')}")
            except Exception as e:
                errors.append(f"Matrix failed: {e}")
                print(f"Matrix error: {e}")

            try:
                strip_res = strip_future.result()
                timings["strip_exec"] = round(strip_res.pop("_elapsed", 0), 3)
                combined_data["strip"] = strip_res
                result_desc.append(f"Strip: {strip_res.get('theme')}")
            except Exception as e:
                errors.append(f"Strip failed: {e}")
                print(f"Strip error: {e}")

    elif intent == "matrix":
        try:
            m_plan = plan.get("matrix", {})
            matrix_payload = _exec_matrix(m_plan)
            timings["matrix_gen"] = round(matrix_payload.pop("_elapsed", 0), 3)
            combined_data["matrix"] = matrix_payload
            result_desc.append(f"Matrix: {matrix_payload.get('prompt_used')}")
        except Exception as e:
            errors.append(f"Matrix failed: {e}")

    else:  # strip
        try:
            s_plan = plan.get("strip", {})
            strip_res = _exec_strip(s_plan)
            timings["strip_exec"] = round(strip_res.pop("_elapsed", 0), 3)
            combined_data["strip"] = strip_res
            result_desc.append(f"Strip: {strip_res.get('theme')}")
        except Exception as e:
            errors.append(f"Strip failed: {e}")
    
    final_description = " | ".join(result_desc)
    timings["total"] = round(time.perf_counter() - t_start, 3)
    
    result: dict[str, Any] = {
        "status": "success" if not errors else "partial_failure",
        "target": intent,
        "description": final_description,
        "speakable_reason": plan.get("speakable_reason"),
        "data": combined_data,
        "timings": timings
    }
    
    if errors:
        result["errors"] = errors
    
    return result


def generate_matrix_image(
    prompt: str,
    model: str = "flux-kontext-pro",
    resolution: tuple = (16, 16),
) -> dict[str, Any]:
    """
    为LED矩阵生成图像
    
    Args:
        prompt: 图像生成提示词
        model: AI模型名称
        resolution: 目标分辨率
        
    Returns:
        包含生成结果的字典
    """
    try:
        result = matrix_service.generate_matrix_data(
            prompt,
            model=model,
            resolution=resolution,
        )
        return {
            "status": "success",
            "prompt_used": result["prompt_used"],
            "model_used": result.get("model_used", model),
            "data": result["data"],
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


def generate_strip_colors(instruction: str) -> dict[str, Any]:
    """
    为LED灯带生成颜色方案
    
    Args:
        instruction: 颜色需求描述
        
    Returns:
        包含颜色方案的字典
    """
    try:
        result = strip_service.generate_strip_colors(instruction)
        return {
            "status": "success",
            "theme": result["theme"],
            "reason": result["reason"],
            "colors": result["final_selection"]
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


def get_matrix_data(format_type: str = "json") -> dict[str, Any]:
    """
    获取当前LED矩阵数据
    
    Args:
        format_type: 返回格式（json或raw）
        
    Returns:
        矩阵数据
    """
    try:
        data = matrix_service.load_data_from_file()
        
        if format_type == "json":
            return {
                "status": "success",
                "format": "json",
                "data": data.get("json", {})
            }
        else:  # raw
            import base64
            raw_data = data.get("raw", bytearray())
            return {
                "status": "success",
                "format": "raw",
                "data": base64.b64encode(bytes(raw_data)).decode('utf-8')
            }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


def get_strip_data() -> dict[str, Any]:
    """
    获取当前LED灯带数据
    
    Returns:
        灯带颜色数据
    """
    try:
        data = strip_service.load_strip_data()
        return {
            "status": "success",
            "colors": data
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }
