import asyncio
import base64
import io
import json
import os
import struct
import time
from typing import Optional, List, Literal, Any

import requests
from fastapi import (
    FastAPI,
    HTTPException,
    BackgroundTasks,
    Body,
    WebSocket,
    WebSocketDisconnect,
    UploadFile,
    File,
    Query,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response, RedirectResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from image_processor import process_image_to_led_data
from config_loader import get_bool, get_config, get_float, get_int

import matrix_service
import strip_service
import strip_effects
import api_core
from ui_templates import DEBUG_UI_HTML, PROMPT_UI_HTML
from prompt_store import (
    load_prompt_state,
    load_prompt_store,
    render_prompt_with_meta,
    save_prompt_state,
    save_prompt_store,
)


class _WebSocketManager:
    def __init__(self) -> None:
        self._connections: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections.add(websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._connections.discard(websocket)

    async def broadcast(self, message: dict) -> None:
        # Best-effort broadcast; failures remove dead connections.
        async with self._lock:
            targets = list(self._connections)

        if not targets:
            return

        dead: list[WebSocket] = []
        for ws in targets:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)

        if dead:
            async with self._lock:
                for ws in dead:
                    self._connections.discard(ws)


WS_MANAGER = _WebSocketManager()


class _MqttPublisher:
    """Optional MQTT publisher.

    Enable with env vars:
    - MQTT_ENABLED=true
    - MQTT_HOST, MQTT_PORT (optional), MQTT_TOPIC

    Dependency is optional: `pip install paho-mqtt`.
    """

    def __init__(self) -> None:
        self._client = None
        self._enabled = get_bool("MQTT_ENABLED", False)
        self._host = get_config("MQTT_HOST", "localhost")
        self._port = get_int("MQTT_PORT", 1883)
        self._topic = get_config("MQTT_TOPIC", "ambient-light/events")

        if not self._enabled:
            return

        try:
            import paho.mqtt.client as mqtt  # type: ignore

            self._client = mqtt.Client()
            self._client.connect(self._host, self._port, 60)
            self._client.loop_start()
        except Exception as e:
            # Keep service running even if MQTT is unavailable.
            print(f"MQTT disabled (init failed): {e}")
            self._client = None
            self._enabled = False

    def publish(self, payload: dict) -> None:
        if not self._enabled or self._client is None:
            return
        try:
            self._client.publish(self._topic, json.dumps(payload, ensure_ascii=False))
        except Exception as e:
            print(f"MQTT publish failed: {e}")


MQTT_PUBLISHER = _MqttPublisher()

# --- Configuration ---
API_KEY = get_config("AIHUBMIX_API_KEY", "")
AIHUBMIX_BASE_URL = "https://aihubmix.com"
MODEL_ID_ROUTER = "gpt-4o-mini"

HW_MATRIX_WIDTH = get_int("HW_MATRIX_WIDTH", 16)
HW_MATRIX_HEIGHT = get_int("HW_MATRIX_HEIGHT", 16)
HW_STRIP_LED_COUNTS = get_config("HW_STRIP_LED_COUNTS", "60")
HW_SYNC_FPS = get_float("HW_SYNC_FPS", 20.0)
HW_DEFAULT_ENCODING = get_config("HW_DEFAULT_ENCODING", "rgb565")
HW_SUPPORTED_ENCODINGS = ["rgb565", "rgb24"]
HW_BRIGHTNESS_FILE = get_config("HW_BRIGHTNESS_FILE", "latest_hw_brightness.json")

HW_FRAME_HEADER_STRUCT = struct.Struct(
    ">4sBBBBHIQHHHI"
)
HW_FRAME_MAGIC = b"ALHW"
HW_FRAME_VERSION = 1
HW_FRAME_TARGET_MATRIX = 1
HW_FRAME_TARGET_STRIP = 2
HW_ENCODING_CODES = {"rgb24": 1, "rgb565": 2, "rgb111": 3}

OPENAPI_TAGS = [
    {
        "name": "Voice",
        "description": "语音指令入口：先返回规划与口播文案，后台异步生成效果。",
    },
    {
        "name": "App",
        "description": "App 控制接口：提交指令或控制灯带模式。",
    },
    {
        "name": "Hardware",
        "description": "硬件读取接口：矩阵/灯带数据与渲染帧。",
    },
    {
        "name": "Hardware Gateway",
        "description": "硬件网关接口：多通道同步下发与控制码。",
    },
    {
        "name": "Matrix",
        "description": "矩阵工具：上传图片下采样 + 生成动画并流式推送。",
    },
]

app = FastAPI(
    title="氛围灯AI Agent",
    description=(
        "基于自然语言生成矩阵像素图与灯带配色的统一入口。\n\n"
        "快速上手：\n"
        "1) 调用 /api/voice/submit 或 /api/app/submit 获取口播文案与执行规划；\n"
        "2) 使用 /api/matrix/animate 生成矩阵动画（可流式推送）；\n"
        "3) 使用 /api/data/* 获取落盘数据或渲染帧；\n"
        "4) 访问 /ui 进行可视化调试。\n\n"
        "提示：生成结果可能通过 WebSocket / MQTT 推送，前端可订阅实时状态。"
    ),
    version="1.0.0",
    openapi_tags=OPENAPI_TAGS,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class VoiceAcceptRequest(BaseModel):
    instruction: str = Field(
        ...,
        description="语音识别后的用户原文指令（不触发生图/落盘，仅返回解释与规划）",
        examples=["我有点困，灯带调亮一点，矩阵显示警示图标"],
    )


class VoiceStripColor(BaseModel):
    name: Optional[str] = Field(default=None, description="颜色名称（可选）")
    rgb: list[int] = Field(..., description="RGB 三元组", examples=[[255, 140, 60]])


class VoiceStripPlan(BaseModel):
    theme: Optional[str] = Field(default=None, description="主题")
    reason: Optional[str] = Field(default=None, description="选择该主题/颜色的理由")
    speakable_reason: Optional[str] = Field(default=None, description="适合口播的一句话理由")
    mode: Optional[str] = Field(default=None, description="灯带模式")
    speed: Optional[float] = Field(default=None, description="模式速度参数")
    colors: list[VoiceStripColor] = Field(default_factory=list, description="颜色列表")
    error: Optional[str] = Field(default=None, description="错误信息（如有）")


class VoiceMatrixPlan(BaseModel):
    scene_prompt: Optional[str] = Field(default=None, description="建议用于矩阵的场景提示词")
    reason: Optional[str] = Field(default=None, description="生成该场景提示的理由")
    speakable_reason: Optional[str] = Field(default=None, description="适合口播的一句话理由")
    suggested_colors: list[list[int]] = Field(default_factory=list, description="建议颜色（可选）")
    image_model: Optional[str] = Field(default=None, description="建议的生图模型")
    current: Optional[Any] = Field(default=None, description="当前矩阵数据（如存在）")
    note: Optional[str] = Field(default=None, description="额外说明")
    error: Optional[str] = Field(default=None, description="错误信息（如有）")


class VoiceAcceptResponse(BaseModel):
    status: str = Field(..., description="状态：accepted")
    target: str = Field(..., description="目标：matrix / strip / both")
    instruction: str = Field(..., description="用户原始指令")
    description: str = Field(..., description="简要描述")
    speakable_reason: str = Field(..., description="适合口播的一句话理由")
    matrix: Optional[VoiceMatrixPlan] = Field(default=None, description="矩阵侧规划")
    strip: Optional[VoiceStripPlan] = Field(default=None, description="灯带侧规划")
    timings: Optional[dict] = Field(default=None, description="各步骤耗时（秒）")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "accepted",
                    "target": "both",
                    "instruction": "营造一个温暖放松的氛围，矩阵显示落日海边像素风",
                    "description": "Planning complete for both",
                    "speakable_reason": "已为你点亮温暖橘光，像夕阳般抚平疲惫。",
                    "matrix": {
                        "scene_prompt": "pixel art sunset beach with a single glowing sun, high contrast, dark background",
                        "reason": "选择高对比简洁主体，适合 16×16 像素显示。",
                        "speakable_reason": "已为你点亮温暖橘光，像夕阳般抚平疲惫。",
                        "suggested_colors": [],
                        "image_model": "flux-kontext-pro",
                        "current": {
                            "width": 16,
                            "height": 16,
                            "pixels": [[[0, 0, 0]]],
                        },
                        "note": "dry-run (no image generated)",
                    },
                    "strip": {
                        "theme": "温暖放松",
                        "reason": "使用暖色系营造舒缓氛围。",
                        "speakable_reason": "已为你点亮温暖橘光，像夕阳般抚平疲惫。",
                        "colors": [
                            {"name": "Warm Orange", "rgb": [255, 140, 60]},
                            {"name": "Soft Pink", "rgb": [255, 160, 190]},
                        ],
                    },
                    "timings": {"planner_llm": 0.42},
                }
            ]
        }
    }


class StripCommand(BaseModel):
    render_target: Literal["cloud", "device"] = Field(
        "cloud",
        description="渲染侧：cloud=云端算帧推送，device=端侧算帧",
    )
    mode: Literal["static", "breath", "chase", "pulse", "flow", "wave", "sparkle"] = Field(
        "static",
        description="灯带模式：常亮/呼吸/流星/脉冲/流动/波浪/闪烁",
    )
    colors: list[VoiceStripColor] = Field(
        default_factory=list,
        description="颜色列表（建议 1~3 个，第一色为主题核心色）",
    )
    brightness: float = Field(1.0, ge=0.0, le=1.0, description="亮度（0~1）")
    speed: float = Field(
        2.0,
        gt=0.0,
        description=(
            "速度参数（与 mode 相关）：breath/flow/wave/sparkle=周期秒数；pulse=周期秒数（可用 mode_options 覆盖）；chase=LED/秒。"
        ),
    )
    led_count: int = Field(60, ge=1, le=2000, description="灯带 LED 数量")
    mode_options: dict | None = Field(
        default=None,
        description="模式扩展参数（如 pulse 的 period_s/duty）",
    )


class StripCommandEnvelope(BaseModel):
    command: StripCommand
    updated_at_ms: int = Field(..., description="更新时间戳（毫秒）")


class HwBrightnessState(BaseModel):
    matrix: float = Field(1.0, ge=0.0, le=1.0, description="矩阵亮度（0~1）")
    strip: float = Field(1.0, ge=0.0, le=1.0, description="灯带亮度（0~1）")


class HwBrightnessEnvelope(BaseModel):
    brightness: HwBrightnessState
    updated_at_ms: int = Field(..., description="更新时间戳（毫秒）")


class HwStripConfig(BaseModel):
    id: int = Field(..., description="灯带编号（1..N）")
    led_count: int = Field(..., description="灯带 LED 数量")


class HwMatrixConfig(BaseModel):
    width: int = Field(..., description="矩阵宽度")
    height: int = Field(..., description="矩阵高度")
    channels: list[str] = Field(default_factory=list, description="矩阵通道列表")


class HwConfigResponse(BaseModel):
    matrix: HwMatrixConfig
    strips: list[HwStripConfig]
    encodings: list[str] = Field(default_factory=list, description="支持的编码")
    sync_fps: float = Field(..., description="同步帧率（fps）")


class HwCommandItem(BaseModel):
    channel: str = Field(..., description="通道，例如 strip:1 或 matrix:0")
    kind: Literal["color_mode", "raw_stream"]
    mode_code: Optional[str] = Field(default=None, description="控制码（color_mode）")
    params: Optional[dict] = Field(default=None, description="控制参数（透传）")
    enabled: Optional[bool] = Field(default=None, description="是否启用 raw stream")
    fps: Optional[float] = Field(default=None, description="raw stream FPS")
    encoding: Optional[str] = Field(default=None, description="raw stream 编码")


class HwCommandListResponse(BaseModel):
    updated_at_ms: int = Field(..., description="命令更新时间戳（毫秒）")
    commands: list[HwCommandItem] = Field(default_factory=list, description="命令列表")


# --- Matrix Image Utilities ---


class MatrixPixelData(BaseModel):
    width: int = Field(..., description="目标宽度", examples=[16])
    height: int = Field(..., description="目标高度", examples=[16])
    pixels: list[list[list[int]]] = Field(
        ..., description="像素矩阵（每个像素为 [R,G,B]）"
    )


class MatrixDownsampleResponse(BaseModel):
    json: MatrixPixelData = Field(..., description="下采样后的矩阵像素数据")
    raw_base64: Optional[str] = Field(
        default=None,
        description="RGB 原始字节流（base64），按行展开：R,G,B,R,G,B,...",
    )
    filename: Optional[str] = Field(default=None, description="上传文件名")
    content_type: Optional[str] = Field(default=None, description="上传文件类型")


class MatrixAnimationRequest(BaseModel):
    instruction: str = Field(..., description="用户自然语言指令")
    width: int = Field(16, ge=1, le=64, description="矩阵宽度")
    height: int = Field(16, ge=1, le=64, description="矩阵高度")
    fps: float = Field(12.0, ge=1.0, le=60.0, description="目标帧率")
    duration_s: float = Field(0.0, ge=0.0, le=300.0, description="动画持续时间（秒，0 表示持续播放）")
    store_frames: bool = Field(True, description="是否落盘完整帧序列")


class MatrixAnimationResponse(BaseModel):
    status: str = Field(..., description="状态：accepted")
    instruction: str = Field(..., description="用户原始指令")
    summary: str = Field(..., description="动画摘要")
    width: int = Field(..., description="矩阵宽度")
    height: int = Field(..., description="矩阵高度")
    fps: float = Field(..., description="目标帧率")
    duration_s: float = Field(..., description="动画持续时间（秒）")
    model_used: str = Field(..., description="动画脚本模型")
    note: Optional[str] = Field(default=None, description="额外说明")
    code: Optional[str] = Field(default=None, description="生成的 Python 动画脚本")
    timings: Optional[dict] = Field(default=None, description="生成耗时（秒）")
    job_id: Optional[str] = Field(default=None, description="异步任务 ID")
    status_url: Optional[str] = Field(default=None, description="任务状态 URL")
    async_mode: Optional[bool] = Field(default=None, description="是否异步执行")


class MatrixAnimationSavedEntry(BaseModel):
    id: str = Field(..., description="保存项 ID")
    instruction: str = Field(..., description="原始指令")
    code: str = Field(..., description="动画脚本代码")
    created_at_ms: int = Field(..., description="保存时间戳（毫秒）")


class MatrixAnimationSavedListResponse(BaseModel):
    items: list[MatrixAnimationSavedEntry] = Field(default_factory=list, description="收藏列表")


class MatrixAnimationSaveResponse(BaseModel):
    saved: MatrixAnimationSavedEntry = Field(..., description="保存的动画")


class MatrixAnimationJobResponse(BaseModel):
    job_id: str = Field(..., description="任务 ID")
    status: str = Field(..., description="任务状态")
    created_at_ms: int = Field(..., description="创建时间戳（毫秒）")
    updated_at_ms: int = Field(..., description="更新时间戳（毫秒）")
    instruction: Optional[str] = Field(default=None, description="用户原始指令")
    summary: Optional[str] = Field(default=None, description="动画摘要")
    width: Optional[int] = Field(default=None, description="矩阵宽度")
    height: Optional[int] = Field(default=None, description="矩阵高度")
    fps: Optional[float] = Field(default=None, description="目标帧率")
    duration_s: Optional[float] = Field(default=None, description="动画持续时间（秒）")
    model_used: Optional[str] = Field(default=None, description="动画脚本模型")
    note: Optional[str] = Field(default=None, description="额外说明")
    error: Optional[str] = Field(default=None, description="错误信息")
    error_detail: Optional[dict] = Field(default=None, description="错误详情")
    frame_count: Optional[int] = Field(default=None, description="帧数")
    fallback_used: Optional[bool] = Field(default=None, description="是否使用兜底")
    last_frame_index: Optional[int] = Field(default=None, description="最后一帧索引")
    code: Optional[str] = Field(default=None, description="生成的动画脚本")

class PromptStoreDocument(BaseModel):
    prompts: dict[str, Any] = Field(default_factory=dict, description="提示词模板集合")

    model_config = {"extra": "allow"}


class PromptStateDocument(BaseModel):
    variants: dict[str, str] = Field(default_factory=dict, description="每类提示词的激活版本")
    ab_test: bool = Field(default=False, description="是否启用 A/B 分流")


class PromptPreviewRequest(BaseModel):
    key: str = Field(..., description="提示词 key")
    variables: dict[str, Any] = Field(default_factory=dict, description="模板变量")
    seed: Optional[str] = Field(default=None, description="A/B 分流种子")


class PromptPreviewResponse(BaseModel):
    prompt: str = Field(..., description="渲染后的提示词")
    variant_id: str = Field(..., description="命中的版本")
    template: str = Field(..., description="命中的模板")


# --- API Routes (Voice + Hardware) ---


def _default_matrix_json(*, width: int = 16, height: int = 16) -> dict:
    return {
        "width": width,
        "height": height,
        "pixels": [[[0, 0, 0] for _ in range(width)] for _ in range(height)],
    }
def _get_matrix_json() -> dict:
    data = matrix_service.load_data_from_file()
    payload = data.get("json")
    if (
        isinstance(payload, dict)
        and isinstance(payload.get("width"), int)
        and isinstance(payload.get("height"), int)
        and isinstance(payload.get("pixels"), list)
    ):
        return payload
    return _default_matrix_json()


async def _run_matrix_animation_job(
    *,
    job_id: str,
    instruction: str,
    width: int,
    height: int,
    fps: float,
    duration_s: float,
    store_frames: bool,
    include_code: bool,
) -> None:
    await matrix_service.MATRIX_ANIMATION_JOBS.update(job_id, status="planning", note="planning")
    try:
        plan = await asyncio.to_thread(
            matrix_service.generate_matrix_animation_code,
            instruction,
            width,
            height,
            fps,
            duration_s,
        )
    except Exception as e:
        error_text = str(e)
        error_detail = {"message": error_text, "missing_dependencies": []}
        await matrix_service.MATRIX_ANIMATION_JOBS.update(
            job_id,
            status="error",
            note="planning failed",
            error=error_text,
            error_detail=error_detail,
        )
        message = {
            "type": "matrix_animation_complete",
            "payload": {
                "status": "error",
                "summary": "矩阵动画",
                "width": width,
                "height": height,
                "fps": fps,
                "duration_s": duration_s,
                "frame_count": 0,
                "error": error_text,
                "fallback_used": False,
                "error_detail": error_detail,
            },
        }
        await WS_MANAGER.broadcast(message)
        MQTT_PUBLISHER.publish(message)
        return

    summary = plan.get("summary", "矩阵动画")
    model_used = plan.get("model_used", "")
    code = plan.get("code", "")
    note = None
    if plan.get("error"):
        note = f"fallback: {plan['error']}"

    duration_for_run = duration_s
    if plan.get("error") or code.strip() == matrix_service.DEFAULT_ANIMATION_CODE.strip():
        duration_for_run = 0.0

    first_frame_event = asyncio.Event()
    fallback_triggered = False

    await matrix_service.MATRIX_ANIMATION_JOBS.update(
        job_id,
        summary=summary,
        model_used=model_used,
        note=note,
        code=code if include_code else None,
        duration_s=duration_for_run,
    )

    async def _on_frame(frame_payload: dict) -> None:
        if not first_frame_event.is_set():
            first_frame_event.set()
        raw = frame_payload.get("raw", b"")
        payload = {
            "ts_ms": frame_payload.get("ts_ms"),
            "frame_index": frame_payload.get("frame_index"),
            "width": frame_payload.get("width"),
            "height": frame_payload.get("height"),
            "fps": frame_payload.get("fps"),
            "encoding": "rgb24",
            "data": base64.b64encode(raw).decode("utf-8"),
        }
        message = {"type": "matrix_frame", "payload": payload}
        await WS_MANAGER.broadcast(message)
        MQTT_PUBLISHER.publish(message)
        await matrix_service.MATRIX_ANIMATION_JOBS.update(
            job_id,
            last_frame_index=frame_payload.get("frame_index"),
        )

    async def _on_complete(payload: dict) -> None:
        status = payload.get("status") or "completed"
        await matrix_service.MATRIX_ANIMATION_JOBS.set_active(None)
        message = {
            "type": "matrix_animation_complete",
            "payload": {
                "status": status,
                "summary": payload.get("summary"),
                "width": payload.get("width"),
                "height": payload.get("height"),
                "fps": payload.get("fps"),
                "duration_s": payload.get("duration_s"),
                "frame_count": payload.get("frame_count"),
                "error": payload.get("error"),
                "fallback_used": payload.get("fallback_used"),
                "error_detail": payload.get("error_detail"),
            },
        }
        await WS_MANAGER.broadcast(message)
        MQTT_PUBLISHER.publish(message)
        await matrix_service.MATRIX_ANIMATION_JOBS.update(
            job_id,
            status=status,
            summary=payload.get("summary"),
            model_used=payload.get("model_used"),
            frame_count=payload.get("frame_count"),
            fallback_used=payload.get("fallback_used"),
            error=payload.get("error"),
            error_detail=payload.get("error_detail"),
        )

    async def _on_fallback(payload: dict) -> None:
        failed_code = payload.get("failed_code", "")
        matrix_service.set_latest_animation_plan(
            {
                "instruction": instruction,
                "code": matrix_service.DEFAULT_ANIMATION_CODE,
            }
        )
        message = {
            "type": "matrix_animation_fallback",
            "payload": {
                "reason": payload.get("reason"),
                "missing_dependencies": payload.get("missing_dependencies"),
                "failed_code": failed_code,
            },
        }
        await WS_MANAGER.broadcast(message)
        MQTT_PUBLISHER.publish(message)
        await matrix_service.MATRIX_ANIMATION_JOBS.update(
            job_id,
            note=payload.get("reason") or "fallback",
        )
        print(f"Matrix animation fallback triggered. Reason: {payload.get('reason')}")
        if failed_code:
            print(f"Failed animation code:\n{failed_code[:2000]}")

    async def _watch_first_frame() -> None:
        nonlocal fallback_triggered
        try:
            await asyncio.wait_for(first_frame_event.wait(), timeout=6.0)
            return
        except asyncio.TimeoutError:
            if fallback_triggered:
                return
            fallback_triggered = True

        reason = "no frames within timeout"
        await matrix_service.MATRIX_ANIMATION_JOBS.update(
            job_id,
            note=reason,
            fallback_used=True,
        )
        await _on_fallback(
            {
                "reason": reason,
                "missing_dependencies": [],
                "failed_code": code,
            }
        )
        try:
            await matrix_service.MATRIX_ANIMATION_RUNNER.stop()
        except Exception:
            pass
        await matrix_service.MATRIX_ANIMATION_RUNNER.start(
            code=matrix_service.DEFAULT_ANIMATION_CODE,
            instruction=instruction,
            summary="默认篝火",
            width=width,
            height=height,
            fps=fps,
            duration_s=0.0,
            store_frames=store_frames,
            model_used="default",
            on_frame=_on_frame,
            on_complete=_on_complete,
            on_fallback=_on_fallback,
        )
        await matrix_service.MATRIX_ANIMATION_JOBS.update(
            job_id,
            status="running",
            summary="默认篝火",
            model_used="default",
            duration_s=0.0,
        )

    try:
        await matrix_service.MATRIX_ANIMATION_RUNNER.start(
            code=code,
            instruction=instruction,
            summary=summary,
            width=width,
            height=height,
            fps=fps,
            duration_s=duration_for_run,
            store_frames=store_frames,
            model_used=model_used,
            on_frame=_on_frame,
            on_complete=_on_complete,
            on_fallback=_on_fallback,
        )

        asyncio.create_task(_watch_first_frame())

        await matrix_service.MATRIX_ANIMATION_JOBS.set_active(job_id)
        await matrix_service.MATRIX_ANIMATION_JOBS.update(job_id, status="running")

        start_payload = {
            "instruction": instruction,
            "summary": summary,
            "width": width,
            "height": height,
            "fps": fps,
            "duration_s": duration_for_run,
            "model_used": model_used,
        }
        start_message = {"type": "matrix_animation_start", "payload": start_payload}
        await WS_MANAGER.broadcast(start_message)
        MQTT_PUBLISHER.publish(start_message)

        matrix_service.set_latest_animation_plan(
            {
                "instruction": instruction,
                "code": code,
            }
        )
    except Exception as e:
        error_text = str(e)
        error_detail = {"message": error_text, "missing_dependencies": []}
        await matrix_service.MATRIX_ANIMATION_JOBS.set_active(None)
        await matrix_service.MATRIX_ANIMATION_JOBS.update(
            job_id,
            status="error",
            note="start failed",
            error=error_text,
            error_detail=error_detail,
        )
        message = {
            "type": "matrix_animation_complete",
            "payload": {
                "status": "error",
                "summary": summary,
                "width": width,
                "height": height,
                "fps": fps,
                "duration_s": duration_s,
                "frame_count": 0,
                "error": error_text,
                "fallback_used": False,
                "error_detail": error_detail,
            },
        }
        await WS_MANAGER.broadcast(message)
        MQTT_PUBLISHER.publish(message)


@app.post(
    "/api/voice/submit",
    response_model=VoiceAcceptResponse,
    tags=["Voice"],
    summary="提交语音指令并返回规划",
    description="返回口播文案与执行规划，实际生图与落盘在后台异步完成。",
    responses={
        400: {
            "description": "请求参数错误",
            "content": {"application/json": {"example": {"detail": "instruction is required"}}},
        },
        500: {
            "description": "服务内部错误",
            "content": {"application/json": {"example": {"detail": "planner failed"}}},
        },
    },
)
async def voice_submit(
    req: VoiceAcceptRequest,
    background_tasks: BackgroundTasks,
) -> VoiceAcceptResponse:
    instruction = (req.instruction or "").strip()
    if not instruction:
        raise HTTPException(status_code=400, detail="instruction is required")

    try:
        planned = api_core.plan_instruction(instruction)
        accepted = api_core.accept_instruction(instruction, planned)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    async def _execute_in_background() -> None:
        try:
            generated = await asyncio.to_thread(api_core.generate_lighting_effect, instruction, planned)
            message = {"type": "generate", "payload": generated}
            await WS_MANAGER.broadcast(message)
            MQTT_PUBLISHER.publish(message)
        except Exception as e:
            message = {
                "type": "generate",
                "payload": {
                    "status": "error",
                    "instruction": instruction,
                    "error": str(e),
                },
            }
            await WS_MANAGER.broadcast(message)
            MQTT_PUBLISHER.publish(message)

    background_tasks.add_task(_execute_in_background)
    return VoiceAcceptResponse.model_validate(accepted)


@app.post(
    "/api/app/submit",
    response_model=VoiceAcceptResponse,
    tags=["App"],
    summary="App 提交指令并返回规划",
    description="等价于语音入口，适用于 App 侧调用。",
    responses={
        400: {
            "description": "请求参数错误",
            "content": {"application/json": {"example": {"detail": "instruction is required"}}},
        },
        500: {
            "description": "服务内部错误",
            "content": {"application/json": {"example": {"detail": "planner failed"}}},
        },
    },
)
async def app_submit(
    req: VoiceAcceptRequest,
    background_tasks: BackgroundTasks,
) -> VoiceAcceptResponse:
    return await voice_submit(req, background_tasks)


@app.api_route(
    "/api/data/matrix/json",
    methods=["GET", "POST"],
    response_model=MatrixPixelData,
    tags=["Hardware"],
    summary="获取矩阵下采样数据 (JSON)",
    description="读取当前落盘的矩阵下采样数据（16x16）。支持 GET 和 POST 方法。",
)
def get_matrix_json() -> MatrixPixelData:

    return MatrixPixelData.model_validate(_get_matrix_json())


@app.api_route(
    "/api/data/matrix/raw",
    methods=["GET", "POST"],
    tags=["Hardware"],
    summary="获取矩阵原始字节流",
    description="读取当前矩阵原始 RGB 字节流（行优先）。支持 GET 和 POST 方法。",
)
def get_matrix_raw() -> Response:
    data = matrix_service.load_data_from_file()
    raw = data.get("raw")
    if not isinstance(raw, (bytes, bytearray)):
        raw = bytearray()
    return Response(content=bytes(raw), media_type="application/octet-stream")


@app.get(
    "/api/data/strip",
    tags=["Hardware"],
    summary="获取灯带颜色数组",
    description="读取当前灯带落盘颜色数组（RGB 列表）。",
)
def get_strip() -> list[list[int]]:
    return strip_service.load_strip_data()


def _to_strip_colors(colors: Any) -> list[VoiceStripColor]:
    out: list[VoiceStripColor] = []
    if not isinstance(colors, list):
        return out

    for c in colors:
        if isinstance(c, dict):
            rgb = c.get("rgb")
            name = c.get("name")
            if isinstance(rgb, list):
                out.append(VoiceStripColor(name=name, rgb=rgb))
            continue
        if isinstance(c, list):
            out.append(VoiceStripColor(rgb=c))
    return out


def _load_strip_command_envelope() -> StripCommandEnvelope:
    cmd = strip_service.load_strip_command()
    colors = _to_strip_colors(cmd.get("colors", []))

    mode = str(cmd.get("mode") or "static").strip().lower()
    if mode not in {"static", "breath", "chase", "pulse", "flow", "wave", "sparkle"}:
        mode = "static"

    try:
        brightness = float(cmd.get("brightness", 1.0))
    except Exception:
        brightness = 1.0

    try:
        speed = float(cmd.get("speed", 2.0))
    except Exception:
        speed = 2.0

    try:
        led_count = int(cmd.get("led_count", 60))
    except Exception:
        led_count = 60

    render_target = str(cmd.get("render_target") or "cloud").strip().lower()
    if render_target not in {"cloud", "device"}:
        render_target = "cloud"

    command = StripCommand(
        render_target=render_target,
        mode=mode,
        colors=colors,
        brightness=brightness,
        speed=speed,
        led_count=led_count,
    )
    return StripCommandEnvelope(
        command=command,
        updated_at_ms=int(cmd.get("updated_at_ms", 0)),
    )


def _clamp_brightness(value: Any, default: float = 1.0) -> float:
    try:
        normalized = float(value)
    except Exception:
        normalized = default
    return max(0.0, min(1.0, normalized))


def _load_hw_brightness_envelope() -> HwBrightnessEnvelope:
    payload: dict[str, Any] = {}
    if os.path.exists(HW_BRIGHTNESS_FILE):
        try:
            with open(HW_BRIGHTNESS_FILE, "r", encoding="utf-8") as f:
                parsed = json.load(f)
                if isinstance(parsed, dict):
                    payload = parsed
        except Exception:
            payload = {}

    brightness_payload = payload.get("brightness") if isinstance(payload.get("brightness"), dict) else payload
    if not isinstance(brightness_payload, dict):
        brightness_payload = {}

    brightness = HwBrightnessState(
        matrix=_clamp_brightness(brightness_payload.get("matrix", 1.0)),
        strip=_clamp_brightness(brightness_payload.get("strip", 1.0)),
    )
    updated_at_ms = int(payload.get("updated_at_ms", 0) or 0)
    return HwBrightnessEnvelope(brightness=brightness, updated_at_ms=updated_at_ms)


def _save_hw_brightness(brightness: HwBrightnessState) -> HwBrightnessEnvelope:
    envelope = HwBrightnessEnvelope(
        brightness=HwBrightnessState(
            matrix=_clamp_brightness(brightness.matrix),
            strip=_clamp_brightness(brightness.strip),
        ),
        updated_at_ms=int(time.time() * 1000),
    )
    with open(HW_BRIGHTNESS_FILE, "w", encoding="utf-8") as f:
        json.dump(envelope.model_dump(), f, ensure_ascii=False)
    return envelope


@app.api_route(
    "/api/data/strip/command",
    methods=["GET", "POST"],
    response_model=StripCommandEnvelope,
    tags=["Hardware"],
    summary="获取灯带控制指令",
    description="读取当前灯带控制模式、颜色、亮度、速度等参数。支持 GET 和 POST 方法。",
)
def get_strip_command() -> StripCommandEnvelope:
    return _load_strip_command_envelope()


def _normalize_strip_encoding(value: str | None) -> str | None:
    normalized = (value or "").strip().lower()
    if normalized in {"", "rgb24", "raw", "raw_base64"}:
        return "rgb24"
    if normalized in {"rgb565", "rgb16"}:
        return "rgb565"
    if normalized in {"rgb111", "bit"}:
        return "rgb111"
    return None


def _parse_hw_strip_led_counts(value: str | None) -> list[int]:
    raw = value or ""
    entries: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            count = int(part)
        except Exception:
            continue
        if count > 0:
            entries.append(count)
    return entries or [60]


def _normalize_hw_encoding(value: str | None) -> str:
    normalized = _normalize_strip_encoding(value)
    if normalized is None:
        raise HTTPException(status_code=400, detail="unsupported encoding")
    if normalized not in HW_SUPPORTED_ENCODINGS:
        raise HTTPException(status_code=400, detail="unsupported encoding")
    return normalized


def _current_sync_seq(sync_fps: float) -> int:
    safe_fps = max(1.0, min(60.0, float(sync_fps)))
    return int(time.time() * safe_fps)


def _build_hw_frame_header(
    *,
    target: int,
    encoding: str,
    flags: int,
    channel_id: int,
    sync_seq: int,
    ts_ms: int,
    param1: int,
    param2: int,
    payload_len: int,
) -> bytes:
    encoding_code = HW_ENCODING_CODES.get(encoding, 0)
    if encoding_code == 0:
        raise HTTPException(status_code=400, detail="unsupported encoding")
    return HW_FRAME_HEADER_STRUCT.pack(
        HW_FRAME_MAGIC,
        HW_FRAME_VERSION,
        target,
        encoding_code,
        flags,
        channel_id,
        sync_seq,
        ts_ms,
        param1,
        param2,
        0,
        payload_len,
    )


def _matrix_raw_to_rgb565(raw: bytes | bytearray) -> bytes:
    packed = bytearray()
    for i in range(0, len(raw), 3):
        if i + 2 >= len(raw):
            break
        r = raw[i]
        g = raw[i + 1]
        b = raw[i + 2]
        value = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)
        packed.append((value >> 8) & 0xFF)
        packed.append(value & 0xFF)
    return bytes(packed)


def _get_hw_matrix_payload(encoding: str, brightness: float | None = None) -> tuple[bytes, int, int]:
    data = matrix_service.load_data_from_file()
    raw = data.get("raw")
    if not isinstance(raw, (bytes, bytearray)):
        raw = bytearray()
    raw_bytes = bytes(raw)
    json_candidate = data.get("json")
    json_payload: dict[str, Any] = json_candidate if isinstance(json_candidate, dict) else {}
    width = int(json_payload.get("width", HW_MATRIX_WIDTH))
    height = int(json_payload.get("height", HW_MATRIX_HEIGHT))
    if brightness is None:
        brightness = _load_hw_brightness_envelope().brightness.matrix
    brightness_factor = _clamp_brightness(brightness)

    if encoding == "rgb24":
        expected = max(0, width * height * 3)
        if expected and len(raw_bytes) == expected:
            return _scale_rgb_payload(raw_bytes, brightness_factor), width, height
        pixels = json_payload.get("pixels")
        if isinstance(pixels, list):
            flat: list[list[int]] = [rgb for row in pixels for rgb in row]
            return _scale_rgb_payload(strip_effects.frame_to_raw_bytes(flat), brightness_factor), width, height
        return _scale_rgb_payload(raw_bytes, brightness_factor), width, height

    if encoding == "rgb565":
        expected = max(0, width * height * 3)
        if expected and len(raw_bytes) == expected:
            return _matrix_raw_to_rgb565(_scale_rgb_payload(raw_bytes, brightness_factor)), width, height
        pixels = json_payload.get("pixels")
        if isinstance(pixels, list):
            flat = [rgb for row in pixels for rgb in row]
            return _matrix_raw_to_rgb565(_scale_rgb_payload(strip_effects.frame_to_raw_bytes(flat), brightness_factor)), width, height
        return _matrix_raw_to_rgb565(_scale_rgb_payload(raw_bytes, brightness_factor)), width, height

    raise HTTPException(status_code=400, detail="unsupported encoding")


def _scale_rgb_payload(raw: bytes | bytearray, brightness: float) -> bytes:
    factor = _clamp_brightness(brightness)
    raw_bytes = bytes(raw)
    if factor >= 0.999:
        return raw_bytes
    if factor <= 0.0:
        return bytes(len(raw_bytes))
    return bytes(max(0, min(255, round(channel * factor))) for channel in raw_bytes)


def _get_hw_strip_payload(
    encoding: str,
    led_count: int,
    now_s: float,
    brightness: float | None = None,
) -> bytes:
    hw_brightness = _load_hw_brightness_envelope().brightness.strip if brightness is None else brightness
    payload = strip_service.render_strip_frame_payload(
        now_s=now_s,
        led_count=led_count,
        brightness_scale=hw_brightness,
        encoding=encoding,
    )
    return payload["raw"]


def _get_matrix_updated_at_ms() -> int:
    try:
        mtime = os.path.getmtime(matrix_service.DATA_FILE)
    except Exception:
        return 0
    return int(mtime * 1000)


def _build_hw_commands() -> tuple[list[dict[str, Any]], int]:
    commands: list[dict[str, Any]] = []
    updated_at_ms = _get_matrix_updated_at_ms()

    strip_led_counts = _parse_hw_strip_led_counts(HW_STRIP_LED_COUNTS)
    cmd = strip_service.load_strip_command()
    strip_updated = int(cmd.get("updated_at_ms", 0) or 0)
    updated_at_ms = max(updated_at_ms, strip_updated)

    default_encoding = _normalize_hw_encoding(HW_DEFAULT_ENCODING)
    sync_fps = max(1.0, min(60.0, HW_SYNC_FPS))

    commands.append(
        {
            "channel": "matrix:0",
            "kind": "raw_stream",
            "enabled": True,
            "fps": sync_fps,
            "encoding": default_encoding,
        }
    )

    render_target = str(cmd.get("render_target") or "cloud").strip().lower()
    mode_code = str(cmd.get("mode_code") or cmd.get("mode") or "TBD")
    params = {
        "brightness": cmd.get("brightness"),
        "speed": cmd.get("speed"),
        "colors": cmd.get("colors"),
    }

    for idx, _ in enumerate(strip_led_counts, start=1):
        channel = f"strip:{idx}"
        if render_target == "device":
            commands.append(
                {
                    "channel": channel,
                    "kind": "color_mode",
                    "mode_code": mode_code,
                    "params": params,
                }
            )
        else:
            commands.append(
                {
                    "channel": channel,
                    "kind": "raw_stream",
                    "enabled": True,
                    "fps": sync_fps,
                    "encoding": default_encoding,
                }
            )

    return commands, updated_at_ms


def _get_hw_strip_led_counts() -> list[int]:
    return _parse_hw_strip_led_counts(HW_STRIP_LED_COUNTS)


def _get_hw_strip_configs() -> list[HwStripConfig]:
    return [
        HwStripConfig(id=idx + 1, led_count=count)
        for idx, count in enumerate(_get_hw_strip_led_counts())
    ]


def _parse_hw_channel(channel: str) -> tuple[int, int]:
    normalized = (channel or "").strip().lower()
    if normalized in {"matrix", "matrix:0"}:
        return HW_FRAME_TARGET_MATRIX, 0
    if normalized.startswith("strip:"):
        try:
            channel_id = int(normalized.split(":", 1)[1])
        except Exception:
            raise HTTPException(status_code=400, detail="invalid channel")
        if channel_id <= 0:
            raise HTTPException(status_code=400, detail="invalid channel")
        return HW_FRAME_TARGET_STRIP, channel_id
    raise HTTPException(status_code=400, detail="invalid channel")


@app.get(
    "/api/data/strip/frame/raw",
    tags=["Hardware"],
    summary="获取灯带渲染帧原始字节",
    description="按当前灯带指令渲染一帧原始字节流，可指定 led_count 与 encoding。",
)
def get_strip_frame_raw(
    led_count: int | None = Query(default=None, ge=1, le=2000),
    encoding: str = Query("rgb24"),
) -> Response:
    envelope = _load_strip_command_envelope()
    try:
        payload = strip_service.render_strip_frame_payload(
            envelope.command.model_dump(),
            now_s=time.time(),
            led_count=led_count,
            encoding=encoding,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return Response(content=payload["raw"], media_type="application/octet-stream")


@app.get(
    "/api/data/strip/frame/json",
    tags=["Hardware"],
    summary="获取灯带渲染帧 JSON",
    description="按当前灯带指令渲染一帧 RGB 数组，可指定 led_count 覆盖。",
)
def get_strip_frame_json(led_count: int | None = Query(default=None, ge=1, le=2000)) -> list[list[int]]:
    envelope = _load_strip_command_envelope()
    payload = strip_service.render_strip_frame_payload(
        envelope.command.model_dump(),
        now_s=time.time(),
        led_count=led_count,
    )
    return payload["frame"]


@app.get(
    "/api/hw/v1/config",
    response_model=HwConfigResponse,
    tags=["Hardware Gateway"],
    summary="获取硬件网关配置",
    description="返回硬件网关的矩阵/灯带配置与同步参数。",
)
def get_hw_config() -> HwConfigResponse:
    sync_fps = max(1.0, min(60.0, float(HW_SYNC_FPS)))
    matrix_config = HwMatrixConfig(
        width=int(HW_MATRIX_WIDTH),
        height=int(HW_MATRIX_HEIGHT),
        channels=["matrix:0"],
    )
    return HwConfigResponse(
        matrix=matrix_config,
        strips=_get_hw_strip_configs(),
        encodings=HW_SUPPORTED_ENCODINGS,
        sync_fps=sync_fps,
    )


@app.get(
    "/api/hw/v1/commands",
    response_model=HwCommandListResponse,
    tags=["Hardware Gateway"],
    summary="获取硬件网关命令",
    description="返回当前控制指令（支持长轮询）。",
)
async def get_hw_commands(
    since: int | None = Query(default=None, ge=0),
    wait_ms: int = Query(0, ge=0, le=10000),
) -> Response:
    commands, updated_at_ms = _build_hw_commands()
    if since is not None and updated_at_ms <= since:
        if wait_ms > 0:
            await asyncio.sleep(wait_ms / 1000)
            commands, updated_at_ms = _build_hw_commands()
    if since is not None and updated_at_ms <= since:
        return Response(status_code=204)
    payload = HwCommandListResponse(
        updated_at_ms=updated_at_ms,
        commands=[HwCommandItem(**item) for item in commands],
    )
    return JSONResponse(content=payload.model_dump())


@app.get(
    "/api/hw/v1/frame/raw",
    tags=["Hardware Gateway"],
    summary="获取硬件同步帧",
    description="返回带有统一帧头的原始二进制帧（支持长轮询）。",
)
async def get_hw_frame_raw(
    channel: str = Query(..., description="通道，例如 strip:1 或 matrix:0"),
    encoding: str | None = Query(default=None, description="rgb565/rgb24"),
    since: int | None = Query(default=None, ge=0),
    wait_ms: int = Query(0, ge=0, le=10000),
) -> Response:
    normalized_encoding = _normalize_hw_encoding(encoding or HW_DEFAULT_ENCODING)
    sync_fps = max(1.0, min(60.0, float(HW_SYNC_FPS)))
    sync_seq = _current_sync_seq(sync_fps)
    if since is not None and sync_seq <= since:
        if wait_ms > 0:
            await asyncio.sleep(wait_ms / 1000)
            sync_seq = _current_sync_seq(sync_fps)
    if since is not None and sync_seq <= since:
        return Response(status_code=204)

    target, channel_id = _parse_hw_channel(channel)
    now_s = time.time()
    ts_ms = int(now_s * 1000)

    if target == HW_FRAME_TARGET_MATRIX:
        payload, width, height = _get_hw_matrix_payload(normalized_encoding)
        header = _build_hw_frame_header(
            target=target,
            encoding=normalized_encoding,
            flags=0,
            channel_id=channel_id,
            sync_seq=sync_seq,
            ts_ms=ts_ms,
            param1=width,
            param2=height,
            payload_len=len(payload),
        )
    else:
        strip_led_counts = _get_hw_strip_led_counts()
        if channel_id > len(strip_led_counts):
            raise HTTPException(status_code=404, detail="strip channel not found")
        led_count = strip_led_counts[channel_id - 1]
        payload = _get_hw_strip_payload(normalized_encoding, led_count, now_s)
        header = _build_hw_frame_header(
            target=target,
            encoding=normalized_encoding,
            flags=0,
            channel_id=channel_id,
            sync_seq=sync_seq,
            ts_ms=ts_ms,
            param1=led_count,
            param2=0,
            payload_len=len(payload),
        )

    return Response(content=header + payload, media_type="application/octet-stream")


@app.post(
    "/api/app/strip/command",
    response_model=StripCommandEnvelope,
    tags=["App"],
    summary="更新灯带控制指令",
    description="写入灯带模式/颜色/亮度等配置，并广播更新事件。",
)
async def app_set_strip_command(body: StripCommand) -> StripCommandEnvelope:
    colors = [c.rgb for c in body.colors] if body.colors else []

    cmd = {
        "render_target": body.render_target,
        "mode": body.mode,
        "colors": colors,
        "brightness": body.brightness,
        "speed": body.speed,
        "led_count": body.led_count,
    }
    strip_service.save_strip_command(cmd)

    # Keep legacy endpoint compatible.
    if colors:
        strip_service.save_strip_data(colors)

    envelope = _load_strip_command_envelope()
    message = {"type": "strip_command_update", "payload": envelope.model_dump()}
    await WS_MANAGER.broadcast(message)
    MQTT_PUBLISHER.publish(message)
    return envelope


@app.get(
    "/api/app/brightness",
    response_model=HwBrightnessEnvelope,
    tags=["App"],
    summary="获取硬件亮度控制",
    description="读取矩阵与灯带的硬件输出亮度。",
)
def app_get_brightness() -> HwBrightnessEnvelope:
    return _load_hw_brightness_envelope()


@app.post(
    "/api/app/brightness",
    response_model=HwBrightnessEnvelope,
    tags=["App"],
    summary="更新硬件亮度控制",
    description="写入矩阵与灯带的硬件输出亮度，并广播亮度更新事件。",
)
async def app_set_brightness(body: HwBrightnessState) -> HwBrightnessEnvelope:
    envelope = _save_hw_brightness(body)
    message = {"type": "brightness_update", "payload": envelope.model_dump()}
    await WS_MANAGER.broadcast(message)
    MQTT_PUBLISHER.publish(message)
    return envelope


@app.get(
    "/api/app/state",
    tags=["App"],
    summary="获取 App 聚合状态",
    description="返回当前矩阵与灯带的综合状态快照。",
)
def app_state() -> dict:
    return {
        "matrix": _get_matrix_json(),
        "strip": {
            "colors": strip_service.load_strip_data(),
            "command": _load_strip_command_envelope().model_dump(),
        },
        "brightness": _load_hw_brightness_envelope().model_dump(),
    }


@app.post(
    "/api/matrix/downsample",
    response_model=MatrixDownsampleResponse,
    tags=["Matrix"],
    summary="上传图片并下采样",
    description="上传图片并转换为指定尺寸的像素矩阵，可返回原始字节流。",
)
async def matrix_downsample(
    file: UploadFile = File(...),
    width: int = Query(16, ge=1, le=64),
    height: int = Query(16, ge=1, le=64),
    include_raw: bool = Query(True),
) -> MatrixDownsampleResponse:
    allowed_types = {"image/png", "image/jpeg", "image/jpg", "image/webp"}
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="unsupported image type")

    content = await file.read()
    max_upload_mb = get_int("MAX_UPLOAD_MB", 10)
    if len(content) > max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail="upload too large")

    try:
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="invalid image")

    max_image_pixels = get_int("MAX_IMAGE_PIXELS", 10000000)
    if (image.size[0] * image.size[1]) > max_image_pixels:
        raise HTTPException(status_code=413, detail="image too large")

    processed = process_image_to_led_data(image, (width, height))
    matrix_service.latest_led_data = processed
    matrix_service.save_data_to_file(processed)

    raw_base64 = (
        base64.b64encode(bytes(processed.get("raw", b""))).decode("utf-8")
        if include_raw
        else None
    )

    payload = {
        "json": processed.get("json") or _default_matrix_json(width=width, height=height),
        "raw_base64": raw_base64,
        "filename": file.filename,
        "content_type": file.content_type,
    }

    message = {"type": "matrix_update", "payload": payload}
    await WS_MANAGER.broadcast(message)
    MQTT_PUBLISHER.publish(message)

    return MatrixDownsampleResponse.model_validate(payload)


@app.api_route(
    "/api/matrix/animate",
    methods=["GET", "POST"],
    response_model=MatrixAnimationResponse,
    tags=["Matrix"],
    summary="生成矩阵动画并实时下发",
    description="生成动画，实时推送帧数据。支持 GET 和 POST 方法。",
)
async def matrix_animate(
    background_tasks: BackgroundTasks,
    req: MatrixAnimationRequest = Body(None),
    include_code: bool = Query(False),
    # For GET support
    instruction: Optional[str] = Query(None),
    width: Optional[int] = Query(None),
    height: Optional[int] = Query(None),
    fps: Optional[float] = Query(None),
    duration_s: Optional[float] = Query(None),
) -> MatrixAnimationResponse:
    if req:
        instruction = (req.instruction or instruction or "").strip()
        width = req.width or width or HW_MATRIX_WIDTH
        height = req.height or height or HW_MATRIX_HEIGHT
        fps = req.fps or fps or 12.0
        if req.duration_s is not None:
            duration_s = req.duration_s
        elif duration_s is not None:
            duration_s = duration_s
        else:
            duration_s = 0.0
        store_frames = req.store_frames
    else:
        instruction = (instruction or "").strip()
        width = width or HW_MATRIX_WIDTH
        height = height or HW_MATRIX_HEIGHT
        fps = fps or 12.0
        duration_s = duration_s if duration_s is not None else 0.0
        store_frames = False

    if not instruction:
        raise HTTPException(status_code=400, detail="instruction is required")

    job = await matrix_service.MATRIX_ANIMATION_JOBS.create(
        instruction=instruction,
        width=width,
        height=height,
        fps=fps,
        duration_s=duration_s,
        store_frames=store_frames,
        include_code=include_code,
    )
    job_id = job["job_id"]
    status_url = f"/api/matrix/animate/job/{job_id}"

    queued_payload = {
        "job_id": job_id,
        "instruction": instruction,
        "summary": "矩阵动画",
        "width": width,
        "height": height,
        "fps": fps,
        "duration_s": duration_s,
    }
    queued_message = {"type": "matrix_animation_queued", "payload": queued_payload}
    await WS_MANAGER.broadcast(queued_message)
    MQTT_PUBLISHER.publish(queued_message)

    if background_tasks is None:
        asyncio.create_task(
            _run_matrix_animation_job(
                job_id=job_id,
                instruction=instruction,
                width=width,
                height=height,
                fps=fps,
                duration_s=duration_s,
                store_frames=store_frames,
                include_code=include_code,
            )
        )
    else:
        background_tasks.add_task(
            _run_matrix_animation_job,
            job_id=job_id,
            instruction=instruction,
            width=width,
            height=height,
            fps=fps,
            duration_s=duration_s,
            store_frames=store_frames,
            include_code=include_code,
        )

    return MatrixAnimationResponse(
        status="accepted",
        instruction=instruction,
        summary="矩阵动画",
        width=width,
        height=height,
        fps=fps,
        duration_s=duration_s,
        model_used="pending",
        note="queued",
        code=None,
        timings=None,
        job_id=job_id,
        status_url=status_url,
        async_mode=True,
    )


@app.get(
    "/api/matrix/animate/saved",
    response_model=MatrixAnimationSavedListResponse,
    tags=["Matrix"],
    summary="读取保存的矩阵动画",
    description="返回已保存的动画脚本列表（仅保存指令与代码）。",
)
def list_saved_matrix_animations() -> MatrixAnimationSavedListResponse:
    items = matrix_service.load_saved_animations()
    return MatrixAnimationSavedListResponse(items=items)


@app.get(
    "/api/matrix/animate/job/{job_id}",
    response_model=MatrixAnimationJobResponse,
    tags=["Matrix"],
    summary="查询矩阵动画任务",
    description="返回矩阵动画异步任务的状态与详情。",
)
async def get_matrix_animation_job(job_id: str) -> MatrixAnimationJobResponse:
    job = await matrix_service.MATRIX_ANIMATION_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return MatrixAnimationJobResponse.model_validate(job)


@app.post(
    "/api/matrix/animate/save",
    response_model=MatrixAnimationSaveResponse,
    tags=["Matrix"],
    summary="保存当前矩阵动画",
    description="保存最近一次生成的动画脚本（仅保存指令与代码）。",
)
def save_matrix_animation() -> MatrixAnimationSaveResponse:
    try:
        saved = matrix_service.save_latest_animation()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return MatrixAnimationSaveResponse(saved=saved)


@app.post(
    "/api/matrix/animate/stop",
    tags=["Matrix"],
    summary="停止矩阵动画",
    description="停止正在运行的矩阵动画并广播停止事件。",
)
async def matrix_animate_stop() -> dict:
    await matrix_service.MATRIX_ANIMATION_RUNNER.stop()
    active_job_id = await matrix_service.MATRIX_ANIMATION_JOBS.get_active()
    if active_job_id:
        await matrix_service.MATRIX_ANIMATION_JOBS.update(
            active_job_id,
            status="stopped",
            note="stopped",
        )
        await matrix_service.MATRIX_ANIMATION_JOBS.set_active(None)
    message = {
        "type": "matrix_animation_complete",
        "payload": {
            "status": "stopped",
            "summary": "animation stopped",
        },
    }
    await WS_MANAGER.broadcast(message)
    MQTT_PUBLISHER.publish(message)
    return {"status": "stopped"}


# --- Prompt Management ---


@app.get(
    "/api/prompts/store",
    tags=["App"],
    summary="读取提示词模板",
    description="返回当前 prompts.json 内容。",
)
def get_prompt_store() -> dict:
    return load_prompt_store()


@app.post(
    "/api/prompts/store",
    response_model=PromptStoreDocument,
    tags=["App"],
    summary="更新提示词模板",
    description="覆盖保存 prompts.json（完整结构）。",
)
def update_prompt_store(payload: PromptStoreDocument) -> PromptStoreDocument:
    store = payload.model_dump()
    save_prompt_store(store)
    return PromptStoreDocument.model_validate(store)


@app.get(
    "/api/prompts/state",
    tags=["App"],
    summary="读取提示词状态",
    description="返回当前提示词版本/分流状态。",
)
def get_prompt_state() -> PromptStateDocument:
    state = load_prompt_state()
    return PromptStateDocument.model_validate(state)


@app.post(
    "/api/prompts/state",
    response_model=PromptStateDocument,
    tags=["App"],
    summary="更新提示词状态",
    description="更新提示词版本选择与 A/B 分流开关。",
)
def update_prompt_state(payload: PromptStateDocument) -> PromptStateDocument:
    state = payload.model_dump()
    save_prompt_state(state)
    return PromptStateDocument.model_validate(state)


@app.post(
    "/api/prompts/preview",
    response_model=PromptPreviewResponse,
    tags=["App"],
    summary="渲染提示词预览",
    description="用变量渲染模板，便于测试提示词。",
)
def preview_prompt(payload: PromptPreviewRequest) -> PromptPreviewResponse:
    result = render_prompt_with_meta(payload.key, payload.variables, seed=payload.seed)
    return PromptPreviewResponse.model_validate(result)


# --- Debug UI ---








@app.get("/ui", include_in_schema=False)
def debug_ui():
    return HTMLResponse(content=DEBUG_UI_HTML)


@app.get("/ui/prompts", include_in_schema=False)
def prompt_ui():
    return HTMLResponse(content=PROMPT_UI_HTML)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await WS_MANAGER.connect(websocket)
    try:
        while True:
            # Keepalive: client may send pings; we ignore payload.
            await websocket.receive_text()
    except WebSocketDisconnect:
        await WS_MANAGER.disconnect(websocket)
    except Exception:
        await WS_MANAGER.disconnect(websocket)


@app.websocket("/ws/matrix/raw")
async def websocket_matrix_raw(websocket: WebSocket):
    await websocket.accept()
    queue_obj = await matrix_service.MATRIX_FRAME_BUS.subscribe()
    try:
        while True:
            raw = await queue_obj.get()
            await websocket.send_bytes(raw)
    except WebSocketDisconnect:
        return
    except Exception:
        return
    finally:
        await matrix_service.MATRIX_FRAME_BUS.unsubscribe(queue_obj)


@app.websocket("/ws/strip/raw")
async def websocket_strip_raw(websocket: WebSocket):
    """High-frequency strip frame stream (binary).

    Query params:
    - fps: target FPS (default 20)
    - led_count: override LED count (optional)
    - encoding: rgb24 | rgb565 | rgb111 (default rgb24)

    Payload:
    - WebSocket binary message (encoding-specific).
    """

    await websocket.accept()

    def _parse_int(value: str | None, default: int) -> int:
        try:
            return int(value) if value is not None else default
        except Exception:
            return default

    def _parse_float(value: str | None, default: float) -> float:
        try:
            return float(value) if value is not None else default
        except Exception:
            return default

    fps = _parse_float(websocket.query_params.get("fps"), 20.0)
    fps = max(1.0, min(60.0, fps))
    interval = 1.0 / fps

    override_led_count = websocket.query_params.get("led_count")
    led_count_override = _parse_int(override_led_count, 0)

    encoding = _normalize_strip_encoding(websocket.query_params.get("encoding"))
    if encoding is None:
        await websocket.close(code=1008)
        return

    last_updated_at: int | None = None
    cached_cmd: dict | None = None

    try:
        while True:
            cmd = strip_service.load_strip_command()
            if str(cmd.get("render_target") or "cloud").strip().lower() != "cloud":
                await asyncio.sleep(interval)
                continue
            updated_at = cmd.get("updated_at_ms")
            if isinstance(updated_at, int) and updated_at != last_updated_at:
                cached_cmd = cmd
                last_updated_at = updated_at
            if cached_cmd is None:
                cached_cmd = cmd

            effective_led_count = (
                led_count_override
                if led_count_override > 0
                else int(cached_cmd.get("led_count", 60))
            )
            payload = strip_service.render_strip_frame_payload(
                cached_cmd,
                now_s=time.time(),
                led_count=effective_led_count,
                encoding=encoding,
            )
            await websocket.send_bytes(payload["raw"])
            await asyncio.sleep(interval)
    except WebSocketDisconnect:
        return
    except Exception:
        return


@app.websocket("/ws/hw/v1")
async def websocket_hw_gateway(websocket: WebSocket) -> None:
    await websocket.accept()

    default_encoding = _normalize_hw_encoding(HW_DEFAULT_ENCODING)
    sync_fps = max(1.0, min(60.0, float(HW_SYNC_FPS)))
    available_channels = {"matrix:0"}
    strip_led_counts = _get_hw_strip_led_counts()
    available_channels.update({f"strip:{idx}" for idx in range(1, len(strip_led_counts) + 1)})

    subscribed_channels = set(available_channels)
    encoding = default_encoding

    try:
        hello_text = await asyncio.wait_for(websocket.receive_text(), timeout=0.2)
        payload = json.loads(hello_text)
        if isinstance(payload, dict) and payload.get("type") == "hello":
            requested = payload.get("subscribe")
            if isinstance(requested, list):
                filtered = {item for item in requested if item in available_channels}
                if filtered:
                    subscribed_channels = filtered
            prefer = payload.get("prefer_encoding")
            if isinstance(prefer, str):
                try:
                    encoding = _normalize_hw_encoding(prefer)
                except HTTPException:
                    encoding = default_encoding
    except asyncio.TimeoutError:
        pass
    except Exception:
        pass

    await websocket.send_text(
        json.dumps(
            {
                "type": "hello_ack",
                "payload": {
                    "sync_fps": sync_fps,
                    "encoding": encoding,
                    "channels": sorted(subscribed_channels),
                },
            },
            ensure_ascii=False,
        )
    )

    stop_event = asyncio.Event()

    async def _recv_loop() -> None:
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            stop_event.set()
        except Exception:
            stop_event.set()

    recv_task = asyncio.create_task(_recv_loop())

    last_commands_ts: int | None = None
    last_sync_seq: int | None = None
    last_brightness_ts: int | None = None

    try:
        while not stop_event.is_set():
            now_s = time.time()
            sync_seq = _current_sync_seq(sync_fps)
            if last_sync_seq is not None and sync_seq == last_sync_seq:
                await asyncio.sleep(0.005)
                continue
            last_sync_seq = sync_seq

            commands, updated_at_ms = _build_hw_commands()
            if last_commands_ts is None or updated_at_ms != last_commands_ts:
                filtered_commands = [
                    item for item in commands if item.get("channel") in subscribed_channels
                ]
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "commands",
                            "payload": {
                                "updated_at_ms": updated_at_ms,
                                "commands": filtered_commands,
                            },
                        },
                        ensure_ascii=False,
                    )
                )
                last_commands_ts = updated_at_ms

            brightness = _load_hw_brightness_envelope()
            if last_brightness_ts is None or brightness.updated_at_ms != last_brightness_ts:
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "brightness_update",
                            "payload": brightness.model_dump(),
                        },
                        ensure_ascii=False,
                    )
                )
                last_brightness_ts = brightness.updated_at_ms

            ts_ms = int(now_s * 1000)
            if "matrix:0" in subscribed_channels:
                payload, width, height = _get_hw_matrix_payload(encoding, brightness.brightness.matrix)
                header = _build_hw_frame_header(
                    target=HW_FRAME_TARGET_MATRIX,
                    encoding=encoding,
                    flags=0,
                    channel_id=0,
                    sync_seq=sync_seq,
                    ts_ms=ts_ms,
                    param1=width,
                    param2=height,
                    payload_len=len(payload),
                )
                await websocket.send_bytes(header + payload)

            cmd = strip_service.load_strip_command()
            render_target = str(cmd.get("render_target") or "cloud").strip().lower()
            if render_target == "cloud":
                for channel in sorted(subscribed_channels):
                    if not channel.startswith("strip:"):
                        continue
                    try:
                        channel_id = int(channel.split(":", 1)[1])
                    except Exception:
                        continue
                    if channel_id <= 0 or channel_id > len(strip_led_counts):
                        continue
                    led_count = strip_led_counts[channel_id - 1]
                    payload = _get_hw_strip_payload(
                        encoding,
                        led_count,
                        now_s,
                        brightness.brightness.strip,
                    )
                    header = _build_hw_frame_header(
                        target=HW_FRAME_TARGET_STRIP,
                        encoding=encoding,
                        flags=0,
                        channel_id=channel_id,
                        sync_seq=sync_seq,
                        ts_ms=ts_ms,
                        param1=led_count,
                        param2=0,
                        payload_len=len(payload),
                    )
                    await websocket.send_bytes(header + payload)

            await asyncio.sleep(max(0.0, (1.0 / sync_fps) * 0.95))
    finally:
        stop_event.set()
        recv_task.cancel()
        try:
            await recv_task
        except Exception:
            pass


_MQTT_STRIP_STREAM_TASK: asyncio.Task | None = None


async def _mqtt_strip_stream_loop(*, fps: float = 20.0, encoding: str = "rgb24") -> None:
    interval = 1.0 / max(1.0, min(60.0, fps))

    last_updated_at: int | None = None
    cached_cmd: dict | None = None
    frame_index = 0
    normalized_encoding = _normalize_strip_encoding(encoding) or "rgb24"

    while True:
        try:
            cmd = strip_service.load_strip_command()
            if str(cmd.get("render_target") or "cloud").strip().lower() != "cloud":
                await asyncio.sleep(interval)
                continue
            updated_at = cmd.get("updated_at_ms")
            if isinstance(updated_at, int) and updated_at != last_updated_at:
                cached_cmd = cmd
                last_updated_at = updated_at
            if cached_cmd is None:
                cached_cmd = cmd

            led_count = int(cached_cmd.get("led_count", 60))
            render_payload = strip_service.render_strip_frame_payload(
                cached_cmd,
                now_s=time.time(),
                led_count=led_count,
                encoding=normalized_encoding,
            )

            payload = {
                "ts_ms": int(time.time() * 1000),
                "frame_index": frame_index,
                "encoding": render_payload["meta"]["encoding"],
                "bit_depth": render_payload["meta"]["bit_depth"],
                "bytes_per_led": render_payload["meta"]["bytes_per_led"],
                "led_count": led_count,
                "fps": fps,
                "transport": "base64",
                "data": base64.b64encode(render_payload["raw"]).decode("utf-8"),
            }
            message = {"type": "strip_frame", "payload": payload}
            MQTT_PUBLISHER.publish(message)
            frame_index = (frame_index + 1) % 1_000_000
        except Exception as e:
            # Best-effort; keep loop alive
            print(f"MQTT strip stream error: {e}")

        await asyncio.sleep(interval)


@app.on_event("startup")
async def _startup_tasks() -> None:
    global _MQTT_STRIP_STREAM_TASK

    enabled = get_bool("MQTT_STRIP_STREAM_ENABLED", False)
    if not enabled:
        return

    fps = get_float("STRIP_STREAM_FPS", 20.0)
    encoding = get_config("STRIP_STREAM_ENCODING", "rgb24")

    _MQTT_STRIP_STREAM_TASK = asyncio.create_task(
        _mqtt_strip_stream_loop(fps=fps, encoding=encoding)
    )


@app.on_event("shutdown")
async def _shutdown_tasks() -> None:
    global _MQTT_STRIP_STREAM_TASK
    if _MQTT_STRIP_STREAM_TASK is not None:
        _MQTT_STRIP_STREAM_TASK.cancel()
        _MQTT_STRIP_STREAM_TASK = None


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
