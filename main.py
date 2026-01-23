import json
import requests
import io
import base64
from typing import Optional, List, Literal, Any
import asyncio
import time

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
from fastapi.responses import HTMLResponse, JSONResponse, Response, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image, UnidentifiedImageError

from image_processor import process_image_to_led_data
from config_loader import get_bool, get_config, get_float, get_int

import matrix_service
import strip_service
import strip_effects
import api_core
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

# --- Models ---

class UserRequest(BaseModel):
    instruction: str = Field(..., description="用户自然语言指令", examples=["营造一个温暖放松的氛围"])

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "instruction": "营造一个温暖放松的氛围，矩阵显示落日海边像素风",
                }
            ]
        }
    }


class GenerationResult(BaseModel):
    target: str = Field(..., description="目标：matrix / strip / both")
    description: str = Field(..., description="一句话摘要")
    data: Optional[dict] = Field(default=None, description="详细数据（包含矩阵像素/灯带配色等）")
    timings: Optional[dict] = Field(default=None, description="各步骤耗时（秒）")


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
    mode: Literal["static", "breath", "chase", "gradient", "pulse", "flow"] = Field(
        "static",
        description="灯带模式：常亮/呼吸/流水/渐变/流动",
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
            "速度参数（与 mode 相关）：breath/gradient=周期秒数；chase=LED/秒。"
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
    duration_s: float = Field(30.0, ge=0.0, le=300.0, description="动画持续时间（秒，0 表示持续播放）")
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
        accepted = api_core.accept_instruction(instruction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    async def _execute_in_background() -> None:
        try:
            generated = await asyncio.to_thread(api_core.generate_lighting_effect, instruction)
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


@app.get(
    "/api/data/matrix/json",
    response_model=MatrixPixelData,
    tags=["Hardware"],
    summary="获取矩阵像素 JSON",
    description="读取当前落盘矩阵像素数据（JSON 结构）。",
)
def get_matrix_json() -> MatrixPixelData:
    return MatrixPixelData.model_validate(_get_matrix_json())


@app.get(
    "/api/data/matrix/raw",
    tags=["Hardware"],
    summary="获取矩阵原始字节流",
    description="读取当前矩阵原始 RGB 字节流（行优先）。",
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
    if mode not in {"static", "breath", "chase", "gradient", "pulse", "flow"}:
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


@app.get(
    "/api/data/strip/command",
    response_model=StripCommandEnvelope,
    tags=["Hardware"],
    summary="获取灯带控制指令",
    description="读取当前灯带控制模式、颜色、亮度、速度等参数。",
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


def _encode_strip_frame(frame: list[list[int]], encoding: str) -> tuple[bytes, dict]:
    normalized = _normalize_strip_encoding(encoding)
    if normalized == "rgb24":
        raw = strip_effects.frame_to_raw_bytes(frame)
        return raw, {"encoding": "rgb24", "bit_depth": 24, "bytes_per_led": 3}
    if normalized == "rgb565":
        raw = strip_effects.frame_to_rgb565_bytes(frame)
        return raw, {"encoding": "rgb565", "bit_depth": 16, "bytes_per_led": 2}
    if normalized == "rgb111":
        raw = strip_effects.frame_to_rgb111_bytes(frame)
        return raw, {"encoding": "rgb111", "bit_depth": 3, "bytes_per_led": None}
    raise HTTPException(status_code=400, detail="unsupported encoding")


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
    cmd = envelope.command.model_dump()

    effective_led_count = led_count if led_count is not None else int(cmd.get("led_count", 60))
    frame = strip_effects.render_strip_frame(cmd, now_s=time.time(), led_count=effective_led_count)
    raw, _ = _encode_strip_frame(frame, encoding)
    return Response(content=raw, media_type="application/octet-stream")


@app.get(
    "/api/data/strip/frame/json",
    tags=["Hardware"],
    summary="获取灯带渲染帧 JSON",
    description="按当前灯带指令渲染一帧 RGB 数组，可指定 led_count 覆盖。",
)
def get_strip_frame_json(led_count: int | None = Query(default=None, ge=1, le=2000)) -> list[list[int]]:
    envelope = _load_strip_command_envelope()
    cmd = envelope.command.model_dump()

    effective_led_count = led_count if led_count is not None else int(cmd.get("led_count", 60))
    return strip_effects.render_strip_frame(cmd, now_s=time.time(), led_count=effective_led_count)


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


@app.post(
    "/api/matrix/animate",
    response_model=MatrixAnimationResponse,
    tags=["Matrix"],
    summary="生成矩阵动画并实时下发",
    description="使用 Gemini 生成 Python 动画脚本并在沙盒中执行，实时推送帧数据。",
)
async def matrix_animate(
    req: MatrixAnimationRequest,
    include_code: bool = Query(False),
) -> MatrixAnimationResponse:
    instruction = (req.instruction or "").strip()
    if not instruction:
        raise HTTPException(status_code=400, detail="instruction is required")

    plan = matrix_service.generate_matrix_animation_code(
        instruction,
        req.width,
        req.height,
        req.fps,
        req.duration_s,
    )

    async def _on_frame(frame_payload: dict) -> None:
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

    async def _on_complete(payload: dict) -> None:
        message = {
            "type": "matrix_animation_complete",
            "payload": {
                "status": payload.get("status"),
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

    async def _on_fallback(payload: dict) -> None:
        failed_code = payload.get("failed_code", "")
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
        print(f"Matrix animation fallback triggered. Reason: {payload.get('reason')}")
        if failed_code:
            print(f"Failed animation code:\n{failed_code[:2000]}")

    await matrix_service.MATRIX_ANIMATION_RUNNER.start(
        code=plan["code"],
        instruction=instruction,
        summary=plan.get("summary", "矩阵动画"),
        width=req.width,
        height=req.height,
        fps=req.fps,
        duration_s=req.duration_s,
        store_frames=req.store_frames,
        model_used=plan.get("model_used", "gemini-3-flash"),
        on_frame=_on_frame,
        on_complete=_on_complete,
        on_fallback=_on_fallback,
    )

    start_payload = {
        "instruction": instruction,
        "summary": plan.get("summary", "矩阵动画"),
        "width": req.width,
        "height": req.height,
        "fps": req.fps,
        "duration_s": req.duration_s,
        "model_used": plan.get("model_used", "gemini-3-flash"),
    }
    start_message = {"type": "matrix_animation_start", "payload": start_payload}
    await WS_MANAGER.broadcast(start_message)
    MQTT_PUBLISHER.publish(start_message)

    note = None
    if plan.get("error"):
        note = f"fallback: {plan['error']}"

    return MatrixAnimationResponse(
        status="accepted",
        instruction=instruction,
        summary=plan.get("summary", "矩阵动画"),
        width=req.width,
        height=req.height,
        fps=req.fps,
        duration_s=req.duration_s,
        model_used=plan.get("model_used", "gemini-3-flash"),
        note=note,
        code=plan.get("code") if include_code else None,
        timings={"animator_llm": plan.get("elapsed")},
    )


@app.post(
    "/api/matrix/animate/stop",
    tags=["Matrix"],
    summary="停止矩阵动画",
    description="停止正在运行的矩阵动画并广播停止事件。",
)
async def matrix_animate_stop() -> dict:
    await matrix_service.MATRIX_ANIMATION_RUNNER.stop()
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


PROMPT_UI_HTML = r"""
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>提示词管理台</title>
  <style>
    * { box-sizing: border-box; }
    :root {
      --bg: #0b1020;
      --panel: rgba(255,255,255,0.06);
      --panel2: rgba(255,255,255,0.09);
      --text: rgba(255,255,255,0.92);
      --muted: rgba(255,255,255,0.65);
      --border: rgba(255,255,255,0.14);
      --accent: #6ee7ff;
      --accent2: #a78bfa;
      --ok: #34d399;
      --danger: #fb7185;
      --radius: 14px;
      --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      --sans: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial;
    }

    body {
      margin: 0;
      font-family: var(--sans);
      color: var(--text);
      background:
        radial-gradient(1200px 800px at 15% 10%, rgba(167, 139, 250, 0.22), transparent 55%),
        radial-gradient(1200px 800px at 85% 30%, rgba(110, 231, 255, 0.18), transparent 60%),
        var(--bg);
      min-height: 100vh;
    }

    .wrap {
      max-width: 1180px;
      margin: 28px auto;
      padding: 0 18px 40px;
    }

    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      margin-bottom: 16px;
    }

    h1 {
      margin: 0;
      font-size: 22px;
    }

    .sub {
      font-size: 13px;
      color: var(--muted);
    }

    .pill {
      padding: 8px 12px;
      border: 1px solid var(--border);
      border-radius: 999px;
      font-size: 12px;
      color: var(--muted);
      background: rgba(0,0,0,0.2);
    }

    .pill a { color: var(--text); text-decoration: none; }

    .grid {
      display: grid;
      grid-template-columns: 280px 1fr;
      gap: 14px;
    }

    @media (max-width: 980px) {
      .grid { grid-template-columns: 1fr; }
    }

    .card {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 14px;
    }

    label {
      display: grid;
      gap: 6px;
      font-size: 12px;
      color: var(--muted);
    }

    input, select, textarea {
      border-radius: 10px;
      border: 1px solid rgba(255,255,255,0.16);
      background: rgba(0,0,0,0.2);
      color: var(--text);
      padding: 8px 10px;
      outline: none;
      font-family: var(--sans);
    }

    textarea { min-height: 220px; resize: vertical; font-family: var(--mono); font-size: 12px; }

    button {
      cursor: pointer;
      border-radius: 10px;
      border: 1px solid rgba(255,255,255,0.16);
      background: rgba(0,0,0,0.2);
      color: var(--text);
      padding: 8px 10px;
      font-weight: 600;
    }

    button.primary {
      background: linear-gradient(135deg, rgba(110,231,255,0.22), rgba(167,139,250,0.22));
      border-color: rgba(110,231,255,0.3);
    }

    .row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }

    .stack { display: grid; gap: 10px; }

    .status {
      font-size: 12px;
      color: var(--muted);
    }

    .status.ok { color: var(--ok); }
    .status.bad { color: var(--danger); }

    pre {
      margin: 0;
      padding: 12px;
      border-radius: 12px;
      background: rgba(0,0,0,0.28);
      border: 1px solid rgba(255,255,255,0.12);
      font-family: var(--mono);
      font-size: 12px;
      color: rgba(255,255,255,0.85);
      white-space: pre-wrap;
    }

    .mini { font-size: 12px; color: var(--muted); }
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <div>
        <h1>提示词管理台</h1>
        <div class="sub">集中维护提示词模板、版本切换与 A/B 测试。</div>
      </div>
      <div class="pill">
        <a href="/ui" target="_blank" rel="noreferrer">调试台</a>
      </div>
    </header>

    <div class="grid">
      <div class="card">
        <div class="stack">
          <label>
            提示词 Key
            <select id="promptKey"></select>
          </label>
          <button id="addPromptBtn">新增提示词</button>
          <label>
            版本
            <select id="variantSelect"></select>
          </label>
          <label>
            权重
            <input id="variantWeight" type="number" min="0" step="0.1" />
          </label>
          <div class="row">
            <button id="addVariantBtn">新增版本</button>
            <button class="primary" id="saveVariantBtn">保存模板</button>
          </div>
          <label>
            激活版本
            <input id="activeVariant" placeholder="例如 v1" />
          </label>
          <label style="display:flex; align-items:center; gap:8px;">
            <input type="checkbox" id="abTestToggle" /> 启用 A/B 分流
          </label>
          <button id="saveStateBtn">保存版本配置</button>
          <div class="status" id="statusText">就绪</div>
        </div>
      </div>

      <div class="card">
        <div class="stack">
          <label>
            模板内容
            <textarea id="variantTemplate" placeholder="在这里编辑模板内容"></textarea>
          </label>
          <div class="row">
            <label>
              预览变量 (JSON)
              <textarea id="previewVars" style="min-height:120px;">{}</textarea>
            </label>
            <div class="stack">
              <label>
                A/B Seed
                <input id="previewSeed" placeholder="可选" />
              </label>
              <button class="primary" id="previewBtn">预览提示词</button>
              <div class="mini" id="previewMeta">-</div>
              <pre id="previewOutput">-</pre>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>

<script>
  const $ = (id) => document.getElementById(id);
  const els = {
    promptKey: $("promptKey"),
    addPromptBtn: $("addPromptBtn"),
    variantSelect: $("variantSelect"),
    variantWeight: $("variantWeight"),
    variantTemplate: $("variantTemplate"),
    addVariantBtn: $("addVariantBtn"),
    saveVariantBtn: $("saveVariantBtn"),
    activeVariant: $("activeVariant"),
    abTestToggle: $("abTestToggle"),
    saveStateBtn: $("saveStateBtn"),
    statusText: $("statusText"),
    previewVars: $("previewVars"),
    previewSeed: $("previewSeed"),
    previewBtn: $("previewBtn"),
    previewOutput: $("previewOutput"),
    previewMeta: $("previewMeta"),
  };

  let store = { prompts: {} };
  let state = { variants: {}, ab_test: false };

  function setStatus(text, ok = null) {
    els.statusText.textContent = text;
    els.statusText.classList.remove("ok", "bad");
    if (ok === true) els.statusText.classList.add("ok");
    if (ok === false) els.statusText.classList.add("bad");
  }

  async function getJson(url) {
    const r = await fetch(url);
    const text = await r.text();
    let data;
    try { data = JSON.parse(text); } catch { data = {}; }
    if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
    return data;
  }

  async function postJson(url, body) {
    const r = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const text = await r.text();
    let data;
    try { data = JSON.parse(text); } catch { data = {}; }
    if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
    return data;
  }

  function getCurrentEntry() {
    const key = els.promptKey.value;
    const entry = store.prompts[key] || { variants: [] };
    if (!Array.isArray(entry.variants)) entry.variants = [];
    return { key, entry };
  }

  function syncVariantForm() {
    const { entry } = getCurrentEntry();
    const variantId = els.variantSelect.value;
    const variant = entry.variants.find((v) => String(v.id) === String(variantId));
    if (!variant) return;
    els.variantTemplate.value = variant.template || "";
    els.variantWeight.value = variant.weight ?? 1;
  }

  function refreshVariantList() {
    const { entry, key } = getCurrentEntry();
    els.variantSelect.innerHTML = "";
    entry.variants.forEach((variant) => {
      const option = document.createElement("option");
      option.value = variant.id;
      option.textContent = `${variant.id}`;
      els.variantSelect.appendChild(option);
    });

    if (!entry.variants.length) {
      const option = document.createElement("option");
      option.value = "";
      option.textContent = "(无版本)";
      els.variantSelect.appendChild(option);
    }

    const active = state.variants[key] || entry.variants[0]?.id || "";
    els.variantSelect.value = active || els.variantSelect.value;
    els.activeVariant.value = state.variants[key] || "";
    syncVariantForm();
  }

  function refreshPromptKeys() {
    const keys = Object.keys(store.prompts || {}).sort();
    els.promptKey.innerHTML = "";
    keys.forEach((key) => {
      const option = document.createElement("option");
      option.value = key;
      option.textContent = key;
      els.promptKey.appendChild(option);
    });
    if (!keys.length) {
      const option = document.createElement("option");
      option.value = "";
      option.textContent = "(无提示词)";
      els.promptKey.appendChild(option);
    }
    refreshVariantList();
  }

  async function loadAll() {
    try {
      store = await getJson("/api/prompts/store");
      state = await getJson("/api/prompts/state");
      if (!store.prompts) store.prompts = {};
      if (!state.variants) state.variants = {};
      els.abTestToggle.checked = !!state.ab_test;
      refreshPromptKeys();
      setStatus("已加载", true);
    } catch (e) {
      setStatus(`加载失败：${e.message}`, false);
    }
  }

  async function saveVariant() {
    const { key, entry } = getCurrentEntry();
    const variantId = els.variantSelect.value;
    const variant = entry.variants.find((v) => String(v.id) === String(variantId));
    if (!variant) {
      setStatus("请选择版本", false);
      return;
    }
    variant.template = els.variantTemplate.value;
    variant.weight = Number(els.variantWeight.value || 1);
    store.prompts[key] = entry;

    try {
      await postJson("/api/prompts/store", store);
      setStatus("模板已保存", true);
    } catch (e) {
      setStatus(`保存失败：${e.message}`, false);
    }
  }

  async function addVariant() {
    const { key, entry } = getCurrentEntry();
    const newId = prompt("新版本 id：");
    if (!newId) return;
    entry.variants.push({ id: newId, weight: 1, template: "" });
    store.prompts[key] = entry;
    refreshVariantList();
    els.variantSelect.value = newId;
    syncVariantForm();
  }

  async function addPromptKey() {
    const key = prompt("新提示词 key：");
    if (!key) return;
    if (!store.prompts) store.prompts = {};
    if (!store.prompts[key]) {
      store.prompts[key] = { variants: [{ id: "v1", weight: 1, template: "" }] };
    }
    refreshPromptKeys();
    els.promptKey.value = key;
    refreshVariantList();
  }

  async function saveState() {
    const { key } = getCurrentEntry();
    state.variants = state.variants || {};
    if (els.activeVariant.value.trim()) {
      state.variants[key] = els.activeVariant.value.trim();
    } else {
      delete state.variants[key];
    }
    state.ab_test = !!els.abTestToggle.checked;

    try {
      const res = await postJson("/api/prompts/state", state);
      state = res;
      setStatus("版本配置已保存", true);
    } catch (e) {
      setStatus(`保存失败：${e.message}`, false);
    }
  }

  async function previewPrompt() {
    const { key } = getCurrentEntry();
    if (!key) {
      setStatus("请选择提示词", false);
      return;
    }
    let variables = {};
    try {
      variables = JSON.parse(els.previewVars.value || "{}");
    } catch (e) {
      setStatus("预览变量 JSON 无效", false);
      return;
    }
    try {
      const res = await postJson("/api/prompts/preview", {
        key,
        variables,
        seed: els.previewSeed.value || null,
      });
      els.previewOutput.textContent = res.prompt || "-";
      els.previewMeta.textContent = `版本：${res.variant_id}`;
      setStatus("预览完成", true);
    } catch (e) {
      setStatus(`预览失败：${e.message}`, false);
    }
  }

  els.promptKey.addEventListener("change", refreshVariantList);
  els.variantSelect.addEventListener("change", syncVariantForm);
  els.saveVariantBtn.addEventListener("click", saveVariant);
  els.addVariantBtn.addEventListener("click", addVariant);
  els.addPromptBtn.addEventListener("click", addPromptKey);
  els.saveStateBtn.addEventListener("click", saveState);
  els.previewBtn.addEventListener("click", previewPrompt);

  loadAll();
</script>
</body>
</html>
"""


DEBUG_UI_HTML = r"""

<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>氛围灯调试台</title>
  <!-- Three.js removed (using CSS gradients for strip preview) -->
  <style>
    * { box-sizing: border-box; }
    code { font-family: var(--mono); font-size: 12px; }
    :root {
      --bg: #0b1020;
      --panel: rgba(255,255,255,0.06);
      --panel2: rgba(255,255,255,0.09);
      --text: rgba(255,255,255,0.92);
      --muted: rgba(255,255,255,0.65);
      --border: rgba(255,255,255,0.14);
      --accent: #6ee7ff;
      --accent2: #a78bfa;
      --danger: #fb7185;
      --ok: #34d399;
      --shadow: 0 20px 50px rgba(0,0,0,0.35);
      --radius: 16px;
      --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      --sans: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
    }

    body {
      margin: 0;
      font-family: var(--sans);
      color: var(--text);
      background:
        radial-gradient(1200px 800px at 15% 10%, rgba(167, 139, 250, 0.22), transparent 55%),
        radial-gradient(1200px 800px at 85% 30%, rgba(110, 231, 255, 0.18), transparent 60%),
        radial-gradient(900px 700px at 50% 90%, rgba(52, 211, 153, 0.10), transparent 60%),
        var(--bg);
      min-height: 100vh;
    }

    .wrap {
      max-width: 1180px;
      margin: 28px auto;
      padding: 0 18px 38px;
    }

    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 16px;
    }

    .title {
      display: grid;
      gap: 6px;
    }

    h1 {
      font-size: 22px;
      margin: 0;
      letter-spacing: 0.2px;
    }

    .sub {
      font-size: 13px;
      color: var(--muted);
      line-height: 1.4;
    }

    .pill {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 10px;
      border: 1px solid var(--border);
      background: rgba(0,0,0,0.18);
      border-radius: 999px;
      font-size: 12px;
      color: var(--muted);
      box-shadow: var(--shadow);
    }

    .pill a { color: var(--text); text-decoration: none; border-bottom: 1px dashed rgba(255,255,255,0.3); }

    .grid {
      display: grid;
      grid-template-columns: 1.05fr 0.95fr;
      gap: 14px;
    }

    @media (max-width: 980px) {
      .grid { grid-template-columns: 1fr; }
    }

    .card {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      overflow: hidden;
    }

    .card .hd {
      padding: 14px 14px 10px;
      border-bottom: 1px solid rgba(255,255,255,0.08);
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 12px;
    }

    .card .hd .t {
      font-weight: 650;
      font-size: 14px;
    }

    .card .hd .hint {
      font-size: 12px;
      color: var(--muted);
    }

    /* New styles for speakable reason */
    .speakable-box {
        margin: 12px 0 6px;
        padding: 12px;
        border-radius: 12px;
        background: linear-gradient(135deg, rgba(167,139,250,0.15), rgba(110,231,255,0.10));
        border: 1px solid rgba(167,139,250,0.25);
        color: #e0e7ff;
        font-size: 15px;
        line-height: 1.5;
        position: relative;
    }
    .speakable-label {
        font-size: 11px;
        text-transform: uppercase;
        color: rgba(255,255,255,0.5);
        margin-bottom: 4px;
        letter-spacing: 0.5px;
        font-weight: 700;
    }

    .card .bd { padding: 14px; }

    textarea {
      width: 100%;
      height: 110px;
      resize: vertical;
      border-radius: 12px;
      border: 1px solid rgba(255,255,255,0.14);
      background: rgba(0,0,0,0.20);
      color: var(--text);
      padding: 12px;
      font-family: var(--sans);
      line-height: 1.5;
      outline: none;
    }

    textarea:focus { border-color: rgba(110,231,255,0.55); box-shadow: 0 0 0 3px rgba(110,231,255,0.12); }

    .row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-top: 10px;
    }

    @media (max-width: 520px) {
      .row { grid-template-columns: 1fr; }
    }

    label {
      display: grid;
      gap: 6px;
      font-size: 12px;
      color: var(--muted);
    }

    input, select {
      height: 38px;
      border-radius: 12px;
      border: 1px solid rgba(255,255,255,0.14);
      background: rgba(0,0,0,0.20);
      color: var(--text);
      padding: 0 10px;
      outline: none;
    }

    input:focus, select:focus { border-color: rgba(167,139,250,0.6); box-shadow: 0 0 0 3px rgba(167,139,250,0.13); }

    .btns {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 12px;
      align-items: center;
    }

    button {
      cursor: pointer;
      border: 1px solid rgba(255,255,255,0.16);
      border-radius: 12px;
      padding: 10px 12px;
      background: rgba(0,0,0,0.22);
      color: var(--text);
      font-weight: 600;
      letter-spacing: 0.2px;
      transition: transform 0.05s ease, background 0.2s ease, border-color 0.2s ease;
    }

    button:hover { background: rgba(255,255,255,0.10); border-color: rgba(255,255,255,0.22); }
    button:active { transform: translateY(1px); }

    .primary {
      background: linear-gradient(135deg, rgba(110,231,255,0.22), rgba(167,139,250,0.22));
      border-color: rgba(110,231,255,0.25);
    }

    .danger { border-color: rgba(251,113,133,0.35); }

    .meta {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
      margin-top: 10px;
      color: var(--muted);
      font-size: 12px;
    }

    .statusDot {
      width: 9px;
      height: 9px;
      border-radius: 99px;
      background: rgba(255,255,255,0.35);
      display: inline-block;
      margin-right: 6px;
    }

    .ok { background: rgba(52,211,153,0.85); }
    .bad { background: rgba(251,113,133,0.85); }

    .split {
      display: grid;
      grid-template-columns: 340px 1fr;
      gap: 12px;
    }

    @media (max-width: 980px) {
      .split { grid-template-columns: 1fr; }
    }

    .previewBox {
      background: var(--panel2);
      border: 1px solid rgba(255,255,255,0.10);
      border-radius: 14px;
      padding: 12px;
    }

    #stripPreview {
      margin-top: 10px;
      height: 24px;
      width: 100%;
      border-radius: 99px; /* Pill shape for strip */
      border: 1px solid rgba(255,255,255,0.15);
      background: #111;
      box-shadow: inset 0 2px 8px rgba(0,0,0,0.5); /* Inner shadow for depth */
      transition: background 0.1s linear;
      position: relative;
      overflow: hidden;
    }
    /* "Diffuser" overlay */
    #stripPreview::after {
      content: "";
      position: absolute;
      top: 0; left: 0; right: 0; bottom: 0;
      background: linear-gradient(to bottom, rgba(255,255,255,0.1), rgba(255,255,255,0) 40%, rgba(0,0,0,0.1));
      pointer-events: none;
    }

    #matrixCanvas {
      width: 100%;
      max-width: 320px;
      aspect-ratio: 1 / 1;
      border-radius: 12px;
      border: 1px solid rgba(255,255,255,0.12);
      background: #000;
      image-rendering: pixelated;
      --matrix-blur: 8px;
    }

    #matrixCanvas.matrix-blur {
      filter: blur(var(--matrix-blur));
      transform: scale(0.92);
    }

    .swatches {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 10px;
      align-items: stretch;
    }

    .swatch {
      display: grid;
      gap: 6px;
      padding: 10px;
      border-radius: 14px;
      border: 1px solid rgba(255,255,255,0.12);
      background: rgba(0,0,0,0.18);
      flex: 1 1 140px;
      max-width: 220px;
      min-width: 0;
      overflow: hidden;
    }

    .swatch span {
      overflow-wrap: anywhere;
    }

    .chip {
      height: 26px;
      border-radius: 10px;
      border: 1px solid rgba(255,255,255,0.10);
      background: #222;
    }

    .kv {
      display: grid;
      gap: 6px;
      margin-top: 10px;
      font-size: 13px;
    }

    .kv b { font-weight: 650; }

    pre {
      margin: 0;
      padding: 12px;
      border-radius: 14px;
      border: 1px solid rgba(255,255,255,0.12);
      background: rgba(0,0,0,0.24);
      overflow: auto;
      max-height: 320px;
      font-family: var(--mono);
      font-size: 12px;
      line-height: 1.4;
      color: rgba(255,255,255,0.85);
    }

    .mini {
      font-size: 12px;
      color: var(--muted);
    }

    .section {
      margin-top: 16px;
      padding-top: 12px;
      border-top: 1px dashed rgba(255,255,255,0.12);
    }

    .section-title {
      font-size: 13px;
      font-weight: 650;
      margin-bottom: 8px;
    }

    .inline {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }

    .input-sm {
      height: 32px;
      border-radius: 10px;
    }

    .log {
      margin-top: 8px;
      padding: 10px;
      border-radius: 12px;
      border: 1px solid rgba(255,255,255,0.12);
      background: rgba(0,0,0,0.22);
      max-height: 160px;
      overflow: auto;
      font-family: var(--mono);
      font-size: 12px;
      line-height: 1.4;
      color: rgba(255,255,255,0.85);
    }
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <div class="title">
        <h1>氛围灯调试台</h1>
        <div class="sub">用于联调 / 预览：调用 <code>/api/voice/submit</code>（先返回口播文案，后台执行生图+落盘），并通过 <code>/api/data/*</code> 预览矩阵与灯带数据。</div>
      </div>
      <div class="pill">
        <span>快捷：</span>
        <a href="/docs" target="_blank" rel="noreferrer">OpenAPI 文档</a>
        <span>·</span>
        <a href="/api/data/matrix/json" target="_blank" rel="noreferrer">当前矩阵</a>
        <span>·</span>
        <a href="/api/data/strip" target="_blank" rel="noreferrer">当前灯带</a>
        <span>·</span>
        <a href="/ui/prompts" target="_blank" rel="noreferrer">提示词管理</a>
      </div>
    </header>

    <div class="grid">
      <div class="card">
        <div class="hd">
          <div class="t">请求</div>
          <div class="hint">提交后会触发生图+落盘（可能较慢）</div>
        </div>
        <div class="bd">
          <label>
            指令（自然语言）
            <textarea id="instruction" placeholder="例如：营造一个温暖放松的氛围；矩阵显示落日海边的像素风场景"></textarea>
          </label>


          <div class="btns">
            <button class="primary" id="runBtn">运行</button>
            <button id="loadCurrentBtn">读取当前硬件数据</button>
            <button class="danger" id="clearBtn">清空</button>
          </div>

          <div class="meta" id="meta">
            <span><span class="statusDot" id="dot"></span><span id="statusText">就绪</span></span>
            <span>耗时：<span id="elapsed">-</span></span>
            <span class="mini">提示：真实生成可能较慢（生图/网络请求）</span>
          </div>

          <div class="section">
            <div class="section-title">图片下采样</div>
            <label>
              选择图片
              <input id="imageFile" type="file" accept="image/*" />
            </label>
            <div class="row">
              <label>
                宽度
                <input class="input-sm" id="matrixWidth" type="number" min="1" max="64" value="16" />
              </label>
              <label>
                高度
                <input class="input-sm" id="matrixHeight" type="number" min="1" max="64" value="16" />
              </label>
            </div>
            <div class="inline" style="margin-top:8px;">
              <label style="display:flex; align-items:center; gap:8px; font-size:12px; color: var(--muted);">
                <input type="checkbox" id="includeRaw" checked /> 包含 raw_base64
              </label>
            </div>
            <div class="btns">
              <button id="downsampleBtn">上传并下采样</button>
            </div>
            <div class="mini" id="downsampleHint">支持 PNG/JPG/WEBP，最大 10MB（可配）。</div>
          </div>

          <div class="section">
            <div class="section-title">矩阵动画</div>
            <label>
              动画指令（可独立）
              <input id="matrixAnimInstruction" placeholder="例如：像素风霓虹波纹" />
            </label>
            <div class="row">
              <label>
                FPS
                <input class="input-sm" id="matrixFps" type="number" min="1" max="60" step="1" value="12" />
              </label>
              <label>
                持续时间 (秒)
                <input class="input-sm" id="matrixDuration" type="number" min="0" max="300" step="0.5" value="30" />
              </label>
            </div>
            <div class="inline" style="margin-top:8px;">
              <label style="display:flex; align-items:center; gap:8px; font-size:12px; color: var(--muted);">
                <input type="checkbox" id="matrixStoreFrames" checked /> 落盘完整帧序列
              </label>
            </div>
            <div class="btns">
              <button class="primary" id="matrixAnimateBtn">生成动画</button>
              <button id="matrixStopBtn">停止动画</button>
            </div>
            <div class="mini" id="matrixAnimHint">使用当前矩阵宽高；持续时间填 0 可循环播放（需手动停止）。</div>
            <div class="mini" id="matrixAnimError" style="color: var(--danger);">-</div>
          </div>

          <div class="section">
            <div class="section-title">灯带指令</div>
            <div class="row">
              <label>
                模式
                <select id="stripMode">
                  <option value="static">static</option>
                  <option value="breath">breath (breathing)</option>
                  <option value="flow">flow (smooth flow)</option>
                  <option value="gradient">gradient (moving aurora)</option>
                  <option value="chase">chase (meteor)</option>
                </select>
              </label>
              <label>
                LED 数量
                <input class="input-sm" id="stripLedCount" type="number" min="1" max="2000" value="60" />
              </label>
            </div>
            <div class="row">
              <label>
                亮度
                <input class="input-sm" id="stripBrightness" type="number" min="0" max="1" step="0.05" value="1" />
              </label>
              <label>
                速度
                <input class="input-sm" id="stripSpeed" type="number" min="0.1" step="0.1" value="2" />
              </label>
            </div>
            <label>
              颜色 (rgb; 分号分隔)
              <input id="stripColors" placeholder="255,140,60;255,160,190" />
            </label>
            <div class="btns">
              <button class="primary" id="stripApplyBtn">下发指令</button>
              <button id="stripLoadBtn">读取当前指令</button>
              <button id="stripPreviewStartBtn">开始预览</button>
              <button id="stripPreviewStopBtn">停止预览</button>
            </div>
            <div class="mini" id="stripCmdHint">颜色为空时保持当前颜色。</div>
          </div>

          <div class="section">
            <div class="section-title">硬件帧</div>
            <div class="inline">
              <label>
                LED 数量
                <input class="input-sm" id="frameLedCount" type="number" min="1" max="2000" value="60" />
              </label>
              <button id="fetchFrameJsonBtn">读取 JSON 帧</button>
              <button id="fetchFrameRawBtn">读取 RAW 帧</button>
            </div>
            <div class="mini" id="frameInfo">-</div>
          </div>
        </div>
      </div>

      <div class="card">
        <div class="hd">
          <div class="t">预览</div>
          <div class="hint">矩阵：Canvas 自适应 · 灯带：色块</div>
        </div>
        <div class="bd">
          
          <!-- Speakable Reason Display -->
          <div class="speakable-box" id="speakableBox" style="display:none">
              <div class="speakable-label">AI 口语反馈 (TTS)</div>
              <div id="speakableText"></div>
          </div>

          <div class="split" style="margin-top:12px; display:block;"> <!-- Remove split grid, stack them -->
            <div class="previewBox" style="margin-bottom: 12px;">
              <div class="kv"><b>矩阵预览</b><span class="mini" id="matrixMeta">-</span></div>
              <div style="display: flex; gap: 12px; flex-wrap: wrap;">
                  <div>
                    <canvas id="matrixCanvas" class="matrix-blur" width="16" height="16"></canvas>
                    <div class="mini" style="margin-top:6px; display:flex; align-items:center; gap:10px;">
                        <label style="display:inline-flex; align-items:center; gap:6px;">
                        <input type="checkbox" id="matrixBlurToggle" checked />
                        高斯模糊预览
                        </label>
                        <label style="display:inline-flex; align-items:center; gap:6px;">
                        强度
                        <input type="range" id="matrixBlurAmount" min="0" max="16" step="1" value="8" />
                        </label>
                    </div>
                  </div>
                  <div style="flex: 1; min-width: 200px;">
                    <div class="kv">
                        <div><b>Prompt</b>：<span id="matrixScene">-</span></div>
                        <div><b>Reason</b>：<span id="matrixReason">-</span></div>
                    </div>
                  </div>
              </div>
            </div>

            <div class="previewBox">
              <div class="kv"><b>灯带预览</b> <span class="mini" id="stripMeta">-</span></div>
              <div id="stripPreview" style="margin-top:10px;"></div>
              
              <div class="kv" style="margin-top:10px;">
                <div><b>Theme</b>：<span id="stripTheme">-</span></div>
                <div><b>Reason</b>：<span id="stripReason">-</span></div>
              </div>
              <div class="swatches" id="swatches"></div>
              <div class="mini" id="stripHint" style="margin-top:10px;">-</div>
            </div>
          </div>

          <div style="margin-top:12px;" class="previewBox">
            <div class="kv"><b>原始响应（调试）</b><span class="mini">可用于粘贴给后端定位问题</span></div>
            <pre id="raw">{}</pre>
          </div>

          <div style="margin-top:12px;" class="previewBox">
            <div class="kv"><b>WebSocket 监控</b><span class="mini" id="wsStatus">未连接</span></div>
            <div class="log" id="wsLog">-</div>
          </div>
        </div>
      </div>
    </div>
  </div>

<script>
  const $ = (id) => document.getElementById(id);

  const els = {
    instruction: $("instruction"),
    runBtn: $("runBtn"),
    clearBtn: $("clearBtn"),
    loadCurrentBtn: $("loadCurrentBtn"),
    dot: $("dot"),
    statusText: $("statusText"),
    elapsed: $("elapsed"),
    raw: $("raw"),
    matrixCanvas: $("matrixCanvas"),
    matrixBlurToggle: $("matrixBlurToggle"),
    matrixBlurAmount: $("matrixBlurAmount"),
    matrixMeta: $("matrixMeta"),
    matrixScene: $("matrixScene"),
    matrixReason: $("matrixReason"),
    stripTheme: $("stripTheme"),
    stripReason: $("stripReason"),
    swatches: $("swatches"),
    stripHint: $("stripHint"),
    stripMeta: $("stripMeta"),
    speakableBox: $("speakableBox"),
    speakableText: $("speakableText"),
    imageFile: $("imageFile"),
    matrixWidth: $("matrixWidth"),
    matrixHeight: $("matrixHeight"),
    includeRaw: $("includeRaw"),
    downsampleBtn: $("downsampleBtn"),
    downsampleHint: $("downsampleHint"),
    matrixFps: $("matrixFps"),
    matrixDuration: $("matrixDuration"),
    matrixStoreFrames: $("matrixStoreFrames"),
    matrixAnimInstruction: $("matrixAnimInstruction"),
    matrixAnimateBtn: $("matrixAnimateBtn"),
    matrixStopBtn: $("matrixStopBtn"),
    matrixAnimHint: $("matrixAnimHint"),
    matrixAnimError: $("matrixAnimError"),
    stripMode: $("stripMode"),
    stripLedCount: $("stripLedCount"),
    stripBrightness: $("stripBrightness"),
    stripSpeed: $("stripSpeed"),
    stripColors: $("stripColors"),
    stripApplyBtn: $("stripApplyBtn"),
    stripLoadBtn: $("stripLoadBtn"),
    stripPreviewStartBtn: $("stripPreviewStartBtn"),
    stripPreviewStopBtn: $("stripPreviewStopBtn"),
    stripCmdHint: $("stripCmdHint"),
    stripPreview: $("stripPreview"),
    frameLedCount: $("frameLedCount"),
    fetchFrameJsonBtn: $("fetchFrameJsonBtn"),
    fetchFrameRawBtn: $("fetchFrameRawBtn"),
    frameInfo: $("frameInfo"),
    wsStatus: $("wsStatus"),
    wsLog: $("wsLog"),
  };

  const wsLogEntries = [];

  function setMatrixBlur(enabled) {
    if (!els.matrixCanvas) return;
    els.matrixCanvas.classList.toggle("matrix-blur", !!enabled);
  }

  function setMatrixBlurAmount(value) {
    if (!els.matrixCanvas) return;
    const amount = Number(value);
    const clamped = Number.isFinite(amount) ? Math.max(0, Math.min(16, amount)) : 0;
    els.matrixCanvas.style.setProperty("--matrix-blur", `${clamped}px`);
  }

  if (els.matrixBlurToggle) {
    setMatrixBlur(els.matrixBlurToggle.checked);
    els.matrixBlurToggle.addEventListener("change", () => {
      setMatrixBlur(els.matrixBlurToggle.checked);
    });
  }

  if (els.matrixBlurAmount) {
    setMatrixBlurAmount(els.matrixBlurAmount.value);
    els.matrixBlurAmount.addEventListener("input", () => {
      setMatrixBlurAmount(els.matrixBlurAmount.value);
    });
  }

  function setStatus(text, ok=null) {
    els.statusText.textContent = text;
    els.dot.classList.remove("ok", "bad");
    if (ok === true) els.dot.classList.add("ok");
    if (ok === false) els.dot.classList.add("bad");
  }

  function drawMatrix(matrixJson) {
    const canvas = els.matrixCanvas;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!matrixJson || !matrixJson.pixels) {
      els.matrixMeta.textContent = "无数据";
      return;
    }

    const w = matrixJson.width || 16;
    const h = matrixJson.height || 16;
    const pixels = matrixJson.pixels;

    canvas.width = w;
    canvas.height = h;

    // Flip vertically: y=0 in data is bottom, but canvas y=0 is top
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const srcY = h - 1 - y;
        const rgb = (pixels[srcY] && pixels[srcY][x]) ? pixels[srcY][x] : [0,0,0];
        ctx.fillStyle = `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
        ctx.fillRect(x, y, 1, 1);
      }
    }

    els.matrixMeta.textContent = `${w}×${h}`;
  }

  function drawMatrixFromRaw(rawBase64, width, height) {
    if (!rawBase64) return;
    const w = Number(width || 16);
    const h = Number(height || 16);
    const canvas = els.matrixCanvas;
    const ctx = canvas.getContext("2d");

    canvas.width = w;
    canvas.height = h;

    const bytes = Uint8Array.from(atob(rawBase64), (c) => c.charCodeAt(0));
    const imageData = ctx.createImageData(w, h);

    // Flip vertically: data row 0 is bottom, canvas row 0 is top
    for (let y = 0; y < h; y++) {
      const srcY = h - 1 - y;
      for (let x = 0; x < w; x++) {
        const srcIdx = (srcY * w + x) * 3;
        const dstIdx = (y * w + x) * 4;
        imageData.data[dstIdx] = bytes[srcIdx] || 0;
        imageData.data[dstIdx + 1] = bytes[srcIdx + 1] || 0;
        imageData.data[dstIdx + 2] = bytes[srcIdx + 2] || 0;
        imageData.data[dstIdx + 3] = 255;
      }
    }

    ctx.putImageData(imageData, 0, 0);
    els.matrixMeta.textContent = `${w}×${h}`;
  }

  function renderStripFromSelection(finalSelection) {
    els.swatches.innerHTML = "";

    if (!Array.isArray(finalSelection) || finalSelection.length === 0) {
      els.stripHint.textContent = "无灯带颜色数据";
      return;
    }

    for (const c of finalSelection) {
      const rgb = c.rgb || [0,0,0];
      const name = c.name || "(unnamed)";
      const div = document.createElement("div");
      div.className = "swatch";
      div.innerHTML = `
        <div class="chip" style="background: rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})"></div>
        <div style="display:flex; justify-content:space-between; gap:10px;">
          <span style="font-weight:650;">${name}</span>
          <span class="mini">rgb(${rgb.join(",")})</span>
        </div>
      `;
      els.swatches.appendChild(div);
    }

    els.stripHint.textContent = `共 ${finalSelection.length} 个色块（此处用于调试预览）`;
  }

  function renderStripFromRgbList(rgbList) {
    const selection = (Array.isArray(rgbList) ? rgbList : []).map((rgb, idx) => ({
      name: `Color ${idx+1}`,
      rgb: rgb,
    }));
    renderStripFromSelection(selection);
  }

  const stripPreviewState = {
    timer: null,
    loading: false,
    ledCount: 60,
  };

  function ensureStripPreview(ledCount) {
    const raw = Number(ledCount);
    const normalized = Number.isFinite(raw) ? Math.max(1, Math.min(2000, Math.round(raw))) : 60;
    stripPreviewState.ledCount = normalized;
    if (els.stripLedCount && String(els.stripLedCount.value || "") !== String(normalized)) {
      els.stripLedCount.value = String(normalized);
    }
    if (els.frameLedCount && String(els.frameLedCount.value || "") !== String(normalized)) {
      els.frameLedCount.value = String(normalized);
    }
    if (els.stripMeta) {
      els.stripMeta.textContent = `${normalized} LEDs`;
    }
  }

  function updateStripPreviewFrame(frame) {
      if (!Array.isArray(frame) || !els.stripPreview) return;
      const count = frame.length;
      if (count === 0) return;

      // Construct CSS linear-gradient
      // Format: linear-gradient(90deg, rgb(r,g,b) 0%, rgb(r,g,b) 1.6%, ...)
      const stops = [];
      for (let i = 0; i < count; i++) {
          const rgb = frame[i] || [0, 0, 0];
          const pct = (i / (count - 1)) * 100;
          stops.push(`rgb(${rgb[0]},${rgb[1]},${rgb[2]}) ${pct.toFixed(2)}%`);
      }
      
      const gradient = `linear-gradient(90deg, ${stops.join(", ")})`;
      els.stripPreview.style.background = gradient;
      
      // Update meta info occasionally or on change
      if (els.stripMeta) {
          els.stripMeta.textContent = `${count} LEDs`;
      }
  }

  async function fetchStripPreviewFrame() {
    if (stripPreviewState.loading) return;
    const ledCount = Number(els.stripLedCount.value || 60);

    stripPreviewState.loading = true;
    try {
      const frame = await getJson(`/api/data/strip/frame/json?led_count=${ledCount}`);
      updateStripPreviewFrame(frame);
    } catch (e) {
      // Silent fail; preview is best-effort.
    } finally {
      stripPreviewState.loading = false;
    }
  }

  function startStripPreview() {
    if (stripPreviewState.timer) return;
    fetchStripPreviewFrame();
    stripPreviewState.timer = setInterval(fetchStripPreviewFrame, 100);
  }

  function stopStripPreview() {
    if (!stripPreviewState.timer) return;
    clearInterval(stripPreviewState.timer);
    stripPreviewState.timer = null;
  }

  function safeStr(v) {
    if (v === null || v === undefined) return "-";
    if (typeof v === "string" && v.trim() === "") return "-";
    return String(v);
  }

  function setWsStatus(text) {
    if (els.wsStatus) {
      els.wsStatus.textContent = text;
    }
  }

  function appendWsLog(type, payload) {
    if (!els.wsLog) return;
    const ts = new Date().toLocaleTimeString();
    let summary = "";
    if (payload && typeof payload === "object") {
      summary = JSON.stringify(payload);
      if (summary.length > 400) summary = summary.slice(0, 400) + "…";
    }
    const line = summary ? `[${ts}] ${type} ${summary}` : `[${ts}] ${type}`;
    wsLogEntries.unshift(line);
    wsLogEntries.splice(8);
    els.wsLog.textContent = wsLogEntries.join("\n");
  }

  function parseRgbInput(value) {
    const entries = (value || "").split(";").map((item) => item.trim()).filter(Boolean);
    const colors = [];

    for (const entry of entries) {
      const parts = entry.split(",").map((item) => item.trim()).filter(Boolean);
      if (parts.length !== 3) continue;
      const nums = parts.map((item) => Number(item));
      if (nums.some((n) => Number.isNaN(n))) continue;
      const rgb = nums.map((n) => Math.min(255, Math.max(0, Math.round(n))));
      colors.push({ rgb });
    }

    return colors;
  }

  function formatColorsInput(colors) {
    if (!Array.isArray(colors)) return "";
    return colors
      .map((c) => (Array.isArray(c) ? c : (c.rgb || [])).join(","))
      .filter(Boolean)
      .join(";");
  }

  function updateStripCommandForm(envelope) {
    if (!envelope || !envelope.command) return;
    const cmd = envelope.command || {};
    if (cmd.mode) els.stripMode.value = cmd.mode;
    if (cmd.led_count !== undefined) {
      els.stripLedCount.value = cmd.led_count;
      els.frameLedCount.value = cmd.led_count;
    }
    if (cmd.brightness !== undefined) els.stripBrightness.value = cmd.brightness;
    if (cmd.speed !== undefined) els.stripSpeed.value = cmd.speed;
    if (cmd.colors) els.stripColors.value = formatColorsInput(cmd.colors);
    ensureStripPreview(cmd.led_count || 60);
  }

  async function postJson(url, body) {
    const r = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const text = await r.text();
    let data;
    try { data = JSON.parse(text); } catch { data = { raw: text }; }
    if (!r.ok) throw new Error(`${r.status} ${r.statusText}: ${text}`);
    return data;
  }

  async function getJson(url) {
    const r = await fetch(url);
    const text = await r.text();
    let data;
    try { data = JSON.parse(text); } catch { data = { raw: text }; }
    if (!r.ok) throw new Error(`${r.status} ${r.statusText}: ${text}`);
    return data;
  }

  async function downsampleImage() {
    const file = els.imageFile.files && els.imageFile.files[0];
    if (!file) {
      setStatus("请选择图片", false);
      return;
    }

    const width = Number(els.matrixWidth.value || 16);
    const height = Number(els.matrixHeight.value || 16);
    const includeRaw = els.includeRaw.checked;

    const form = new FormData();
    form.append("file", file);

    setStatus("上传中…", null);
    els.elapsed.textContent = "-";

    const t0 = performance.now();
    try {
      const url = `/api/matrix/downsample?width=${width}&height=${height}&include_raw=${includeRaw ? "true" : "false"}`;
      const r = await fetch(url, { method: "POST", body: form });
      const text = await r.text();
      let data;
      try { data = JSON.parse(text); } catch { data = { raw: text }; }
      if (!r.ok) throw new Error(`${r.status} ${r.statusText}: ${text}`);

      els.raw.textContent = JSON.stringify(data, null, 2);
      drawMatrix(data.json || null);
      els.matrixScene.textContent = "(上传图片下采样)";
      els.matrixReason.textContent = "-";
      setStatus("下采样完成", true);
    } catch (e) {
      setStatus(`失败：${e.message}`, false);
      els.raw.textContent = JSON.stringify({ error: e.message }, null, 2);
    } finally {
      const t1 = performance.now();
      els.elapsed.textContent = `${Math.round(t1 - t0)} ms`;
    }
  }

  async function animateMatrix() {
    const customInstruction = els.matrixAnimInstruction.value.trim();
    const instruction = customInstruction || els.instruction.value.trim();
    if (!instruction) {
      setStatus("请输入指令", false);
      return;
    }

    const width = Number(els.matrixWidth.value || 16);
    const height = Number(els.matrixHeight.value || 16);
    const fps = Number(els.matrixFps.value || 12);
    const duration = Number(els.matrixDuration.value || 30);
    const storeFrames = !!els.matrixStoreFrames.checked;

    const payload = {
      instruction,
      width,
      height,
      fps,
      duration_s: duration,
      store_frames: storeFrames,
    };

    setStatus("生成动画脚本…", null);
    els.elapsed.textContent = "-";
    if (els.matrixAnimError) {
      els.matrixAnimError.textContent = "-";
    }
    const t0 = performance.now();

    try {
      const res = await postJson("/api/matrix/animate", payload);
      els.raw.textContent = JSON.stringify(res, null, 2);
      els.matrixScene.textContent = safeStr(res.summary || instruction);
      els.matrixReason.textContent = "动画运行中…";
      els.matrixAnimHint.textContent = "动画已启动，等待帧推送";
      setStatus("动画已启动", true);
    } catch (e) {
      setStatus(`失败：${e.message}`, false);
      els.raw.textContent = JSON.stringify({ error: e.message }, null, 2);
      els.matrixAnimHint.textContent = "动画启动失败";
    } finally {
      const t1 = performance.now();
      els.elapsed.textContent = `${Math.round(t1 - t0)} ms`;
    }
  }

  async function stopMatrixAnimation() {
    setStatus("停止动画…", null);
    els.elapsed.textContent = "-";
    const t0 = performance.now();

    try {
      const res = await postJson("/api/matrix/animate/stop", {});
      els.raw.textContent = JSON.stringify(res, null, 2);
      els.matrixReason.textContent = "动画已停止";
      els.matrixAnimHint.textContent = "动画已停止";
      setStatus("动画已停止", true);
    } catch (e) {
      setStatus(`失败：${e.message}`, false);
      els.raw.textContent = JSON.stringify({ error: e.message }, null, 2);
    } finally {
      const t1 = performance.now();
      els.elapsed.textContent = `${Math.round(t1 - t0)} ms`;
    }
  }

  async function applyStripCommand() {
    const colorInput = els.stripColors.value;
    const colors = parseRgbInput(colorInput);
    if (colorInput.trim() && colors.length === 0) {
      setStatus("颜色格式不正确", false);
      els.stripCmdHint.textContent = "颜色格式示例：255,140,60;255,160,190";
      return;
    }

    const payload = {
      mode: els.stripMode.value || "static",
      colors: colors,
      brightness: Number(els.stripBrightness.value || 1),
      speed: Number(els.stripSpeed.value || 2),
      led_count: Number(els.stripLedCount.value || 60),
    };

    setStatus("下发灯带指令…", null);
    els.elapsed.textContent = "-";
    const t0 = performance.now();

    try {
      const res = await postJson("/api/app/strip/command", payload);
      els.raw.textContent = JSON.stringify(res, null, 2);
      updateStripCommandForm(res);
      const previewColors = res.command && res.command.colors ? res.command.colors : colors;
      if (previewColors.length) {
        renderStripFromSelection(previewColors);
        els.stripHint.textContent = `指令已更新（${previewColors.length} 色）`;
      } else {
        els.stripHint.textContent = "指令已更新";
      }
      els.stripCmdHint.textContent = "指令已写入";
      ensureStripPreview(res.command ? res.command.led_count : payload.led_count);
      startStripPreview();
      setStatus("灯带指令已更新", true);
    } catch (e) {
      setStatus(`失败：${e.message}`, false);
      els.raw.textContent = JSON.stringify({ error: e.message }, null, 2);
    } finally {
      const t1 = performance.now();
      els.elapsed.textContent = `${Math.round(t1 - t0)} ms`;
    }
  }

  async function loadStripCommand() {
    setStatus("读取灯带指令…", null);
    els.elapsed.textContent = "-";
    const t0 = performance.now();

    try {
      const res = await getJson("/api/data/strip/command");
      updateStripCommandForm(res);
      if (res.command && res.command.colors) {
        renderStripFromSelection(res.command.colors);
        els.stripHint.textContent = `当前指令（${res.command.colors.length} 色）`;
      }
      ensureStripPreview(res.command ? res.command.led_count : 60);
      startStripPreview();
      els.stripCmdHint.textContent = "已读取当前指令";
      els.raw.textContent = JSON.stringify(res, null, 2);
      setStatus("灯带指令已加载", true);
    } catch (e) {
      setStatus(`失败：${e.message}`, false);
      els.raw.textContent = JSON.stringify({ error: e.message }, null, 2);
    } finally {
      const t1 = performance.now();
      els.elapsed.textContent = `${Math.round(t1 - t0)} ms`;
    }
  }

  async function fetchFrameJson() {
    const ledCount = Number(els.frameLedCount.value || 60);
    setStatus("读取 JSON 帧…", null);
    els.elapsed.textContent = "-";
    const t0 = performance.now();

    try {
      const frame = await getJson(`/api/data/strip/frame/json?led_count=${ledCount}`);
      const count = Array.isArray(frame) ? frame.length : 0;
      const preview = Array.isArray(frame) ? frame.slice(0, 24) : [];
      if (preview.length) {
        renderStripFromRgbList(preview);
        els.stripHint.textContent = `帧预览前 ${preview.length} 颗 LED`;
      }

      updateStripPreviewFrame(frame);
      els.frameInfo.textContent = `JSON 帧：${count} 颗 LED`;

      els.raw.textContent = JSON.stringify({ led_count: ledCount, preview: preview, total: count }, null, 2);
      setStatus("JSON 帧已读取", true);
    } catch (e) {
      setStatus(`失败：${e.message}`, false);
      els.raw.textContent = JSON.stringify({ error: e.message }, null, 2);
    } finally {
      const t1 = performance.now();
      els.elapsed.textContent = `${Math.round(t1 - t0)} ms`;
    }
  }

  async function fetchFrameRaw() {
    const ledCount = Number(els.frameLedCount.value || 60);
    setStatus("读取 RAW 帧…", null);
    els.elapsed.textContent = "-";
    const t0 = performance.now();

    try {
      const r = await fetch(`/api/data/strip/frame/raw?led_count=${ledCount}`);
      if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
      const buffer = await r.arrayBuffer();
      els.frameInfo.textContent = `RAW 帧：${buffer.byteLength} bytes`;
      els.raw.textContent = JSON.stringify({ led_count: ledCount, raw_bytes: buffer.byteLength }, null, 2);
      setStatus("RAW 帧已读取", true);
    } catch (e) {
      setStatus(`失败：${e.message}`, false);
      els.raw.textContent = JSON.stringify({ error: e.message }, null, 2);
    } finally {
      const t1 = performance.now();
      els.elapsed.textContent = `${Math.round(t1 - t0)} ms`;
    }
  }


  function renderGenerate(res) {
    els.raw.textContent = JSON.stringify(res, null, 2);

    // Prefer top-level speakable_reason, fall back to nested
    let speak = res.speakable_reason;
    if (!speak && res.matrix) speak = res.matrix.speakable_reason;
    if (!speak && res.strip) speak = res.strip.speakable_reason;
    
    if (speak) {
        els.speakableBox.style.display = "block";
        els.speakableText.textContent = speak;
    } else {
        els.speakableBox.style.display = "none";
    }

    // Adapt to different response structures (VoiceSubmit vs Generate)
    // Structure 1 (VoiceSubmit): res.matrix / res.strip (direct plans)
    // Structure 2 (Generate/WS): res.data.matrix / res.data.strip
    
    const m = res.matrix || (res.data ? res.data.matrix : {}) || {};
    const s = res.strip || (res.data ? res.data.strip : {}) || {};

    // Matrix
    els.matrixScene.textContent = safeStr(m.scene_prompt || m.prompt_used);
    els.matrixReason.textContent = safeStr(m.reason);
    
    // Explicitly handle "Generating" state for matrix
    // If we have a prompt but NO json data, it means it's still generating in background.
    if (m.scene_prompt && !m.json) {
        els.matrixMeta.textContent = "生成中…";
        // Clear canvas or show placeholder
        const canvas = els.matrixCanvas;
        const ctx = canvas.getContext("2d");
        ctx.fillStyle = "#222";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
    } else {
        drawMatrix(m.json || null);
    }

    // Strip
    els.stripTheme.textContent = safeStr(s.theme);
    els.stripReason.textContent = safeStr(s.reason);
    
    // Support both 'colors' (Plan) and 'final_selection' (Exec)
    const colors = s.colors || s.final_selection || [];
    renderStripFromSelection(colors);
  }

  async function run() {
    const instruction = els.instruction.value.trim();

    if (!instruction) {
      setStatus("请输入指令", false);
      return;
    }

    setStatus("请求中…", null);
    els.elapsed.textContent = "-";

    const t0 = performance.now();
    try {
      // Use the new VOICE Submit API (Parallel execution)
      const res = await postJson("/api/voice/submit", { instruction });
      renderGenerate(res);
      setStatus("已规划 (后台执行中)", true);
    } catch (e) {
      setStatus(`失败：${e.message}`, false);
      els.raw.textContent = JSON.stringify({ error: e.message }, null, 2);
    } finally {
      const t1 = performance.now();
      els.elapsed.textContent = `${Math.round(t1 - t0)} ms`;
    }
  }

  async function loadCurrent() {
    setStatus("读取当前数据…", null);
    els.elapsed.textContent = "-";

    const t0 = performance.now();
    try {
      const [matrix, strip, stripCommand] = await Promise.all([
        getJson("/api/data/matrix/json"),
        getJson("/api/data/strip"),
        getJson("/api/data/strip/command"),
      ]);

      els.raw.textContent = JSON.stringify({ matrix, strip, stripCommand }, null, 2);

      drawMatrix(matrix);
      els.matrixScene.textContent = "(当前落盘数据)";
      els.matrixReason.textContent = "-";

      if (Array.isArray(strip)) {
        els.stripTheme.textContent = "(当前落盘数据)";
        els.stripReason.textContent = "-";
        renderStripFromRgbList(strip);
      } else {
        els.stripTheme.textContent = "-";
        els.stripReason.textContent = "-";
        els.swatches.innerHTML = "";
      }

      if (stripCommand) {
        updateStripCommandForm(stripCommand);
        els.stripCmdHint.textContent = "已同步当前指令";
      }

      startStripPreview();
      setStatus("完成", true);
    } catch (e) {
      setStatus(`失败：${e.message}`, false);
      els.raw.textContent = JSON.stringify({ error: e.message }, null, 2);
    } finally {
      const t1 = performance.now();
      els.elapsed.textContent = `${Math.round(t1 - t0)} ms`;
    }
  }

  function clearAll() {
    els.instruction.value = "";
    els.raw.textContent = "{}";
    els.matrixScene.textContent = "-";
    els.matrixReason.textContent = "-";
    els.matrixMeta.textContent = "-";
    els.stripTheme.textContent = "-";
    els.stripReason.textContent = "-";
    els.swatches.innerHTML = "";
    els.stripHint.textContent = "-";
    els.speakableBox.style.display = "none";
    els.imageFile.value = "";
    els.matrixWidth.value = "16";
    els.matrixHeight.value = "16";
    els.includeRaw.checked = true;
    els.matrixFps.value = "12";
    els.matrixDuration.value = "30";
    els.matrixStoreFrames.checked = true;
    els.matrixAnimInstruction.value = "";
    els.matrixAnimHint.textContent = "使用当前矩阵宽高；持续时间填 0 可循环播放（需手动停止）。";
    if (els.matrixAnimError) {
      els.matrixAnimError.textContent = "-";
    }
    if (els.matrixBlurToggle) {
      els.matrixBlurToggle.checked = true;
      setMatrixBlur(true);
    }
    if (els.matrixBlurAmount) {
      els.matrixBlurAmount.value = "8";
      setMatrixBlurAmount(els.matrixBlurAmount.value);
    }
    els.stripMode.value = "static";
    els.stripLedCount.value = "60";
    els.stripBrightness.value = "1";
    els.stripSpeed.value = "2";
    els.stripColors.value = "";
    els.frameLedCount.value = "60";
    els.frameInfo.textContent = "-";
    if (els.wsLog) {
      wsLogEntries.length = 0;
      els.wsLog.textContent = "-";
    }
    stopStripPreview();
    drawMatrix(null);
    setStatus("就绪", null);
    els.elapsed.textContent = "-";
  }

  els.runBtn.addEventListener("click", run);
  els.clearBtn.addEventListener("click", clearAll);
  els.loadCurrentBtn.addEventListener("click", loadCurrent);
  els.downsampleBtn.addEventListener("click", downsampleImage);
  els.matrixAnimateBtn.addEventListener("click", animateMatrix);
  els.matrixStopBtn.addEventListener("click", stopMatrixAnimation);
  els.stripApplyBtn.addEventListener("click", applyStripCommand);
  els.stripLoadBtn.addEventListener("click", loadStripCommand);
  els.stripPreviewStartBtn.addEventListener("click", startStripPreview);
  els.stripPreviewStopBtn.addEventListener("click", stopStripPreview);
  els.fetchFrameJsonBtn.addEventListener("click", fetchFrameJson);
  els.fetchFrameRawBtn.addEventListener("click", fetchFrameRaw);

  window.addEventListener("resize", () => {
    // No-op for now (CSS handles resizing)
  });

  function connectWs() {
    try {
      const proto = (location.protocol === "https:") ? "wss" : "ws";
      const ws = new WebSocket(`${proto}://${location.host}/ws`);

      ws.onopen = () => {
        setStatus("WebSocket 已连接", true);
        setWsStatus("已连接");
        appendWsLog("connected");
        // Keepalive ping every 20s
        setInterval(() => {
          try { ws.send("ping"); } catch (_) {}
        }, 20000);
      };

      ws.onmessage = (evt) => {
        let msg;
        try { msg = JSON.parse(evt.data); } catch { return; }
        if (!msg || !msg.type) return;

        // Live preview: generate results
        if (msg.type === "generate") {
          renderGenerate(msg.payload);
          setStatus("收到推送：generate", true);
          appendWsLog("generate", msg.payload);
        }

        if (msg.type === "matrix_update") {
          const p = msg.payload || {};
          drawMatrix(p.json || null);
          els.matrixScene.textContent = "(矩阵已更新)";
          els.matrixReason.textContent = "-";
          appendWsLog("matrix_update", msg.payload);
        }

        if (msg.type === "matrix_animation_start") {
          const p = msg.payload || {};
          els.matrixScene.textContent = safeStr(p.summary || "矩阵动画");
          els.matrixReason.textContent = "动画运行中…";
          els.matrixAnimHint.textContent = `动画已启动 (${p.width || 16}×${p.height || 16}, ${p.fps || 0} fps)`;
          if (els.matrixAnimError) {
            els.matrixAnimError.textContent = "-";
          }
          appendWsLog("matrix_animation_start", msg.payload);
        }

        if (msg.type === "matrix_frame") {
          const p = msg.payload || {};
          drawMatrixFromRaw(p.data, p.width, p.height);
          const frameIndex = p.frame_index ?? "-";
          els.matrixReason.textContent = `帧 ${frameIndex}`;
          els.matrixAnimHint.textContent = `动画进行中（帧 ${frameIndex}）`;
          appendWsLog("matrix_frame", { frame_index: p.frame_index });
        }

        if (msg.type === "matrix_animation_fallback") {
          const p = msg.payload || {};
          const reason = p.reason || "未知错误";
          const missing = Array.isArray(p.missing_dependencies) ? p.missing_dependencies : [];
          const failedCode = p.failed_code || "";
          const missingText = missing.length ? `缺少依赖 ${missing.join(", ")}` : reason;
          els.matrixAnimHint.textContent = `已切换兜底动画（原因：${missingText}）`;
          if (els.matrixAnimError) {
            els.matrixAnimError.textContent = `动画错误详情：${reason}`;
          }
          appendWsLog("matrix_animation_fallback", { reason, missing_dependencies: missing });
          if (failedCode) {
            console.error("=== 失败的动画脚本 ===\n" + failedCode);
            appendWsLog("failed_animation_code", { code_preview: failedCode.slice(0, 500) + (failedCode.length > 500 ? "..." : "") });
          }
        }

        if (msg.type === "matrix_animation_complete") {
          const p = msg.payload || {};
          els.matrixReason.textContent = p.status === "completed" ? "动画完成" : "动画结束";
          let hint = "动画已完成";
          const detail = p.error_detail || {};
          const missing = Array.isArray(detail.missing_dependencies) ? detail.missing_dependencies : [];
          const detailMessage = detail.message ? String(detail.message) : "";
          const reason = missing.length
            ? `缺少依赖 ${missing.join(", ")}`
            : (detailMessage || p.error || "未知错误");
          if (p.fallback_used) {
            hint = `动画已完成（已降级兜底，原因：${reason}）`;
          } else if (missing.length) {
            hint = `动画出错：缺少依赖 ${missing.join(", ")}`;
          } else if (detailMessage) {
            hint = `动画出错：${detailMessage}`;
          } else if (p.error) {
            hint = `动画出错：${p.error}`;
          }
          els.matrixAnimHint.textContent = hint;
          if (els.matrixAnimError) {
            if (p.fallback_used || missing.length || detailMessage || p.error) {
              els.matrixAnimError.textContent = `动画错误详情：${reason}`;
            }
          }
          if (p.fallback_used) {
            appendWsLog("matrix_animation_fallback", { reason, missing_dependencies: missing });
          }
          appendWsLog("matrix_animation_complete", msg.payload);
        }

        if (msg.type === "strip_command_update") {

          updateStripCommandForm(msg.payload);
          startStripPreview();
          appendWsLog("strip_command_update", msg.payload);
        }

        if (msg.type !== "generate" && msg.type !== "matrix_update" && msg.type !== "strip_command_update") {
          appendWsLog(msg.type, msg.payload);
        }
      };

      ws.onclose = () => {
        setStatus("WebSocket 已断开（重连中）", false);
        setWsStatus("断开，重连中…");
        appendWsLog("disconnected");
        setTimeout(connectWs, 1200);
      };

      ws.onerror = () => {
        // onclose will handle reconnect
      };
    } catch (_) {
      setStatus("WebSocket 不可用", false);
      setWsStatus("不可用");
      appendWsLog("error");
    }
  }

  setStatus("就绪", null);
  setWsStatus("连接中…");
  drawMatrix(null);
  ensureStripPreview(Number(els.stripLedCount.value || 60));
  startStripPreview();
  connectWs();
</script>
</body>
</html>
"""


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
            frame = strip_effects.render_strip_frame(
                cached_cmd,
                now_s=time.time(),
                led_count=effective_led_count,
            )
            raw, _ = _encode_strip_frame(frame, encoding)
            await websocket.send_bytes(raw)
            await asyncio.sleep(interval)
    except WebSocketDisconnect:
        return
    except Exception:
        return


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
            frame = strip_effects.render_strip_frame(
                cached_cmd,
                now_s=time.time(),
                led_count=led_count,
            )
            raw, meta = _encode_strip_frame(frame, normalized_encoding)

            payload = {
                "ts_ms": int(time.time() * 1000),
                "frame_index": frame_index,
                "encoding": meta["encoding"],
                "bit_depth": meta["bit_depth"],
                "bytes_per_led": meta["bytes_per_led"],
                "led_count": led_count,
                "fps": fps,
                "transport": "base64",
                "data": base64.b64encode(raw).decode("utf-8"),
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
