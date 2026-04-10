from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


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

