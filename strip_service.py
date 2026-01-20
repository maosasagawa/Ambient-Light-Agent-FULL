import json
import math
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import requests

# Configuration (Shared with main, could be moved to config)
API_KEY = os.environ.get("AIHUBMIX_API_KEY", "")
AIHUBMIX_BASE_URL = "https://aihubmix.com"
MODEL_ID_CHAT = "gpt-5-mini"

UNIFIED_API_HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}",
}

# Optional: plain-text knowledge base for strip color generation.
# Format: one entry per line.
STRIP_KB_FILE = os.environ.get(
    "STRIP_KB_FILE", os.path.join(os.path.dirname(__file__), "strip_kb.txt")
)

# Data Persistence
DATA_FILE = "latest_strip_data.json"
STRIP_COMMAND_FILE = "latest_strip_command.json"


@dataclass(frozen=True)
class _KbIndexCache:
    mtime: float
    # List of (token_set, line_text)
    index: List[Tuple[Set[str], str]]


_KB_INDEX_CACHE: Optional[_KbIndexCache] = None


# Lightweight synonym groups used for KB retrieval.
# Implementation goal: low-latency, no external dependencies.
_SYNONYM_GROUPS: List[Set[str]] = [
    {"疲劳", "困", "困倦", "瞌睡", "打瞌睡", "疲惫", "sleepy", "tired", "drowsy"},
    {"警示", "警告", "提神", "醒脑", "注意", "alert", "warning"},
    {"海边", "海滩", "海洋", "大海", "海", "ocean", "sea", "beach"},
    {"森林", "树林", "林地", "丛林", "forest", "jungle"},
    {"放松", "舒缓", "治愈", "relax", "calm", "cozy"},
    {"专注", "学习", "工作", "focus", "study", "work"},
    {"睡眠", "冥想", "静心", "助眠", "sleep", "meditation"},
    {"阴雨", "下雨", "雨天", "rain", "rainy"},
]


def color_distance(rgb1: List[int], rgb2: List[int]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(rgb1, rgb2)))


def _clamp_int(value: Any, *, low: int, high: int) -> int:
    try:
        n = int(value)
    except Exception:
        return low
    return max(low, min(high, n))


def _normalize_rgb(value: Any) -> List[int]:
    if not isinstance(value, list) or len(value) != 3:
        return [0, 0, 0]
    return [
        _clamp_int(value[0], low=0, high=255),
        _clamp_int(value[1], low=0, high=255),
        _clamp_int(value[2], low=0, high=255),
    ]


def get_validation_status(rgb: List[int]) -> Dict[str, Any]:
    """Validate a color for LED strip usage.

    We keep validation permissive to allow low-energy themes (sleep/meditation),
    while still avoiding colors that are too dark to be visible.
    """

    max_val = max(rgb)
    if max_val < 40:
        return {"valid": False, "reason": "亮度不足"}
    return {"valid": True, "reason": None}


def save_strip_data(data: List[List[int]]) -> None:
    try:
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    except Exception as e:
        print(f"Failed to save strip data: {e}")


def load_strip_data() -> List[List[int]]:
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                parsed = json.load(f)
                if isinstance(parsed, list):
                    return [
                        _normalize_rgb(rgb)
                        for rgb in parsed
                        if isinstance(rgb, list) and len(rgb) == 3
                    ]
        except Exception:
            pass
    return [[0, 170, 255]]  # Default Blue


def save_strip_command(command: Dict[str, Any]) -> None:
    payload = dict(command or {})
    payload.setdefault("updated_at_ms", int(time.time() * 1000))
    try:
        with open(STRIP_COMMAND_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
    except Exception as e:
        print(f"Failed to save strip command: {e}")


def load_strip_command() -> Dict[str, Any]:
    if os.path.exists(STRIP_COMMAND_FILE):
        try:
            with open(STRIP_COMMAND_FILE, "r", encoding="utf-8") as f:
                parsed = json.load(f)
                if isinstance(parsed, dict):
                    return parsed
        except Exception:
            pass

    # Default command is compatible with legacy strip data.
    colors = load_strip_data()
    return {
        "render_target": "cloud",
        "mode": "static",
        "colors": colors,
        "brightness": 1.0,
        "speed": 2.0,
        "led_count": 60,
    }


def _tokenize_for_retrieval(text: str) -> Set[str]:
    """Tokenize text for lightweight retrieval.

    - English tokens: [a-z0-9_]+
    - CJK: unigrams + bigrams for better matching without a tokenizer.
    """

    lowered = (text or "").lower()
    word_tokens = re.findall(r"[a-z0-9_]+", lowered)

    cjk_chars = re.findall(r"[\u4e00-\u9fff]", text)
    bigrams: List[str] = []
    for i in range(len(cjk_chars) - 1):
        bigrams.append(cjk_chars[i] + cjk_chars[i + 1])

    return set(word_tokens) | set(cjk_chars) | set(bigrams)


def _expand_with_synonyms(tokens: Set[str]) -> Set[str]:
    expanded = set(tokens)
    for group in _SYNONYM_GROUPS:
        if expanded & group:
            expanded |= group
    return expanded


def _load_kb_index_cached(file_path: str, max_lines: int = 2000) -> List[Tuple[Set[str], str]]:
    global _KB_INDEX_CACHE

    if not os.path.exists(file_path):
        _KB_INDEX_CACHE = _KbIndexCache(mtime=-1, index=[])
        return []

    try:
        mtime = os.path.getmtime(file_path)
    except OSError:
        _KB_INDEX_CACHE = _KbIndexCache(mtime=-1, index=[])
        return []

    if _KB_INDEX_CACHE is not None and _KB_INDEX_CACHE.mtime == mtime:
        return _KB_INDEX_CACHE.index

    lines: List[str] = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for ln in f:
                s = ln.strip()
                if not s:
                    continue
                if s.startswith("#"):
                    continue
                lines.append(s)
                if len(lines) >= max_lines:
                    break
    except Exception:
        lines = []

    index: List[Tuple[Set[str], str]] = [(_tokenize_for_retrieval(line), line) for line in lines]
    _KB_INDEX_CACHE = _KbIndexCache(mtime=mtime, index=index)
    return index


def get_strip_kb_context(user_input: str, *, top_k: int = 6) -> str:
    """Return a short KB context block for prompt injection.

    Notes:
    - KB format is one line per entry.
    - Retrieval is token-overlap with synonym expansion.
    - Returns a compact block to minimize token usage and latency.
    """

    query_tokens = _expand_with_synonyms(_tokenize_for_retrieval(user_input))
    if not query_tokens:
        return ""

    index = _load_kb_index_cached(STRIP_KB_FILE)
    if not index:
        return ""

    scored: List[Tuple[int, str]] = []
    for tokens, line in index:
        overlap = len(query_tokens & tokens)
        if overlap > 0:
            scored.append((overlap, line))

    scored.sort(key=lambda x: x[0], reverse=True)
    chosen = [line for _, line in scored[: max(0, int(top_k))]]
    if not chosen:
        return ""

    joined = "\n".join(f"- {line}" for line in chosen)
    return (
        "\n\n# 补充资料（本地知识库，每行一条，供参考）\n"
        "以下是从本地知识库召回的若干条目，用于补充上下文；如与用户意图冲突，以用户意图为准。\n"
        f"{joined}\n"
    )


_STRIP_PROMPT_TEMPLATE = """
# 角色和目标
您是一名富有创造力的灯光设计师和色彩理论专家。您的目标是根据用户的情绪或场景描述，为智能环境灯生成一系列颜色候选。

# 选择规则
1. 安全至上：任何涉及行车安全或用户健康（如疲劳、车速过快）的信号，应无条件获得最高优先级。如果用户输入与此类安全问题相关，优先生成警示性或能提升注意力的颜色方案，并在`reason`字段中明确说明这是出于安全考虑。
2. 用户意图优先：用户主动表达的指令或明确的交互操作，其优先级高于所有系统推断的潜在状态。
3. 优先级排序：不同模态信息应有预设的优先级层级，在高优先级模态与低优先级模态冲突时，高优先级模态胜出。规则顺序为：安全至上 > 用户意图优先 > 系统推断。

# 输出格式
您必须仅以有效的JSON对象形式回复，不要包含任何Markdown格式或额外文本。JSON对象必须严格遵循以下结构：
{
  "theme": "一个对用户请求的简短中文概括",
  "color_count_suggestion": <一个数字，1、2或3>,
  "candidate_colors": [
    {"name": "描述性颜色名称1", "rgb": [<R>, <G>, <B>]},
    {"name": "描述性颜色名称2", "rgb": [<R>, <G>, <B>]},
    {"name": "描述性颜色名称3", "rgb": [<R>, <G>, <B>]},
    {"name": "描述性颜色名称4", "rgb": [<R>, <G>, <B>]},
    {"name": "描述性颜色名称5", "rgb": [<R>, <G>, <B>]}
  ],
  "reason": "一段简短的中文解释，说明为什么这个调色板和颜色数量适合该主题"
}

# 配色规则
1. 鲜艳和饱和度：通常情况下，颜色应鲜艳且饱和。但在特定低能量主题（如'睡眠'、'阴雨'、'冥想'）下，应适当降低饱和度，以匹配情绪
2. 明亮：颜色应具有光感。RGB值中最大的分量通常应该较高。若需表达暗色，请尽量保持色相但降低整体亮度逻辑。
3.色彩和谐与多样化：候选颜色列表中的颜色组合必须在视觉上和谐。如果是单色主题，提供同色系的深浅变化；如果是多色主题，确保颜色之间具有互补或对比的美感，避免视觉杂乱，也不宜过于相近。
4. RGB值：R、G、B值必须是0到255之间的整数。
5. 主题优先（重要）：在输出颜色列表中的第一个颜色必须是用户主题中最具代表性和核心的颜色。例如，对于“海边”或“海洋”，第一个候选颜色必须是蓝色（水），而不是绿色（棕榈树）。对于“森林”，必须是绿色。这确保了核心氛围的捕捉。

{kb_context}

# 用户输入
{user_input}
""".strip()


def _select_final_colors(candidates: List[Dict[str, Any]], *, count: int) -> List[Dict[str, Any]]:
    if not candidates:
        return []

    count = max(1, min(3, int(count)))

    valid_candidates: List[Dict[str, Any]] = []
    for c in candidates:
        rgb = _normalize_rgb(c.get("rgb"))
        status = get_validation_status(rgb)
        if status.get("valid") is True:
            name = str(c.get("name") or "(unnamed)")
            valid_candidates.append({"name": name, "rgb": rgb})

    if not valid_candidates:
        return []

    # Keep the first candidate as the "core" color (per prompt rule #5).
    final_selection: List[Dict[str, Any]] = [valid_candidates[0]]

    # For additional colors, prefer diversity by max-min distance.
    available = valid_candidates[1:]
    while len(final_selection) < count and available:
        best_idx = -1
        best_score = -1.0
        for idx, candidate in enumerate(available):
            score = min(color_distance(candidate["rgb"], s["rgb"]) for s in final_selection)
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx < 0:
            break

        # Threshold keeps colors from being too similar.
        if best_score < 40:
            break

        final_selection.append(available.pop(best_idx))

    return final_selection


def generate_strip_colors(user_input: str) -> Dict[str, Any]:
    """Generate strip colors from user input using an LLM.

    Contract:
    - Returns `theme`, `reason`, `final_selection`.
    - Persists `latest_strip_data.json` as `[[R,G,B], ...]`.

    The model is prompted to output:
    - theme (str)
    - color_count_suggestion (1..3)
    - candidate_colors (list of colors)
    - reason (str)

    We keep parsing tolerant to allow partial/short outputs.
    """

    text = (user_input or "").strip().replace("\n", " ")
    if not text:
        return {
            "theme": "默认",
            "reason": "空输入，使用默认值",
            "final_selection": [{"name": "Default Blue", "rgb": [0, 170, 255]}],
        }

    kb_context = get_strip_kb_context(text)
    prompt = _STRIP_PROMPT_TEMPLATE.format(user_input=text, kb_context=kb_context)

    payload = {
        "model": MODEL_ID_CHAT,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
        "temperature": 0.7,
    }

    try:
        response = requests.post(
            f"{AIHUBMIX_BASE_URL}/v1/chat/completions",
            headers=UNIFIED_API_HEADERS,
            json=payload,
            timeout=20,
        )
        response.raise_for_status()
        result_json = response.json()
        content = result_json["choices"][0]["message"]["content"]
        llm_data = json.loads(content)
    except Exception as e:
        print(f"LLM generation failed: {e}")
        return {
            "theme": "默认",
            "reason": "生成失败，使用默认值",
            "final_selection": [{"name": "Default Blue", "rgb": [0, 170, 255]}],
        }

    theme = str(llm_data.get("theme") or "未命名")
    reason = str(llm_data.get("reason") or "")
    count = _clamp_int(llm_data.get("color_count_suggestion", 2), low=1, high=3)

    candidate_colors_raw = llm_data.get("candidate_colors", [])
    candidates: List[Dict[str, Any]] = [c for c in candidate_colors_raw if isinstance(c, dict)]

    final_selection = _select_final_colors(candidates, count=count)
    if not final_selection:
        final_selection = [{"name": "Default Blue", "rgb": [0, 170, 255]}]

    final_rgb_list = [c["rgb"] for c in final_selection]
    save_strip_data(final_rgb_list)
    save_strip_command(
        {
            "mode": "static",
            "colors": final_rgb_list,
            "brightness": 1.0,
            "speed": 2.0,
            "led_count": 60,
        }
    )

    return {
        "theme": theme,
        "reason": reason,
        "final_selection": final_selection,
    }
