import json
import math
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from ai_client import post_chat_json
from config_loader import get_config
from prompt_store import render_prompt
import strip_effects

API_KEY = get_config("AIHUBMIX_API_KEY", "")
MODEL_ID_CHAT = "gpt-5-mini"

# Optional: plain-text knowledge base for strip color generation.
# Format: one entry per line.
STRIP_KB_FILE = get_config(
    "STRIP_KB_FILE", os.path.join(os.path.dirname(__file__), "strip_kb.txt")
)

# Data Persistence
DATA_FILE = "latest_strip_data.json"
STRIP_COMMAND_FILE = "latest_strip_command.json"


@dataclass(frozen=True)
class _KbEntry:
    line: str
    term_freq: Dict[str, int]
    doc_len: int


@dataclass(frozen=True)
class _KbIndexCache:
    mtime: float
    entries: List[_KbEntry]
    doc_freq: Dict[str, int]
    avg_doc_len: float


_KB_INDEX_CACHE: Optional[_KbIndexCache] = None

_BM25_K1 = 1.5
_BM25_B = 0.75
_BM25_MIN_SCORE: float = 1.0  # 低于此分数的条目不注入提示词，避免弱相关 KB 锚定颜色


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


def _normalize_color_entry(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        rgb = value.get("rgb")
        if not isinstance(rgb, list) or len(rgb) != 3:
            return None
        name = value.get("name")
        normalized: dict[str, Any] = {"rgb": _normalize_rgb(rgb)}
        if isinstance(name, str) and name.strip():
            normalized["name"] = name.strip()
        return normalized

    if isinstance(value, list) and len(value) == 3:
        return {"rgb": _normalize_rgb(value)}

    return None


def get_validation_status(rgb: List[int]) -> Dict[str, Any]:
    """Validate a color for LED strip usage.

    We keep validation permissive to allow low-energy themes (sleep/meditation),
    while still avoiding colors that are too dark to be visible.
    """

    max_val = max(rgb)
    if max_val < 40:
        return {"valid": False, "reason": "亮度不足"}
    return {"valid": True, "reason": None}


def save_strip_data(data: List[Any]) -> None:
    normalized: list[list[int]] = []
    for item in data:
        color_entry = _normalize_color_entry(item)
        if color_entry is not None:
            normalized.append(color_entry["rgb"])
    try:
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(normalized, f, ensure_ascii=False)
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
    payload = normalize_strip_command(command)
    payload.setdefault("updated_at_ms", int(time.time() * 1000))
    try:
        with open(STRIP_COMMAND_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        # Keep legacy color-only data in sync from the richer command document.
        save_strip_data(payload.get("colors", []))
    except Exception as e:
        print(f"Failed to save strip command: {e}")


def load_strip_command() -> Dict[str, Any]:
    if os.path.exists(STRIP_COMMAND_FILE):
        try:
            with open(STRIP_COMMAND_FILE, "r", encoding="utf-8") as f:
                parsed = json.load(f)
                if isinstance(parsed, dict):
                    return normalize_strip_command(parsed)
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


def normalize_strip_command(command: Dict[str, Any] | None) -> Dict[str, Any]:
    payload = dict(command or {})
    colors = payload.get("colors")
    normalized_colors: list[dict[str, Any]] = []
    if isinstance(colors, list):
        for item in colors:
            color_entry = _normalize_color_entry(item)
            if color_entry is not None:
                normalized_colors.append(color_entry)

    payload["render_target"] = str(payload.get("render_target") or "cloud").strip().lower()
    payload["mode"] = str(payload.get("mode") or "static").strip().lower()
    payload["colors"] = normalized_colors or [{"rgb": [0, 170, 255]}]

    try:
        brightness = float(payload.get("brightness", 1.0) or 1.0)
    except Exception:
        brightness = 1.0
    payload["brightness"] = max(0.0, min(1.0, brightness))

    try:
        speed = float(payload.get("speed", 2.0))
    except Exception:
        speed = 2.0
    payload["speed"] = speed if speed > 0 else 2.0

    try:
        led_count = int(payload.get("led_count", 60))
    except Exception:
        led_count = 60
    payload["led_count"] = max(1, min(2000, led_count))

    mode_options = payload.get("mode_options")
    payload["mode_options"] = mode_options if isinstance(mode_options, dict) else None
    return payload


def render_strip_frame_payload(
    command: Dict[str, Any] | None = None,
    *,
    now_s: float | None = None,
    led_count: int | None = None,
    brightness_scale: float = 1.0,
    encoding: str = "rgb24",
) -> dict[str, Any]:
    normalized = normalize_strip_command(command if command is not None else load_strip_command())
    effective_led_count = led_count if led_count is not None else int(normalized.get("led_count", 60))

    render_command = dict(normalized)
    render_command["brightness"] = max(0.0, min(1.0, normalized["brightness"] * brightness_scale))
    frame = strip_effects.render_strip_frame(render_command, now_s=now_s, led_count=effective_led_count)

    encoding_name = str(encoding or "rgb24").strip().lower()
    if encoding_name == "rgb24":
        raw = strip_effects.frame_to_raw_bytes(frame)
        meta = {"encoding": "rgb24", "bit_depth": 24, "bytes_per_led": 3}
    elif encoding_name == "rgb565":
        raw = strip_effects.frame_to_rgb565_bytes(frame)
        meta = {"encoding": "rgb565", "bit_depth": 16, "bytes_per_led": 2}
    elif encoding_name == "rgb111":
        raw = strip_effects.frame_to_rgb111_bytes(frame)
        meta = {"encoding": "rgb111", "bit_depth": 3, "bytes_per_led": None}
    else:
        raise ValueError("unsupported encoding")

    return {
        "command": render_command,
        "frame": frame,
        "raw": raw,
        "meta": meta,
        "led_count": effective_led_count,
    }


def _tokenize_for_retrieval_terms(text: str) -> List[str]:
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

    return word_tokens + cjk_chars + bigrams


def _tokenize_for_retrieval(text: str) -> Set[str]:
    return set(_tokenize_for_retrieval_terms(text))


def _expand_with_synonyms(tokens: Set[str]) -> Set[str]:
    expanded = set(tokens)
    for group in _SYNONYM_GROUPS:
        if expanded & group:
            expanded |= group
    return expanded



def _build_kb_search_text(line: str) -> str:
    """Build search text from a KB JSON line.

    Only indexes ``theme`` and ``keywords`` — the fields that carry
    intent/topic signals.  Description and tips contain natural-language
    prose full of common words that would inflate BM25 scores for
    unrelated queries (e.g. "慢一点" matching "一点" in descriptions).
    """

    try:
        payload = json.loads(line)
    except Exception:
        return line

    if not isinstance(payload, dict):
        return line

    weighted_parts: List[str] = []

    theme = str(payload.get("theme") or "").strip()
    if theme:
        weighted_parts.extend([theme] * 4)

    keywords = payload.get("keywords")
    if isinstance(keywords, list):
        for keyword in keywords:
            text = str(keyword or "").strip()
            if text:
                weighted_parts.extend([text] * 3)

    return " ".join(weighted_parts) or line


def _load_kb_index_cached(file_path: str, max_lines: int = 2000) -> _KbIndexCache:
    global _KB_INDEX_CACHE

    if not os.path.exists(file_path):
        _KB_INDEX_CACHE = _KbIndexCache(mtime=-1, entries=[], doc_freq={}, avg_doc_len=0.0)
        return _KB_INDEX_CACHE

    try:
        mtime = os.path.getmtime(file_path)
    except OSError:
        _KB_INDEX_CACHE = _KbIndexCache(mtime=-1, entries=[], doc_freq={}, avg_doc_len=0.0)
        return _KB_INDEX_CACHE

    if _KB_INDEX_CACHE is not None and _KB_INDEX_CACHE.mtime == mtime:
        return _KB_INDEX_CACHE

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

    entries: List[_KbEntry] = []
    doc_freq: Dict[str, int] = {}
    total_doc_len = 0

    for line in lines:
        search_text = _build_kb_search_text(line)
        terms = _tokenize_for_retrieval_terms(search_text)
        if not terms:
            continue

        term_freq: Dict[str, int] = {}
        for term in terms:
            term_freq[term] = term_freq.get(term, 0) + 1

        doc_len = len(terms)
        total_doc_len += doc_len
        entries.append(_KbEntry(line=line, term_freq=term_freq, doc_len=doc_len))

        for term in term_freq:
            doc_freq[term] = doc_freq.get(term, 0) + 1

    avg_doc_len = (total_doc_len / len(entries)) if entries else 0.0
    _KB_INDEX_CACHE = _KbIndexCache(
        mtime=mtime,
        entries=entries,
        doc_freq=doc_freq,
        avg_doc_len=avg_doc_len,
    )
    return _KB_INDEX_CACHE


def _bm25_score(query_terms: Set[str], entry: _KbEntry, cache: _KbIndexCache) -> float:
    if not query_terms or not entry.term_freq:
        return 0.0

    score = 0.0
    total_docs = max(1, len(cache.entries))
    avg_doc_len = cache.avg_doc_len or 1.0

    for term in query_terms:
        tf = entry.term_freq.get(term, 0)
        if tf <= 0:
            continue

        df = cache.doc_freq.get(term, 0)
        idf = math.log(1.0 + (total_docs - df + 0.5) / (df + 0.5))
        denom = tf + _BM25_K1 * (1.0 - _BM25_B + _BM25_B * (entry.doc_len / avg_doc_len))
        score += idf * ((tf * (_BM25_K1 + 1.0)) / denom)

    return score


def retrieve_strip_kb_entries(user_input: str, *, top_k: int = 3) -> List[str]:
    """Return retrieved KB entries as raw JSON lines."""

    # Expand synonyms first (so single-char triggers like "困" pull in multi-char synonyms),
    # then keep only tokens with length >= 2 to avoid single-char IDF noise in BM25.
    all_tokens = _tokenize_for_retrieval(user_input)
    expanded = _expand_with_synonyms(all_tokens)
    query_tokens = {t for t in expanded if len(t) >= 2}
    if not query_tokens:
        return []

    cache = _load_kb_index_cached(STRIP_KB_FILE)
    if not cache.entries:
        return []

    scored: List[Tuple[float, str]] = []
    for entry in cache.entries:
        score = _bm25_score(query_tokens, entry, cache)
        if score >= _BM25_MIN_SCORE:
            scored.append((score, entry.line))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [line for _, line in scored[: max(0, int(top_k))]]


def get_strip_kb_context(user_input: str, *, top_k: int = 3) -> str:
    """Return a short KB context block for prompt injection.

    Notes:
    - KB format is one line per entry.
    - Retrieval uses BM25 with synonym expansion.
    - Returns a compact block to minimize token usage and latency.
    """

    chosen = retrieve_strip_kb_entries(user_input, top_k=top_k)
    if not chosen:
        return ""

    joined = "\n".join(f"- {line}" for line in chosen)
    return (
        "\n\n# 补充资料（本地知识库，每行一条，供参考）\n"
        "以下是从本地知识库召回的若干条目，可作为灵感和风格参考，而非硬性约束；请优先满足用户意图，并根据美学搭配、色彩和谐与整体氛围自由判断是否采纳、部分采纳或不采纳这些条目。\n"
        f"{joined}\n"
    )


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


def get_current_state_desc() -> str:
    """Return a string description of the current strip state for LLM context."""
    cmd = load_strip_command()
    mode = cmd.get("mode", "static")
    colors = cmd.get("colors", [])
    speed = cmd.get("speed", 2.0)
    brightness = cmd.get("brightness", 1.0)
    
    color_parts: list[str] = []
    for color in colors:
        color_entry = _normalize_color_entry(color)
        if color_entry is None:
            continue
        label = color_entry.get("name")
        rgb = color_entry["rgb"]
        color_parts.append(f"{label} RGB{rgb}" if label else f"RGB{rgb}")

    color_desc = ", ".join(color_parts)
    return f"模式：{mode}；颜色：[{color_desc}]；速度：{speed}；亮度：{brightness}"


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
    current_state = get_current_state_desc()
    prompt = render_prompt(
        "strip",
        {"user_input": text, "kb_context": kb_context, "current_state": current_state},
        seed=text,
    )
    prompt = (
        "请严格使用简体中文输出 theme、reason、颜色名称等所有自然语言字段，"
        "不要输出英文解释，不要夹杂英文句子。\n\n"
        + prompt
    )

    try:
        llm_data, _elapsed = post_chat_json(
            model=MODEL_ID_CHAT,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.7,
            timeout_s=20,
            api_key=API_KEY,
        )
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
    
    try:
        brightness = float(llm_data.get("brightness", 1.0))
    except Exception:
        brightness = 1.0
    brightness = max(0.0, min(1.0, brightness))

    candidate_colors_raw = llm_data.get("candidate_colors", [])
    candidates: List[Dict[str, Any]] = [c for c in candidate_colors_raw if isinstance(c, dict)]

    final_selection = _select_final_colors(candidates, count=count)
    if not final_selection:
        final_selection = [{"name": "Default Blue", "rgb": [0, 170, 255]}]

    final_rgb_list = [c["rgb"] for c in final_selection]
    save_strip_command(
        {
            "mode": "static",
            "colors": final_rgb_list,
            "brightness": brightness,
            "speed": 2.0,
            "led_count": 60,
        }
    )

    return {
        "theme": theme,
        "reason": reason,
        "final_selection": final_selection,
    }
