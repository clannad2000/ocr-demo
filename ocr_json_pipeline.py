#!/usr/bin/env python3
"""Standalone JSON-page OCR pipeline for illustrated math books.

The demo intentionally uses only the Python standard library plus Poppler CLI
tools. API credentials are read from process environment variables or a project
.env file and are never serialized into OCR results.

This variant writes page PNG and JSON records but omits
``pages/page-NNNN.md``. It still writes ``manifest.json``, ``summary.md``, and
``study.html`` in study mode. Translation is performed through DeepL Remote MCP.
"""

from __future__ import annotations

import argparse
import atexit
import base64
import collections
import concurrent.futures
import copy
import dataclasses
import difflib
import hashlib
import html
import json
import os
import pathlib
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Any


DEEPSEEK_PROMPT = "<image>\n<|grounding|>OCR this image."

PRIMARY_PROMPT = """Perform exact OCR on this English illustrated math-book page.
Transcribe every visible textual element exactly, including speech bubbles,
captions, headings, questions, sound effects, handwritten text, tables, chart
cells, page numbers, diagram labels, numbers, formulas, operators, units, and
punctuation. Do not translate, solve, infer, summarize, normalize, correct, or
omit decorative or repeated text. Preserve natural reading order. For tables
and charts, preserve row order and transcribe every cell. Return a numbered
list with one visual text region per item. If any text is genuinely unreadable,
transcribe the visible portion and append [uncertain]."""

VERIFY_PROMPT = """Act as an independent exact OCR verifier for this English
illustrated math-book page. Transcribe all visible text exactly. Pay special
attention to small numbers, page numbers, diagram labels, formulas, operators,
units, repeated labels, sound effects, and stylized handwriting. Do not solve,
translate, summarize, normalize, or invent text. Preserve natural reading
order and return one numbered item per visual text region."""

STUDY_PRIMARY_PROMPT = """Read this English illustrated math-book page for a
personal-study translation. Extract only natural-language English that affects
understanding: dialogue, questions, instructions, explanations, definitions,
headings or contents entries, captions, and footnotes. Preserve wording and
numerals exactly; do not translate, solve, correct, summarize, or invent.

Exclude decorative sound effects, standalone printed page-number footers, pure
number tables such as hundred charts, formulas without explanatory prose,
standalone numeric geometry labels or side lengths, and crossed-out drafting
text. Keep all numbers, formulas, and units that occur inside an otherwise
translatable sentence or phrase. Keep page references and page ranges attached
to contents or index entries because they are needed for navigation.
For contents and index entries, omit decorative leader dots and keep only the
entry wording plus its page number or page range, for example "Angles - 12".

Return only one JSON object with this schema:
{"regions":[{"id":"r001","type":"dialogue|question|instruction|explanation|definition|heading|contents|caption|footnote|other","source_text":"exact English"}],"skipped":{"sound_effects":[],"numeric_tables":false,"formulas":[],"page_numbers":[],"numeric_diagram_labels":[],"struck_out_text":[]}}
Use stable IDs in reading order. Use an empty array when there is no region."""

STUDY_VERIFY_PROMPT = """Independently extract only English natural-language
text that should be translated for personal study of this illustrated math
book: dialogue, questions, instructions, explanations, definitions, headings
or contents entries, captions, and footnotes. Preserve exact wording. Exclude
decorative sound effects, standalone page-number footers, pure number tables,
prose-free formulas, standalone numeric geometry labels or side lengths, and
crossed-out drafting text. Keep numbers and units inside translatable sentences,
and keep page references attached to contents or index entries. Do not
translate, solve, correct, or invent.
Omit decorative leader dots in contents/index entries while keeping their page
references.
Return only JSON: {"regions":[{"id":"r001","type":"dialogue|question|instruction|explanation|definition|heading|contents|caption|footnote|other","source_text":"exact English"}],"skipped":{}}"""

TRANSLATION_PROMPT = """Translate the supplied English math-book study regions
into concise, natural Simplified Chinese. Translate faithfully without solving,
explaining, adding, or deleting information. Preserve names and mathematical
meaning. Keep numbers, operators, and formulas exact; translate units naturally.
Return only JSON with this schema:
{"translations":[{"id":"r001","translation":"Chinese translation"}]}
Return exactly one item for every supplied ID.

Use this terminology consistently:
- skip-counting = 跳数
- hundred chart = 百数表
- polyomino / polyominoes = 多格骨牌
- triomino = 三格骨牌; tetromino = 四格骨牌
- rep-tile = 自相似分割图形（rep-tile）
- rectilinear shape = 直线形图形
- scalene triangle = 不等边三角形
- equilateral triangle = 等边三角形
- isosceles triangle = 等腰三角形
In a rep-tile context, "copies of itself" means smaller copies similar to the
original, not copies congruent to the original. Preserve fictional creature
names such as elefinch and octapug in English. Preserve contents/index page
references exactly."""

TRANSLATION_VERIFY_PROMPT = """Review English-to-Simplified-Chinese translations
from an illustrated Grade 3 mathematics guide. Compare meaning, mathematical
terminology, wordplay, names, numbers, and instructional intent. Flag only real
problems; do not rewrite acceptable stylistic variants. Pay special attention
to short labels whose meaning depends on the other regions on the same page.

Return only one JSON object:
{"issues":[{"id":"r001","category":"meaning|math_term|wordplay|name|number|omission|addition|fluency","explanation":"concise Chinese explanation","suggested_translation":"corrected Chinese"}]}
Use an empty issues array when all translations are acceptable. IDs must come
from the supplied regions. This is an advisory review: never silently replace
the supplied translation."""

CHAPTER_REVIEW_PROMPT = """Perform a blind, chapter-level review of an English
to Simplified Chinese mathematics-book translation. The input contains a
compact terminology table, character-name candidates, headings, repeated
expressions, and every saved English/Chinese region grouped by page.

Use the whole chapter to judge meaning and consistency. Only report actual
errors. Do not report acceptable stylistic alternatives. If your analysis
concludes that an item is acceptable, omit it completely. Do not expose
reasoning or self-corrections. Do not solve exercises. Do not modify any saved
translation. Page and region IDs must come from the input.

Return only one JSON object:
{"issues":[{"page":18,"id":"r003","category":"meaning|math_term|wordplay|name|number|omission|addition|fluency","suggested_translation":"corrected Chinese","reason":"one concise Chinese sentence"}],"global_consistency":[{"category":"term|name|title|repeated_expression","item":"English item","preferred_translation":"Chinese","reason":"one concise Chinese sentence","affected":[{"page":18,"id":"r003"}]}]}
Use empty arrays when there are no findings. Keep reasons concise. Never include
an issue merely to say that it is not an issue."""

CHAPTER_TERMINOLOGY = (
    ("acute angle", "锐角"),
    ("right angle", "直角"),
    ("obtuse angle", "钝角"),
    ("angle", "角"),
    ("side", "边"),
    ("vertex / corner", "顶点"),
    ("diagonal", "对角线"),
    ("skip-counting", "跳数"),
    ("hundred chart", "百数表"),
    ("polyomino", "多格骨牌"),
    ("triomino", "三格骨牌"),
    ("tetromino", "四格骨牌"),
    ("rep-tile", "自相似分割图形（rep-tile）"),
    ("rectilinear shape", "直线形图形"),
    ("scalene triangle", "不等边三角形"),
    ("equilateral triangle", "等边三角形"),
    ("isosceles triangle", "等腰三角形"),
)

CHAPTER_NAME_TERMS = (
    "Beast Academy",
    "Professor Grok",
    "GROGG",
    "Winnie",
    "Lizzie",
    "Rotunda",
)

STUDY_TYPES = {
    "dialogue",
    "question",
    "instruction",
    "explanation",
    "definition",
    "heading",
    "contents",
    "caption",
    "footnote",
    "other",
}

DEEPSEEK_ENDPOINT = "https://api.siliconflow.cn/v1/chat/completions"
DASHSCOPE_CHAT_ENDPOINT = (
    "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
)
DASHSCOPE_RESPONSES_ENDPOINT = (
    "https://dashscope.aliyuncs.com/compatible-mode/v1/responses"
)
DEEPL_MCP_ENDPOINT = "https://mcp.deepl.com/v1/mcp"

DEFAULT_MODEL_PROFILES = {
    "deepseek-ocr": {
        "provider": "siliconflow",
        "model": "deepseek-ai/DeepSeek-OCR",
        "api_key_name": "siliconflow",
        "chat_endpoint": DEEPSEEK_ENDPOINT,
        "enable_thinking": False,
        "max_tokens": 4096,
    },
    "qwen-flash": {
        "provider": "dashscope",
        "model": "qwen3.6-flash",
        "api_key_name": "dashscope",
        "chat_endpoint": DASHSCOPE_CHAT_ENDPOINT,
        "responses_endpoint": DASHSCOPE_RESPONSES_ENDPOINT,
        "enable_thinking": True,
        "max_tokens": 8192,
    },
    "qwen-max": {
        "provider": "dashscope",
        "model": "qwen3.7-max-2026-06-08",
        "api_key_name": "dashscope",
        "chat_endpoint": DASHSCOPE_CHAT_ENDPOINT,
        "responses_endpoint": DASHSCOPE_RESPONSES_ENDPOINT,
        "enable_thinking": False,
        "max_tokens": 16384,
    },
}

DEFAULT_MODEL_USAGE = {
    "layout_ocr": "deepseek-ocr",
    "primary_ocr": "qwen-flash",
    "ocr_verifier": "qwen-max",
}

print_lock = threading.Lock()


class TeeStream:
    def __init__(self, terminal: Any, log_file: Any) -> None:
        self.terminal = terminal
        self.log_file = log_file
        self.lock = threading.Lock()

    def write(self, value: str) -> int:
        with self.lock:
            self.terminal.write(value)
            self.log_file.write(value)
        return len(value)

    def flush(self) -> None:
        with self.lock:
            self.terminal.flush()
            self.log_file.flush()

    def isatty(self) -> bool:
        return self.terminal.isatty()

    @property
    def encoding(self) -> str:
        return getattr(self.terminal, "encoding", "utf-8")


class TeeLogging:
    def __init__(self, log_path: pathlib.Path | None) -> None:
        self.log_path = log_path
        self.handle: Any = None
        self.original_stdout: Any = None
        self.original_stderr: Any = None

    def __enter__(self) -> "TeeLogging":
        if self.log_path is None:
            return self
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.log_path.open("a", encoding="utf-8", buffering=1)
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        sys.stdout = TeeStream(sys.stdout, self.handle)
        sys.stderr = TeeStream(sys.stderr, self.handle)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.handle is None:
            return
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        self.handle.close()


class DeepLMcpBridge:
    """Thread-safe JSON-lines bridge to the project-local Node MCP client."""

    def __init__(
        self,
        node_command: str,
        script_path: pathlib.Path,
        timeout_seconds: int = 360,
    ) -> None:
        self.node_command = node_command
        self.script_path = script_path
        self.timeout_seconds = timeout_seconds
        self.process: subprocess.Popen[str] | None = None
        self.request_id = 0
        self.lock = threading.Lock()
        self.response_queue: queue.Queue[str | BaseException | None] | None = None
        self.stdout_thread: threading.Thread | None = None
        self.stderr_thread: threading.Thread | None = None

    def _start(self) -> None:
        if self.process is not None and self.process.poll() is None:
            return
        self.process = subprocess.Popen(
            [self.node_command, str(self.script_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="strict",
            bufsize=1,
        )
        process = self.process
        self.response_queue = queue.Queue()
        response_queue = self.response_queue

        def forward_stdout() -> None:
            assert process.stdout is not None
            try:
                for line in process.stdout:
                    response_queue.put(line)
            except BaseException as error:
                response_queue.put(error)
            finally:
                response_queue.put(None)

        def forward_stderr() -> None:
            assert process.stderr is not None
            for line in process.stderr:
                sys.stderr.write(line)
                sys.stderr.flush()

        self.stdout_thread = threading.Thread(
            target=forward_stdout,
            name="deepl-mcp-stdout",
            daemon=True,
        )
        self.stdout_thread.start()
        self.stderr_thread = threading.Thread(
            target=forward_stderr,
            name="deepl-mcp-stderr",
            daemon=True,
        )
        self.stderr_thread.start()

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            self._start()
            assert self.process is not None
            assert self.process.stdin is not None
            assert self.process.stdout is not None
            assert self.response_queue is not None
            self.request_id += 1
            expected_id = self.request_id
            request = dict(payload)
            request["id"] = expected_id
            try:
                self.process.stdin.write(
                    json.dumps(request, ensure_ascii=False) + "\n"
                )
                self.process.stdin.flush()
            except (BrokenPipeError, OSError) as error:
                exit_code = self.process.poll()
                raise RuntimeError(
                    f"DeepL MCP helper stopped before request (exit={exit_code})"
                ) from error
            try:
                line = self.response_queue.get(timeout=self.timeout_seconds)
            except queue.Empty:
                self.close()
                raise TimeoutError(
                    f"DeepL MCP request timed out after {self.timeout_seconds}s"
                )
            if line is None:
                exit_code = self.process.poll()
                raise RuntimeError(
                    f"DeepL MCP helper returned no response (exit={exit_code})"
                )
            if isinstance(line, BaseException):
                raise RuntimeError("Cannot read DeepL MCP helper output") from line
            try:
                response = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"Invalid DeepL MCP helper response: {line[:500]}"
                ) from error
            if response.get("id") != expected_id:
                raise RuntimeError(
                    f"DeepL MCP response id mismatch: {response.get('id')}"
                )
            if not response.get("ok"):
                raise RuntimeError(response.get("error") or "DeepL MCP request failed")
            result = response.get("result")
            if not isinstance(result, dict):
                raise RuntimeError("DeepL MCP helper returned an invalid result")
            return result

    def close(self) -> None:
        process = self.process
        self.process = None
        self.response_queue = None
        if process is None:
            return
        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)


deepl_bridge_lock = threading.Lock()
deepl_bridges: dict[tuple[str, str], DeepLMcpBridge] = {}


def get_deepl_bridge(args: argparse.Namespace) -> DeepLMcpBridge:
    key = (args.deepl_node_command, str(args.deepl_bridge_script))
    with deepl_bridge_lock:
        bridge = deepl_bridges.get(key)
        if bridge is None:
            bridge = DeepLMcpBridge(
                node_command=args.deepl_node_command,
                script_path=args.deepl_bridge_script,
            )
            deepl_bridges[key] = bridge
        return bridge


def close_deepl_bridges() -> None:
    with deepl_bridge_lock:
        bridges = list(deepl_bridges.values())
        deepl_bridges.clear()
    for bridge in bridges:
        bridge.close()


atexit.register(close_deepl_bridges)


@dataclasses.dataclass
class ApiResult:
    provider: str
    model: str
    mode: str
    status: int | None
    elapsed_seconds: float
    usage: dict[str, Any]
    content: str
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def parse_pages(spec: str) -> list[int]:
    pages: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            start, end = int(left), int(right)
            if start < 1 or end < start:
                raise ValueError(f"Invalid page range: {part}")
            pages.update(range(start, end + 1))
        else:
            page = int(part)
            if page < 1:
                raise ValueError(f"Invalid page number: {page}")
            pages.add(page)
    if not pages:
        raise ValueError("No pages selected")
    return sorted(pages)


def pdf_page_count(pdf: pathlib.Path, pdfinfo_command: str = "pdfinfo") -> int:
    completed = subprocess.run(
        [pdfinfo_command, str(pdf)],
        check=True,
        capture_output=True,
        text=True,
    )
    match = re.search(r"^Pages:\s+(\d+)\s*$", completed.stdout, re.MULTILINE)
    if not match:
        raise RuntimeError("Could not read PDF page count from pdfinfo")
    return int(match.group(1))


def atomic_write_text(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_write_json(path: pathlib.Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_page_json(path: pathlib.Path, record: dict[str, Any]) -> None:
    """Persist one page record without creating per-page Markdown."""

    atomic_write_json(path, record)


def render_page(
    pdf: pathlib.Path,
    page: int,
    image_path: pathlib.Path,
    long_edge: int,
    resume: bool,
    pdftoppm_command: str = "pdftoppm",
) -> None:
    if resume and image_path.exists() and image_path.stat().st_size > 0:
        return
    image_path.parent.mkdir(parents=True, exist_ok=True)
    prefix = image_path.with_suffix("")
    subprocess.run(
        [
            pdftoppm_command,
            "-f",
            str(page),
            "-l",
            str(page),
            "-scale-to",
            str(long_edge),
            "-png",
            "-singlefile",
            str(pdf),
            str(prefix),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    if not image_path.exists() or image_path.stat().st_size == 0:
        raise RuntimeError(f"Rendering did not produce {image_path}")


def supports_poppler_pdftoppm(command: pathlib.Path) -> bool:
    try:
        completed = subprocess.run(
            [str(command), "-h"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    help_text = completed.stdout + completed.stderr
    return "-scale-to" in help_text and "-singlefile" in help_text


def find_poppler_pdftoppm(explicit: str | None = None) -> str:
    candidates: list[pathlib.Path] = []
    if explicit:
        candidates.append(pathlib.Path(explicit).expanduser())
    else:
        command_from_path = shutil.which("pdftoppm")
        if command_from_path:
            candidates.append(pathlib.Path(command_from_path))
        for directory in os.environ.get("PATH", "").split(os.pathsep):
            if directory:
                candidates.append(pathlib.Path(directory) / "pdftoppm")
                # Windows stores the Poppler executable with this suffix.
                # Checking it on other platforms is harmless and keeps this
                # lookup independent of the host's executable-suffix rules.
                candidates.append(pathlib.Path(directory) / "pdftoppm.exe")
        candidates.extend(
            [
                pathlib.Path("/usr/local/opt/poppler/bin/pdftoppm"),
                pathlib.Path("/opt/homebrew/opt/poppler/bin/pdftoppm"),
            ]
        )
        brew = shutil.which("brew")
        if brew:
            try:
                completed = subprocess.run(
                    [brew, "--prefix", "poppler"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=True,
                )
                prefix = completed.stdout.strip()
                if prefix:
                    candidates.append(pathlib.Path(prefix) / "bin" / "pdftoppm")
            except (OSError, subprocess.SubprocessError):
                pass
    seen: set[str] = set()
    for candidate in candidates:
        try:
            resolved = str(candidate.resolve())
        except OSError:
            resolved = str(candidate)
        if resolved in seen or not candidate.is_file():
            continue
        seen.add(resolved)
        if supports_poppler_pdftoppm(candidate):
            return str(candidate)
    if explicit:
        raise RuntimeError(
            f"Configured pdftoppm does not support Poppler options: {explicit}"
        )
    raise RuntimeError(
        "No Poppler-compatible pdftoppm found; Xpdf pdftoppm is not sufficient"
    )


def find_matching_pdfinfo(pdftoppm_command: str) -> str:
    pdftoppm_path = pathlib.Path(pdftoppm_command).resolve()
    sibling_names = (
        ("pdfinfo.exe", "pdfinfo")
        if pdftoppm_path.suffix.lower() == ".exe"
        else ("pdfinfo", "pdfinfo.exe")
    )
    for name in sibling_names:
        sibling = pdftoppm_path.with_name(name)
        if sibling.is_file():
            return str(sibling)
    command = shutil.which("pdfinfo")
    if command:
        return command
    raise RuntimeError("Required command not found: pdfinfo")


def image_data_url(path: pathlib.Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def post_json(
    endpoint: str,
    payload: dict[str, Any],
    api_key: str,
    retries: int = 2,
) -> tuple[int, dict[str, Any]]:
    body = json.dumps(payload).encode("utf-8")
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        request = urllib.request.Request(
            endpoint,
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                data = json.loads(response.read().decode("utf-8"))
                return response.status, data
        except urllib.error.HTTPError as error:
            last_error = error
            if error.code not in {408, 429, 500, 502, 503, 504} or attempt >= retries:
                body_text = error.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"HTTP {error.code}: {body_text[:1000]}") from error
        except (TimeoutError, urllib.error.URLError) as error:
            last_error = error
            if attempt >= retries:
                raise RuntimeError(f"Network error: {error}") from error
        time.sleep(2**attempt)
    raise RuntimeError(f"Request failed: {last_error}")


def chat_content(data: dict[str, Any]) -> str:
    choices = data.get("choices") or []
    if not choices:
        return ""
    content = choices[0].get("message", {}).get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    return str(content)


def responses_content(data: dict[str, Any]) -> str:
    chunks: list[str] = []
    for item in data.get("output", []):
        if item.get("type") != "message":
            continue
        for part in item.get("content", []):
            if part.get("type") == "output_text" and part.get("text"):
                chunks.append(part["text"])
    return "\n".join(chunks)


def apply_thinking_setting(
    payload: dict[str, Any],
    provider: str,
    enable_thinking: bool | None,
) -> None:
    """Apply the provider-specific thinking-mode request field."""

    if enable_thinking is None:
        return
    if provider == "deepseek":
        payload["thinking"] = {
            "type": "enabled" if enable_thinking else "disabled"
        }
    else:
        payload["enable_thinking"] = enable_thinking


def call_deepseek(
    image_url: str,
    api_key: str,
    model: str,
    endpoint: str = DEEPSEEK_ENDPOINT,
    provider: str = "siliconflow",
    max_tokens: int = 4096,
) -> ApiResult:
    started = time.monotonic()
    try:
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url, "detail": "high"},
                        },
                        {"type": "text", "text": DEEPSEEK_PROMPT},
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        status, data = post_json(endpoint, payload, api_key)
        return ApiResult(
            provider=provider,
            model=model,
            mode="grounded-detailed",
            status=status,
            elapsed_seconds=round(time.monotonic() - started, 2),
            usage=data.get("usage", {}),
            content=chat_content(data),
        )
    except Exception as error:
        return ApiResult(
            provider=provider,
            model=model,
            mode="grounded-detailed",
            status=None,
            elapsed_seconds=round(time.monotonic() - started, 2),
            usage={},
            content="",
            error=f"{type(error).__name__}: {error}",
        )


def call_qwen_primary(
    image_url: str,
    api_key: str,
    model: str,
    prompt: str = PRIMARY_PROMPT,
    endpoint: str = DASHSCOPE_CHAT_ENDPOINT,
    provider: str = "dashscope",
    enable_thinking: bool | None = True,
    max_tokens: int = 8192,
) -> ApiResult:
    started = time.monotonic()
    try:
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        apply_thinking_setting(payload, provider, enable_thinking)
        status, data = post_json(endpoint, payload, api_key)
        return ApiResult(
            provider=provider,
            model=model,
            mode="thinking" if enable_thinking else "non-thinking",
            status=status,
            elapsed_seconds=round(time.monotonic() - started, 2),
            usage=data.get("usage", {}),
            content=chat_content(data),
        )
    except Exception as error:
        return ApiResult(
            provider=provider,
            model=model,
            mode="thinking" if enable_thinking else "non-thinking",
            status=None,
            elapsed_seconds=round(time.monotonic() - started, 2),
            usage={},
            content="",
            error=f"{type(error).__name__}: {error}",
        )


def call_qwen_verifier(
    image_url: str,
    api_key: str,
    model: str,
    prompt: str = VERIFY_PROMPT,
    endpoint: str = DASHSCOPE_RESPONSES_ENDPOINT,
    provider: str = "dashscope",
    enable_thinking: bool | None = False,
    max_tokens: int = 8192,
) -> ApiResult:
    started = time.monotonic()
    try:
        payload = {
            "model": model,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_image", "image_url": image_url},
                        {"type": "input_text", "text": prompt},
                    ],
                }
            ],
            "temperature": 0,
            "max_output_tokens": max_tokens,
        }
        apply_thinking_setting(payload, provider, enable_thinking)
        status, data = post_json(endpoint, payload, api_key)
        return ApiResult(
            provider=provider,
            model=model,
            mode="thinking" if enable_thinking else "non-thinking",
            status=status,
            elapsed_seconds=round(time.monotonic() - started, 2),
            usage=data.get("usage", {}),
            content=responses_content(data),
        )
    except Exception as error:
        return ApiResult(
            provider=provider,
            model=model,
            mode="thinking" if enable_thinking else "non-thinking",
            status=None,
            elapsed_seconds=round(time.monotonic() - started, 2),
            usage={},
            content="",
            error=f"{type(error).__name__}: {error}",
        )


def call_qwen_translation(
    regions: list[dict[str, str]],
    api_key: str,
    model: str,
    endpoint: str = DASHSCOPE_CHAT_ENDPOINT,
    provider: str = "dashscope",
    enable_thinking: bool | None = False,
    max_tokens: int = 4096,
) -> ApiResult:
    started = time.monotonic()
    try:
        source = [
            {"id": region["id"], "source_text": region["source_text"]}
            for region in regions
        ]
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": TRANSLATION_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(source, ensure_ascii=False),
                },
            ],
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        apply_thinking_setting(payload, provider, enable_thinking)
        status, data = post_json(endpoint, payload, api_key)
        return ApiResult(
            provider=provider,
            model=model,
            mode=(
                "translation-thinking"
                if enable_thinking
                else "translation-non-thinking"
            ),
            status=status,
            elapsed_seconds=round(time.monotonic() - started, 2),
            usage=data.get("usage", {}),
            content=chat_content(data),
        )
    except Exception as error:
        return ApiResult(
            provider=provider,
            model=model,
            mode=(
                "translation-thinking"
                if enable_thinking
                else "translation-non-thinking"
            ),
            status=None,
            elapsed_seconds=round(time.monotonic() - started, 2),
            usage={},
            content="",
            error=f"{type(error).__name__}: {error}",
        )


def call_deepl_mcp_translation(
    args: argparse.Namespace,
    regions: list[dict[str, str]],
) -> ApiResult:
    started = time.monotonic()
    try:
        bridge = get_deepl_bridge(args)
        translated: list[dict[str, str]] = []
        detected_languages: set[str] = set()
        for index, region in enumerate(regions):
            request = {
                    "action": "translate",
                    "endpoint": args.deepl_mcp_endpoint,
                    "oauth_callback_port": args.deepl_oauth_callback_port,
                    "env_file": str(args.deepl_env_file),
                    "text": region["source_text"],
                    "source_lang": args.deepl_source_lang,
                    "target_lang": args.deepl_target_lang,
                    "formality": args.deepl_formality,
                    "glossary_id": args.deepl_glossary_id,
                    "style_id": args.deepl_style_id,
                    "context": build_translation_context(
                        args.deepl_context, regions, index
                    ),
                    "custom_instructions": args.deepl_custom_instructions,
                }
            result = request_deepl_with_retry(bridge, request)
            text = str(result.get("text", "")).strip()
            if not text:
                raise ValueError(f"DeepL returned empty text for {region['id']}")
            detected = result.get("detectedSourceLanguage")
            if detected:
                detected_languages.add(str(detected))
            translated.append({"id": region["id"], "translation": text})
        return ApiResult(
            provider="deepl-mcp",
            model="DeepL Remote MCP",
            mode="translation-oauth",
            status=200,
            elapsed_seconds=round(time.monotonic() - started, 2),
            usage={
                "source_characters": sum(
                    len(region["source_text"]) for region in regions
                ),
                "region_count": len(regions),
                "detected_source_languages": sorted(detected_languages),
            },
            content=json.dumps(
                {"translations": translated}, ensure_ascii=False
            ),
        )
    except Exception as error:
        return ApiResult(
            provider="deepl-mcp",
            model="DeepL Remote MCP",
            mode="translation-oauth",
            status=None,
            elapsed_seconds=round(time.monotonic() - started, 2),
            usage={},
            content="",
            error=f"{type(error).__name__}: {error}",
        )


def call_translation(
    args: argparse.Namespace, regions: list[dict[str, str]]
) -> ApiResult:
    if args.translation_provider == "deepl_mcp":
        return call_deepl_mcp_translation(args, regions)
    profile = args.translation_profile_config
    return call_qwen_translation(
        regions,
        profile["api_key"],
        profile["model"],
        profile["chat_endpoint"],
        profile["provider"],
        profile["enable_thinking"],
        profile["max_tokens"],
    )


def build_translation_context(
    base_context: str,
    regions: list[dict[str, str]],
    current_index: int,
    limit: int = 300,
) -> str:
    """Build compact page context without changing the text being translated."""

    if limit < 1:
        return ""
    base = base_context.strip()[: min(120, limit)]
    order = sorted(
        range(len(regions)),
        key=lambda index: (abs(index - current_index), index),
    )
    nearby = " | ".join(
        f"{regions[index]['id']}: {regions[index]['source_text']}"
        for index in order
    )
    current_id = regions[current_index]["id"]
    parts = [base, f"Current region: {current_id}. Nearby page text: {nearby}"]
    return " ".join(part for part in parts if part)[:limit]


def request_deepl_with_retry(
    bridge: Any,
    payload: dict[str, Any],
    retries: int = 2,
) -> dict[str, Any]:
    transient_markers = (
        "timeout",
        "timed out",
        "fetch failed",
        "temporarily",
        "rate limit",
        "429",
        "502",
        "503",
        "504",
    )
    for attempt in range(retries + 1):
        try:
            return bridge.request(payload)
        except Exception as error:
            message = str(error).lower()
            if attempt >= retries or not any(
                marker in message for marker in transient_markers
            ):
                raise
            time.sleep(2**attempt)
    raise RuntimeError("DeepL retry loop ended unexpectedly")


def call_translation_verifier(
    args: argparse.Namespace,
    regions: list[dict[str, str]],
) -> ApiResult:
    started = time.monotonic()
    profile = args.translation_verifier_profile_config
    try:
        source = [
            {
                "id": region["id"],
                "type": region["type"],
                "source_text": region["source_text"],
                "translation": region["translation"],
            }
            for region in regions
        ]
        payload = {
            "model": profile["model"],
            "messages": [
                {"role": "system", "content": TRANSLATION_VERIFY_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(source, ensure_ascii=False),
                },
            ],
            "temperature": 0,
            "max_tokens": profile["max_tokens"],
        }
        apply_thinking_setting(
            payload, profile["provider"], profile["enable_thinking"]
        )
        status, data = post_json(
            profile["chat_endpoint"], payload, profile["api_key"]
        )
        return ApiResult(
            provider=profile["provider"],
            model=profile["model"],
            mode=(
                "translation-verifier-thinking"
                if profile["enable_thinking"]
                else "translation-verifier-non-thinking"
            ),
            status=status,
            elapsed_seconds=round(time.monotonic() - started, 2),
            usage=data.get("usage", {}),
            content=chat_content(data),
        )
    except Exception as error:
        return ApiResult(
            provider=profile["provider"],
            model=profile["model"],
            mode=(
                "translation-verifier-thinking"
                if profile["enable_thinking"]
                else "translation-verifier-non-thinking"
            ),
            status=None,
            elapsed_seconds=round(time.monotonic() - started, 2),
            usage={},
            content="",
            error=f"{type(error).__name__}: {error}",
        )


def call_chapter_translation_verifier(
    args: argparse.Namespace,
    review_input: str,
    model_config: dict[str, Any],
) -> ApiResult:
    started = time.monotonic()
    config = model_config
    try:
        payload = {
            "model": config["model"],
            "messages": [
                {"role": "system", "content": CHAPTER_REVIEW_PROMPT},
                {"role": "user", "content": review_input},
            ],
            "temperature": 0,
            "max_tokens": config["max_tokens"],
        }
        apply_thinking_setting(
            payload, config["provider"], config.get("enable_thinking")
        )
        status, data = post_json(
            config["endpoint"], payload, config["api_key"]
        )
        return ApiResult(
            provider=config["provider"],
            model=config["model"],
            mode=(
                "chapter-translation-verifier-thinking"
                if config.get("enable_thinking")
                else "chapter-translation-verifier-non-thinking"
            ),
            status=status,
            elapsed_seconds=round(time.monotonic() - started, 2),
            usage=data.get("usage", {}),
            content=chat_content(data),
        )
    except Exception as error:
        return ApiResult(
            provider=config["provider"],
            model=config["model"],
            mode=(
                "chapter-translation-verifier-thinking"
                if config.get("enable_thinking")
                else "chapter-translation-verifier-non-thinking"
            ),
            status=None,
            elapsed_seconds=round(time.monotonic() - started, 2),
            usage={},
            content="",
            error=f"{type(error).__name__}: {error}",
        )


def deepseek_regions(content: str) -> list[dict[str, Any]]:
    pattern = re.compile(
        r"<\|ref\|>(.*?)<\|/ref\|>\s*<\|det\|>(.*?)<\|/det\|>",
        re.DOTALL,
    )
    regions: list[dict[str, Any]] = []
    for text, raw_box in pattern.findall(content):
        box: Any = raw_box.strip()
        try:
            box = json.loads(box)
        except json.JSONDecodeError:
            pass
        regions.append({"text": text.strip(), "box": box})
    return regions


def numbered_item_count(content: str) -> int:
    return len(re.findall(r"(?m)^\s*\d+\.\s+", content))


def without_list_markers(content: str) -> str:
    return re.sub(r"(?m)^\s*\d+\.\s+", "", content)


def numeric_token_counts(content: str) -> collections.Counter[str]:
    content = without_list_markers(content)
    values = re.findall(r"(?<![A-Za-z])\d+(?:,\d{3})*(?:\.\d+)?", content)
    return collections.Counter(value.replace(",", "") for value in values)


def numeric_tokens(content: str) -> set[str]:
    values = numeric_token_counts(content)
    return {value.replace(",", "") for value in values}


def word_tokens(content: str) -> set[str]:
    content = without_list_markers(content).lower().replace("’", "'")
    return set(re.findall(r"[a-z]+(?:'[a-z]+)?|\d+(?:,\d{3})*", content))


def normalize_text(content: str) -> str:
    content = without_list_markers(content)
    content = content.replace("’", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", content).strip().lower()


def extract_json_object(content: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", content):
        try:
            value, _ = decoder.raw_decode(content[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("No valid JSON object found")


def parse_study_result(content: str) -> dict[str, Any]:
    value = extract_json_object(content)
    raw_regions = value.get("regions")
    if not isinstance(raw_regions, list):
        raise ValueError("study JSON must contain a regions array")
    regions: list[dict[str, str]] = []
    for index, raw_region in enumerate(raw_regions, start=1):
        if not isinstance(raw_region, dict):
            raise ValueError(f"region {index} is not an object")
        region_id = str(raw_region.get("id", "")).strip()
        region_type = str(raw_region.get("type", "")).strip().lower()
        source_text = str(raw_region.get("source_text", "")).strip()
        regions.append(
            {
                "id": region_id,
                "type": region_type,
                "source_text": source_text,
            }
        )
    skipped = value.get("skipped", {})
    if not isinstance(skipped, dict):
        skipped = {}
    return {"regions": regions, "skipped": skipped}


def parse_translation_result(content: str) -> dict[str, str]:
    value = extract_json_object(content)
    raw_translations = value.get("translations")
    if not isinstance(raw_translations, list):
        raise ValueError("translation JSON must contain a translations array")
    translations: dict[str, str] = {}
    for index, item in enumerate(raw_translations, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"translation {index} is not an object")
        region_id = str(item.get("id", "")).strip()
        translation = str(item.get("translation", "")).strip()
        if not region_id or not translation:
            raise ValueError(f"translation {index} has an empty id or text")
        translations[region_id] = translation
    return translations


def parse_translation_verification(
    content: str,
    expected_ids: set[str],
) -> list[dict[str, str]]:
    value = extract_json_object(content)
    raw_issues = value.get("issues")
    if not isinstance(raw_issues, list):
        raise ValueError("Translation verifier JSON must contain an issues array")
    allowed_categories = {
        "meaning",
        "math_term",
        "wordplay",
        "name",
        "number",
        "omission",
        "addition",
        "fluency",
    }
    issues: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for raw in raw_issues:
        if not isinstance(raw, dict):
            raise ValueError("Translation verifier issue must be an object")
        region_id = str(raw.get("id", "")).strip()
        category = str(raw.get("category", "")).strip()
        explanation = str(raw.get("explanation", "")).strip()
        suggestion = str(raw.get("suggested_translation", "")).strip()
        if region_id not in expected_ids:
            raise ValueError(f"Translation verifier returned unknown ID: {region_id}")
        if category not in allowed_categories:
            raise ValueError(
                f"Translation verifier returned invalid category: {category}"
            )
        if not explanation:
            raise ValueError("Translation verifier issue is missing an explanation")
        key = (region_id, category)
        if key in seen:
            continue
        seen.add(key)
        issues.append(
            {
                "id": region_id,
                "category": category,
                "explanation": explanation,
                "suggested_translation": suggestion,
            }
        )
    return issues


def compare_translation_verifier(
    regions: list[dict[str, str]],
    verifier: ApiResult | None,
) -> dict[str, Any]:
    if verifier is None:
        return {"available": False, "error": None, "issues": []}
    if verifier.error:
        return {
            "available": True,
            "error": verifier.error,
            "issues": [],
        }
    try:
        issues = parse_translation_verification(
            verifier.content, {region["id"] for region in regions}
        )
    except ValueError as error:
        return {
            "available": True,
            "error": f"invalid_translation_verifier_json: {error}",
            "issues": [],
        }
    return {"available": True, "error": None, "issues": issues}


def parse_chapter_review(
    content: str,
    valid_regions: set[tuple[int, str]],
) -> dict[str, Any]:
    value = extract_json_object(content)
    raw_issues = value.get("issues")
    raw_consistency = value.get("global_consistency")
    if not isinstance(raw_issues, list) or not isinstance(raw_consistency, list):
        raise ValueError(
            "Chapter review JSON must contain issues and global_consistency arrays"
        )
    allowed_categories = {
        "meaning",
        "math_term",
        "wordplay",
        "name",
        "number",
        "omission",
        "addition",
        "fluency",
    }
    issues: list[dict[str, Any]] = []
    seen: set[tuple[int, str, str]] = set()
    for raw in raw_issues:
        if not isinstance(raw, dict):
            raise ValueError("Chapter review issue must be an object")
        try:
            page = int(raw.get("page"))
        except (TypeError, ValueError) as error:
            raise ValueError("Chapter review issue has an invalid page") from error
        region_id = str(raw.get("id", "")).strip()
        category = str(raw.get("category", "")).strip()
        suggestion = str(raw.get("suggested_translation", "")).strip()
        reason = str(raw.get("reason", "")).strip()
        if (page, region_id) not in valid_regions:
            raise ValueError(
                f"Chapter review returned unknown region: page {page} {region_id}"
            )
        if category not in allowed_categories:
            raise ValueError(
                f"Chapter review returned invalid category: {category}"
            )
        if not reason:
            raise ValueError("Chapter review issue is missing a reason")
        key = (page, region_id, category)
        if key in seen:
            continue
        seen.add(key)
        issues.append(
            {
                "page": page,
                "id": region_id,
                "category": category,
                "suggested_translation": suggestion,
                "reason": reason,
            }
        )

    consistency: list[dict[str, Any]] = []
    for raw in raw_consistency:
        if not isinstance(raw, dict):
            raise ValueError("Global consistency finding must be an object")
        affected: list[dict[str, Any]] = []
        for item in raw.get("affected", []):
            if not isinstance(item, dict):
                continue
            try:
                page = int(item.get("page"))
            except (TypeError, ValueError):
                continue
            region_id = str(item.get("id", "")).strip()
            if (page, region_id) in valid_regions:
                affected.append({"page": page, "id": region_id})
        consistency.append(
            {
                "category": str(raw.get("category", "")).strip(),
                "item": str(raw.get("item", "")).strip(),
                "preferred_translation": str(
                    raw.get("preferred_translation", "")
                ).strip(),
                "reason": str(raw.get("reason", "")).strip(),
                "affected": affected,
            }
        )
    return {"issues": issues, "global_consistency": consistency}


def study_source_text(parsed: dict[str, Any]) -> str:
    return "\n".join(region["source_text"] for region in parsed["regions"])


def layout_has_probable_study_text(layout: ApiResult) -> bool:
    regions = deepseek_regions(layout.content)
    texts = [region["text"] for region in regions]
    if not texts:
        texts = [layout.content]
    for text in texts:
        words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)
        if len(words) >= 2:
            return True
        if len(words) == 1 and words[0].lower() in {
            "contents",
            "chapter",
            "example",
            "definition",
            "solution",
        }:
            return True
    return False


def analyze_study_risk(layout: ApiResult, primary: ApiResult) -> dict[str, Any]:
    flags: list[str] = []
    regions = deepseek_regions(layout.content)
    parsed: dict[str, Any] | None = None
    parse_error: str | None = None
    if layout.error:
        flags.append("layout_api_error")
    if primary.error or not primary.content.strip():
        flags.append("primary_api_error_or_empty")
    else:
        try:
            parsed = parse_study_result(primary.content)
        except ValueError as error:
            parse_error = str(error)
            flags.append("invalid_study_json")

    if parsed is not None:
        ids = [region["id"] for region in parsed["regions"]]
        if any(not region_id for region_id in ids):
            flags.append("empty_region_id")
        if len(ids) != len(set(ids)):
            flags.append("duplicate_region_id")
        if any(not region["source_text"] for region in parsed["regions"]):
            flags.append("empty_source_text")
        if any(region["type"] not in STUDY_TYPES for region in parsed["regions"]):
            flags.append("unknown_study_region_type")
        if any("[uncertain]" in region["source_text"].lower() for region in parsed["regions"]):
            flags.append("primary_uncertain")
        if not parsed["regions"] and layout_has_probable_study_text(layout):
            flags.append("possible_translatable_text_omission")

    return {
        "flags": flags,
        "deepseek_region_count": len(regions),
        "primary_item_count": len(parsed["regions"]) if parsed is not None else 0,
        "study_json_valid": parsed is not None,
        "study_json_error": parse_error,
        "ignored_checks": [
            "sound_effects",
            "pure_numeric_tables",
            "prose_free_formulas",
            "standalone_numeric_diagram_labels",
            "standalone_page_number_footers",
        ],
    }


def analyze_risk(
    layout: ApiResult,
    primary: ApiResult,
    page: int,
    printed_page_offset: int | None,
) -> dict[str, Any]:
    flags: list[str] = []
    regions = deepseek_regions(layout.content)
    item_count = numbered_item_count(primary.content)
    layout_text = "\n".join(region["text"] for region in regions)
    if not layout_text.strip():
        layout_text = re.sub(
            r"<\|det\|>.*?<\|/det\|>", " ", layout.content, flags=re.DOTALL
        )
    layout_numbers = numeric_tokens(layout_text)
    primary_numbers = numeric_tokens(primary.content)
    layout_words = word_tokens(layout_text)
    primary_words = word_tokens(primary.content)

    if layout.error:
        flags.append("layout_api_error")
    if primary.error or not primary.content.strip():
        flags.append("primary_api_error_or_empty")
    if "[uncertain]" in primary.content.lower():
        flags.append("primary_uncertain")
    token_coverage = None
    if layout_words:
        token_coverage = len(layout_words & primary_words) / len(layout_words)
    if token_coverage is not None and token_coverage < 0.70:
        flags.append("possible_region_omission")
    if len(layout_numbers) >= 3 and len(primary_numbers) >= 3:
        smaller = min(len(layout_numbers), len(primary_numbers))
        similarity = len(layout_numbers & primary_numbers) / smaller
        if similarity < 0.75:
            flags.append("numeric_set_disagreement")
    else:
        similarity = None
    expected_printed_page = None
    if printed_page_offset is not None:
        expected_printed_page = page + printed_page_offset
        if re.search(
            rf"\b{expected_printed_page}\s*\+\s*\d", without_list_markers(primary.content)
        ):
            flags.append("suspicious_page_number_merge")
        tail = "\n".join(primary.content.splitlines()[-4:])
        layout_has_page_number = re.search(
            rf"\b{expected_printed_page}\b", layout_text
        )
        if layout_has_page_number and not re.search(rf"\b{expected_printed_page}\b", tail):
            flags.append("expected_page_number_not_in_tail")

    return {
        "flags": flags,
        "deepseek_region_count": len(regions),
        "primary_item_count": item_count,
        "layout_token_coverage_in_primary": token_coverage,
        "layout_numeric_tokens": sorted(layout_numbers, key=lambda x: (len(x), x)),
        "primary_numeric_tokens": sorted(primary_numbers, key=lambda x: (len(x), x)),
        "numeric_set_similarity": similarity,
        "expected_printed_page": expected_printed_page,
    }


def compare_verifier(primary: ApiResult, verifier: ApiResult | None) -> dict[str, Any]:
    if verifier is None:
        return {"available": False}
    primary_text = normalize_text(primary.content)
    verifier_text = normalize_text(verifier.content)
    similarity = difflib.SequenceMatcher(None, primary_text, verifier_text).ratio()
    primary_numbers = numeric_tokens(primary.content)
    verifier_numbers = numeric_tokens(verifier.content)
    primary_counts = numeric_token_counts(primary.content)
    verifier_counts = numeric_token_counts(verifier.content)
    count_differences = {
        token: [primary_counts[token], verifier_counts[token]]
        for token in sorted(primary_counts.keys() | verifier_counts.keys())
        if primary_counts[token] != verifier_counts[token]
    }
    return {
        "available": True,
        "error": verifier.error,
        "text_similarity": round(similarity, 4),
        "primary_only_numbers": sorted(primary_numbers - verifier_numbers),
        "verifier_only_numbers": sorted(verifier_numbers - primary_numbers),
        "numeric_count_differences": count_differences,
        "needs_human_review": (
            bool(verifier.error)
            or similarity < 0.995
            or bool(count_differences)
        ),
    }


def compare_study_verifier(
    primary: ApiResult, verifier: ApiResult | None
) -> dict[str, Any]:
    if verifier is None:
        return {"available": False}
    if verifier.error:
        return {
            "available": True,
            "error": verifier.error,
            "needs_human_review": True,
        }
    try:
        primary_parsed = parse_study_result(primary.content)
        verifier_parsed = parse_study_result(verifier.content)
    except ValueError as error:
        return {
            "available": True,
            "error": f"invalid_study_json: {error}",
            "needs_human_review": True,
        }
    primary_text = normalize_text(study_source_text(primary_parsed))
    verifier_text = normalize_text(study_source_text(verifier_parsed))
    similarity = difflib.SequenceMatcher(None, primary_text, verifier_text).ratio()
    return {
        "available": True,
        "error": None,
        "text_similarity": round(similarity, 4),
        "primary_region_count": len(primary_parsed["regions"]),
        "verifier_region_count": len(verifier_parsed["regions"]),
        "needs_human_review": similarity < 0.97,
    }


def build_study_output(
    primary: ApiResult, translation: ApiResult | None
) -> tuple[dict[str, Any], list[str]]:
    reasons: list[str] = []
    try:
        parsed = parse_study_result(primary.content)
    except ValueError as error:
        return {
            "regions": [],
            "skipped": {},
            "error": f"invalid_study_json: {error}",
        }, ["invalid_study_json"]

    translation_map: dict[str, str] = {}
    if parsed["regions"]:
        if translation is None or translation.error:
            reasons.append("translation_api_error_or_empty")
        else:
            try:
                translation_map = parse_translation_result(translation.content)
            except ValueError:
                reasons.append("invalid_translation_json")

    expected_ids = {region["id"] for region in parsed["regions"]}
    if parsed["regions"] and set(translation_map) != expected_ids:
        reasons.append("translation_id_mismatch")
    output_regions = [
        {
            **region,
            "translation": translation_map.get(region["id"], ""),
        }
        for region in parsed["regions"]
    ]
    for region in output_regions:
        source_length = len(re.sub(r"\s+", "", region["source_text"]))
        translation_length = len(re.sub(r"\s+", "", region["translation"]))
        if (
            source_length >= 15
            and translation_length > max(source_length * 3, source_length + 80)
        ):
            reasons.append("translation_suspicious_expansion")
    return {
        "regions": output_regions,
        "skipped": parsed["skipped"],
        "error": None,
    }, list(dict.fromkeys(reasons))


def collect_review_reasons(
    risk: dict[str, Any], verifier_comparison: dict[str, Any]
) -> list[str]:
    reasons = list(risk["flags"])
    if verifier_comparison.get("available"):
        if verifier_comparison.get("error"):
            reasons.append("verifier_api_error")
        similarity = verifier_comparison.get("text_similarity")
        if similarity is not None and similarity < 0.995:
            reasons.append("primary_verifier_text_disagreement")
        if verifier_comparison.get("numeric_count_differences"):
            reasons.append("primary_verifier_numeric_count_disagreement")
    return list(dict.fromkeys(reasons))


def collect_study_review_reasons(
    risk: dict[str, Any],
    verifier_comparison: dict[str, Any],
    translation_reasons: list[str],
    translation_verifier_comparison: dict[str, Any] | None = None,
) -> list[str]:
    reasons = list(risk["flags"]) + list(translation_reasons)
    if verifier_comparison.get("available"):
        if verifier_comparison.get("error"):
            reasons.append("verifier_api_or_json_error")
        similarity = verifier_comparison.get("text_similarity")
        if similarity is not None and similarity < 0.97:
            reasons.append("primary_verifier_study_text_disagreement")
    if translation_verifier_comparison:
        if translation_verifier_comparison.get("error"):
            reasons.append("translation_verifier_api_or_json_error")
        if translation_verifier_comparison.get("issues"):
            reasons.append("translation_semantic_issue")
    return list(dict.fromkeys(reasons))


def page_markdown(record: dict[str, Any]) -> str:
    if record.get("mode") == "study":
        return study_page_markdown(record)
    page = record["page"]
    risk = record["risk"]
    verifier_compare = record["verifier_comparison"]
    flags = ", ".join(risk["flags"]) or "none"
    reasons = ", ".join(record.get("review_reasons", [])) or "none"
    parts = [
        f"# PDF第{page}页",
        "",
        f"- 需要人工复核：**{'是' if record['review_required'] else '否'}**",
        f"- 风险标记：`{flags}`",
        f"- 复核原因：`{reasons}`",
        f"- DeepSeek文字区域数：{risk['deepseek_region_count']}",
        f"- 主OCR编号项目数：{risk['primary_item_count']}",
        "",
        "## 主OCR - qwen3.6-flash思考模式",
        "",
        record["primary"]["content"] or "[no output]",
        "",
        "## 版面OCR - DeepSeek-OCR精细定位模式",
        "",
        record["layout"]["content"] or "[no output]",
    ]
    if record.get("verifier"):
        parts.extend(
            [
                "",
                "## 独立复核 - qwen3.7-max非思考模式",
                "",
                record["verifier"]["content"] or "[no output]",
                "",
                "## 复核比较",
                "",
                "```json",
                json.dumps(verifier_compare, ensure_ascii=False, indent=2),
                "```",
            ]
        )
    parts.append("")
    return "\n".join(parts)


def study_page_markdown(record: dict[str, Any]) -> str:
    page = record["page"]
    reasons = ", ".join(record.get("review_reasons", [])) or "none"
    parts = [
        f"# PDF第{page}页 - 学习翻译",
        "",
        f"- 需要人工复核：**{'是' if record['review_required'] else '否'}**",
        f"- 复核原因：`{reasons}`",
        f"- 翻译条目数：{len(record.get('study', {}).get('regions', []))}",
        "",
    ]
    for index, region in enumerate(record.get("study", {}).get("regions", []), start=1):
        parts.extend(
            [
                f"## {index}. {region['type']} ({region['id']})",
                "",
                f"**英文：** {region['source_text']}",
                "",
                f"**中文：** {region['translation'] or '[translation missing]'}",
                "",
            ]
        )
    if not record.get("study", {}).get("regions"):
        parts.extend(["本页没有需要翻译的英文自然语言。", ""])
    if record.get("study", {}).get("skipped"):
        parts.extend(
            [
                "## 已忽略内容",
                "",
                "```json",
                json.dumps(record["study"]["skipped"], ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        )
    translation_comparison = record.get("translation_verifier_comparison", {})
    issues = translation_comparison.get("issues", [])
    if issues:
        parts.extend(
            [
                "## 翻译语义复核建议（未自动改写）",
                "",
                "```json",
                json.dumps(issues, ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        )
    return "\n".join(parts)


def process_page(args: argparse.Namespace, page: int) -> dict[str, Any]:
    pages_dir = args.output / "pages"
    image_path = pages_dir / f"page-{page:04d}.png"
    json_path = pages_dir / f"page-{page:04d}.json"
    if (
        args.resume
        and json_path.exists()
        and page not in args.redo_pages
    ):
        return json.loads(json_path.read_text(encoding="utf-8"))

    render_page(
        args.pdf,
        page,
        image_path,
        args.long_edge,
        args.resume,
        args.pdftoppm_command,
    )
    image_url = image_data_url(image_path)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        layout_profile = args.layout_profile_config
        primary_profile = args.primary_profile_config
        layout_future = executor.submit(
            call_deepseek,
            image_url,
            layout_profile["api_key"],
            layout_profile["model"],
            layout_profile["chat_endpoint"],
            layout_profile["provider"],
            layout_profile["max_tokens"],
        )
        primary_future = executor.submit(
            call_qwen_primary,
            image_url,
            primary_profile["api_key"],
            primary_profile["model"],
            STUDY_PRIMARY_PROMPT if args.mode == "study" else PRIMARY_PROMPT,
            primary_profile["chat_endpoint"],
            primary_profile["provider"],
            primary_profile["enable_thinking"],
            primary_profile["max_tokens"],
        )
        layout = layout_future.result()
        primary = primary_future.result()

    if args.mode == "study":
        risk = analyze_study_risk(layout, primary)
    else:
        risk = analyze_risk(layout, primary, page, args.printed_page_offset)
    should_verify = args.verify == "always" or (
        args.verify == "auto" and bool(risk["flags"])
    )
    verifier = None
    if should_verify:
        verifier_profile = args.verifier_profile_config
        verifier = call_qwen_verifier(
            image_url,
            verifier_profile["api_key"],
            verifier_profile["model"],
            STUDY_VERIFY_PROMPT if args.mode == "study" else VERIFY_PROMPT,
            verifier_profile["responses_endpoint"],
            verifier_profile["provider"],
            verifier_profile["enable_thinking"],
            verifier_profile["max_tokens"],
        )
    translation = None
    translation_verifier = None
    study = None
    if args.mode == "study":
        try:
            parsed = parse_study_result(primary.content)
        except ValueError:
            parsed = {"regions": []}
        if parsed["regions"]:
            translation = call_translation(args, parsed["regions"])
        study, translation_reasons = build_study_output(primary, translation)
        if (
            args.translation_verify == "always"
            and study["regions"]
            and not translation_reasons
        ):
            translation_verifier = call_translation_verifier(
                args, study["regions"]
            )
        translation_verifier_comparison = compare_translation_verifier(
            study["regions"], translation_verifier
        )
        verifier_comparison = compare_study_verifier(primary, verifier)
        review_reasons = collect_study_review_reasons(
            risk,
            verifier_comparison,
            translation_reasons,
            translation_verifier_comparison,
        )
    else:
        verifier_comparison = compare_verifier(primary, verifier)
        review_reasons = collect_review_reasons(risk, verifier_comparison)
    review_required = bool(review_reasons)

    record = {
        "page": page,
        "mode": args.mode,
        "image": str(image_path.relative_to(args.output)),
        "layout": layout.as_dict(),
        "primary": primary.as_dict(),
        "risk": risk,
        "verifier": verifier.as_dict() if verifier else None,
        "translation": translation.as_dict() if translation else None,
        "translation_verifier": (
            translation_verifier.as_dict() if translation_verifier else None
        ),
        "translation_verifier_comparison": (
            translation_verifier_comparison if args.mode == "study" else None
        ),
        "study": study,
        "verifier_comparison": verifier_comparison,
        "review_reasons": review_reasons,
        "review_required": review_required,
    }
    write_page_json(json_path, record)
    with print_lock:
        print(
            json.dumps(
                {
                    "page": page,
                    "primary_seconds": primary.elapsed_seconds,
                    "layout_seconds": layout.elapsed_seconds,
                    "verified": verifier is not None,
                    "translated": translation is not None,
                    "flags": risk["flags"],
                    "review_required": review_required,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return record


def reanalyze_page(args: argparse.Namespace, page: int) -> dict[str, Any]:
    pages_dir = args.output / "pages"
    json_path = pages_dir / f"page-{page:04d}.json"
    if not json_path.exists():
        raise ValueError(f"Existing page result not found: {json_path}")
    record = json.loads(json_path.read_text(encoding="utf-8"))
    layout = ApiResult(**record["layout"])
    primary = ApiResult(**record["primary"])
    verifier = ApiResult(**record["verifier"]) if record.get("verifier") else None
    mode = record.get("mode", "exact")
    if mode == "study":
        translation = (
            ApiResult(**record["translation"]) if record.get("translation") else None
        )
        record["risk"] = analyze_study_risk(layout, primary)
        record["study"], translation_reasons = build_study_output(
            primary, translation
        )
        record["verifier_comparison"] = compare_study_verifier(primary, verifier)
        translation_verifier = (
            ApiResult(**record["translation_verifier"])
            if record.get("translation_verifier")
            else None
        )
        record["translation_verifier_comparison"] = compare_translation_verifier(
            record["study"]["regions"], translation_verifier
        )
        record["review_reasons"] = collect_study_review_reasons(
            record["risk"],
            record["verifier_comparison"],
            translation_reasons,
            record["translation_verifier_comparison"],
        )
    else:
        record["risk"] = analyze_risk(
            layout, primary, page, args.printed_page_offset
        )
        record["verifier_comparison"] = compare_verifier(primary, verifier)
        record["review_reasons"] = collect_review_reasons(
            record["risk"], record["verifier_comparison"]
        )
    record["review_required"] = bool(record["review_reasons"])
    write_page_json(json_path, record)
    return record


def retranslate_page(args: argparse.Namespace, page: int) -> dict[str, Any]:
    pages_dir = args.output / "pages"
    json_path = pages_dir / f"page-{page:04d}.json"
    if not json_path.exists():
        raise ValueError(f"Existing page result not found: {json_path}")
    record = json.loads(json_path.read_text(encoding="utf-8"))
    if record.get("mode") != "study":
        raise ValueError(f"Page {page} is not a study-mode result")
    primary = ApiResult(**record["primary"])
    try:
        parsed = parse_study_result(primary.content)
    except ValueError as error:
        raise ValueError(f"Page {page} has invalid study JSON: {error}") from error
    translation = None
    if parsed["regions"]:
        translation = call_translation(args, parsed["regions"])
    record["translation"] = translation.as_dict() if translation else None
    record["study"], translation_reasons = build_study_output(primary, translation)
    translation_verifier = None
    if (
        args.translation_verify == "always"
        and record["study"]["regions"]
        and not translation_reasons
    ):
        translation_verifier = call_translation_verifier(
            args, record["study"]["regions"]
        )
    record["translation_verifier"] = (
        translation_verifier.as_dict() if translation_verifier else None
    )
    record["translation_verifier_comparison"] = compare_translation_verifier(
        record["study"]["regions"], translation_verifier
    )
    layout = ApiResult(**record["layout"])
    verifier = ApiResult(**record["verifier"]) if record.get("verifier") else None
    record["risk"] = analyze_study_risk(layout, primary)
    record["verifier_comparison"] = compare_study_verifier(primary, verifier)
    record["review_reasons"] = collect_study_review_reasons(
        record["risk"],
        record["verifier_comparison"],
        translation_reasons,
        record["translation_verifier_comparison"],
    )
    record["review_required"] = bool(record["review_reasons"])
    write_page_json(json_path, record)
    with print_lock:
        print(
            json.dumps(
                {
                    "page": page,
                    "retranslated": True,
                    "translation_seconds": (
                        translation.elapsed_seconds if translation else 0
                    ),
                    "review_required": record["review_required"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return record


def reverify_translation_page(
    args: argparse.Namespace, page: int
) -> dict[str, Any]:
    """Review saved translations without repeating OCR or translation."""

    pages_dir = args.output / "pages"
    json_path = pages_dir / f"page-{page:04d}.json"
    if not json_path.exists():
        raise ValueError(f"Existing page result not found: {json_path}")
    record = json.loads(json_path.read_text(encoding="utf-8"))
    if record.get("mode") != "study":
        raise ValueError(f"Page {page} is not a study-mode result")

    regions = record.get("study", {}).get("regions", [])
    missing_ids = [region["id"] for region in regions if not region.get("translation")]
    if missing_ids:
        raise ValueError(
            f"Page {page} has missing translations ({', '.join(missing_ids)}); "
            "run --retranslate first"
        )

    translation_verifier = None
    if regions:
        translation_verifier = call_translation_verifier(args, regions)
    record["translation_verifier"] = (
        translation_verifier.as_dict() if translation_verifier else None
    )
    record["translation_verifier_comparison"] = compare_translation_verifier(
        regions, translation_verifier
    )

    layout = ApiResult(**record["layout"])
    primary = ApiResult(**record["primary"])
    verifier = ApiResult(**record["verifier"]) if record.get("verifier") else None
    translation = (
        ApiResult(**record["translation"]) if record.get("translation") else None
    )
    _, translation_reasons = build_study_output(primary, translation)
    record["risk"] = analyze_study_risk(layout, primary)
    record["verifier_comparison"] = compare_study_verifier(primary, verifier)
    record["review_reasons"] = collect_study_review_reasons(
        record["risk"],
        record["verifier_comparison"],
        translation_reasons,
        record["translation_verifier_comparison"],
    )
    record["review_required"] = bool(record["review_reasons"])
    write_page_json(json_path, record)
    with print_lock:
        print(
            json.dumps(
                {
                    "page": page,
                    "translation_reverified": True,
                    "verifier_seconds": (
                        translation_verifier.elapsed_seconds
                        if translation_verifier
                        else 0
                    ),
                    "semantic_issues": len(
                        record["translation_verifier_comparison"].get(
                            "issues", []
                        )
                    ),
                    "review_required": record["review_required"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return record


def merge_with_existing_records(
    output: pathlib.Path,
    updated_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge subset maintenance results into the existing aggregate page set."""

    manifest_path = output / "manifest.json"
    existing_pages: list[int] = []
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            existing_pages = [int(page) for page in manifest.get("pages", [])]
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            existing_pages = []
    if not existing_pages:
        for path in sorted((output / "pages").glob("page-*.json")):
            match = re.fullmatch(r"page-(\d+)\.json", path.name)
            if match:
                existing_pages.append(int(match.group(1)))

    by_page = {record["page"]: record for record in updated_records}
    for page in existing_pages:
        if page in by_page:
            continue
        path = output / "pages" / f"page-{page:04d}.json"
        if not path.is_file():
            raise ValueError(f"Aggregate page result is missing: {path}")
        by_page[page] = json.loads(path.read_text(encoding="utf-8"))
    return [by_page[page] for page in sorted(by_page)]


def load_existing_page_records(
    output: pathlib.Path,
    pages: list[int],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for page in pages:
        path = output / "pages" / f"page-{page:04d}.json"
        if not path.is_file():
            raise ValueError(f"Existing page result not found: {path}")
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("mode") != "study":
            raise ValueError(f"Page {page} is not a study-mode result")
        regions = record.get("study", {}).get("regions", [])
        missing = [region["id"] for region in regions if not region.get("translation")]
        if missing:
            raise ValueError(
                f"Page {page} has missing translations ({', '.join(missing)}); "
                "run --retranslate first"
            )
        records.append(record)
    return records


def discover_existing_study_pages(output: pathlib.Path) -> list[int]:
    manifest_path = output / "manifest.json"
    pages: list[int] = []
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            pages = sorted({int(page) for page in manifest.get("pages", [])})
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pages = []
    if not pages:
        for path in sorted((output / "pages").glob("page-*.json")):
            match = re.fullmatch(r"page-(\d+)\.json", path.name)
            if match:
                pages.append(int(match.group(1)))
    if not pages:
        raise ValueError(f"No existing study pages found in: {output}")
    return sorted(set(pages))


def compact_page_label(pages: list[int]) -> str:
    ranges: list[str] = []
    start = previous = pages[0]
    for page in pages[1:]:
        if page == previous + 1:
            previous = page
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = page
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return "_".join(ranges)


def scope_backfill_paths_for_subset(
    settings: dict[str, Any],
    pages: list[int],
    all_pages: list[int],
) -> dict[str, Any]:
    if pages == all_pages:
        return settings
    scoped = dict(settings)
    tag = f"-pages-{compact_page_label(pages)}"
    for key in ("output_pdf", "final_translation", "plan", "report", "spotcheck"):
        path = scoped[key]
        scoped[key] = path.with_name(f"{path.stem}{tag}{path.suffix}")
    return scoped


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_adjudication_inputs(
    output: pathlib.Path,
    adjudication: dict[str, Any],
) -> None:
    expected = adjudication.get("inputs", {})
    if not isinstance(expected, dict):
        raise ValueError("Adjudication inputs must be an object")
    input_files = expected.get("files")
    if not isinstance(input_files, dict) or not input_files:
        raise ValueError("Adjudication inputs.files must be a non-empty object")
    for filename, raw_hash in input_files.items():
        filename = str(filename)
        if pathlib.Path(filename).name != filename:
            raise ValueError(
                f"Adjudication input filename must not contain a path: {filename}"
            )
        expected_hash = str(raw_hash).strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
            raise ValueError(
                f"Adjudication has an invalid SHA-256 for {filename}"
            )
        path = output / filename
        if not path.is_file():
            raise ValueError(f"Adjudication source file is missing: {path}")
        if sha256_file(path) != expected_hash:
            raise ValueError(
                f"Adjudication source changed after review: {filename}"
            )


def apply_chapter_adjudication(
    records: list[dict[str, Any]],
    adjudication: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if adjudication.get("schema_version") != 1:
        raise ValueError("Unsupported adjudication schema_version")
    raw_decisions = adjudication.get("decisions")
    if not isinstance(raw_decisions, list):
        raise ValueError("Adjudication decisions must be an array")
    human_review = adjudication.get("human_review", [])
    if not isinstance(human_review, list):
        raise ValueError("Adjudication human_review must be an array")

    revised = copy.deepcopy(records)
    region_map = {
        (record["page"], region["id"]): region
        for record in revised
        for region in record.get("study", {}).get("regions", [])
    }
    allowed = {"replace", "normalize", "discard"}
    seen: set[tuple[int, str]] = set()
    applied: list[dict[str, Any]] = []
    for raw in raw_decisions:
        if not isinstance(raw, dict):
            raise ValueError("Each adjudication decision must be an object")
        try:
            coordinate = (int(raw["page"]), str(raw["id"]))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Adjudication decision has invalid page or id") from error
        if coordinate in seen:
            raise ValueError(
                f"Duplicate adjudication decision: p{coordinate[0]}-{coordinate[1]}"
            )
        seen.add(coordinate)
        if coordinate not in region_map:
            raise ValueError(
                f"Adjudication references unknown region: p{coordinate[0]}-{coordinate[1]}"
            )
        decision = str(raw.get("decision", ""))
        if decision not in allowed:
            raise ValueError(
                f"Invalid adjudication decision for p{coordinate[0]}-{coordinate[1]}: {decision}"
            )
        final_translation = str(raw.get("final_translation", "")).strip()
        if not final_translation:
            raise ValueError(
                f"Adjudication has empty final translation: p{coordinate[0]}-{coordinate[1]}"
            )
        region = region_map[coordinate]
        original = str(region.get("translation", ""))
        if decision == "discard" and final_translation != original:
            raise ValueError(
                f"Discard decision must preserve current translation: p{coordinate[0]}-{coordinate[1]}"
            )
        if decision != "discard":
            region["translation"] = final_translation
        metadata = {
            "decision": decision,
            "reason": str(raw.get("reason", "")).strip(),
            "confidence": raw.get("confidence"),
            "child_note": str(raw.get("child_note", "")).strip(),
        }
        region["codex_adjudication"] = metadata
        applied.append(
            {
                "page": coordinate[0],
                "id": coordinate[1],
                "decision": decision,
                "original_translation": original,
                "final_translation": str(region["translation"]),
                **metadata,
            }
        )

    human_coordinates: list[dict[str, Any]] = []
    for item in human_review:
        if not isinstance(item, dict):
            raise ValueError("Each human_review item must be an object")
        coordinate = (int(item["page"]), str(item["id"]))
        if coordinate not in region_map:
            raise ValueError(
                f"Human review references unknown region: p{coordinate[0]}-{coordinate[1]}"
            )
        human_coordinates.append(item)

    for record in revised:
        record["review_required"] = any(
            int(item["page"]) == record["page"] for item in human_coordinates
        )
        record["review_reasons"] = (
            ["codex_adjudication_needs_human"] if record["review_required"] else []
        )
    application = {
        "schema_version": 1,
        "source_adjudication": "chapter_review-codex-adjudication.json",
        "applied_count": len(applied),
        "human_review_count": len(human_coordinates),
        "changes": applied,
        "human_review": human_coordinates,
    }
    return revised, application


def chapter_name_candidates(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    occurrences: dict[str, list[tuple[int, str, str]]] = collections.defaultdict(list)
    for record in records:
        for region in record.get("study", {}).get("regions", []):
            source = region["source_text"]
            for name in CHAPTER_NAME_TERMS:
                if re.search(rf"(?<![A-Za-z]){re.escape(name)}(?![A-Za-z])", source, re.I):
                    occurrences[name].append(
                        (record["page"], region["id"], region["translation"])
                    )
            for match in re.finditer(r"\bProfessor\s+[A-Z][A-Za-z'-]+\b", source):
                name = match.group(0)
                if name in CHAPTER_NAME_TERMS:
                    continue
                occurrences[name].append(
                    (record["page"], region["id"], region["translation"])
                )
    candidates: list[dict[str, Any]] = []
    for name, items in sorted(occurrences.items()):
        candidates.append(
            {
                "name": name,
                "count": len(items),
                "locations": [f"p{page}-{region_id}" for page, region_id, _ in items],
                "translation_examples": list(
                    dict.fromkeys(text[:100] for _, _, text in items)
                )[:3],
            }
        )
    return candidates


def build_chapter_review_input(
    args: argparse.Namespace,
    records: list[dict[str, Any]],
) -> str:
    lines = [
        "# 整章翻译盲审输入",
        "",
        "> 本文件只包含翻译规则、压缩对照表和已提取的英中文本。",
        "> 不包含页面图片，也不包含既有逐页复核的候选问题。",
        "",
        "## 复核规则",
        "",
        "- 只报告确认存在的译义、数学术语、双关、名称、数字、遗漏或一致性问题。",
        "- 可接受的风格差异不应报告。分析后认为没问题的条目必须省略。",
        "- 不解题，不增补原文信息，不自动改写任何已保存译文。",
        "- 使用每条的页码和区域ID精确定位。",
        "",
        "## 固定术语表",
        "",
        "| English | 首选中文 |",
        "|---|---|",
    ]
    lines.extend(f"| {english} | {chinese} |" for english, chinese in CHAPTER_TERMINOLOGY)
    if args.deepl_custom_instructions:
        lines.extend(["", "## 已配置翻译约束", ""])
        lines.extend(f"- {item}" for item in args.deepl_custom_instructions)

    lines.extend(["", "## 角色名与专有名称候选", ""])
    names = chapter_name_candidates(records)
    if names:
        lines.extend(
            [
                "| English | 出现次数 | 位置 | 当前译文示例 |",
                "|---|---:|---|---|",
            ]
        )
        for item in names:
            examples = " / ".join(item["translation_examples"]).replace("|", "\\|")
            lines.append(
                f"| {item['name']} | {item['count']} | "
                f"{', '.join(item['locations'])} | {examples} |"
            )
    else:
        lines.append("无自动提取候选。")

    headings = [
        (record["page"], region)
        for record in records
        for region in record.get("study", {}).get("regions", [])
        if region.get("type") in {"heading", "contents"}
    ]
    lines.extend(["", "## 标题与目录对照", ""])
    if headings:
        lines.extend(["| 位置 | English | 当前译文 |", "|---|---|---|"])
        for page, region in headings:
            source = region["source_text"].replace("|", "\\|").replace("\n", " ")
            translation = region["translation"].replace("|", "\\|").replace("\n", " ")
            lines.append(f"| p{page}-{region['id']} | {source} | {translation} |")
    else:
        lines.append("无标题或目录条目。")

    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = collections.defaultdict(list)
    for record in records:
        for region in record.get("study", {}).get("regions", []):
            key = re.sub(r"\s+", " ", region["source_text"].strip().lower())
            grouped[key].append((record["page"], region))
    repeated = [items for items in grouped.values() if len(items) > 1]
    lines.extend(["", "## 重复表达对照", ""])
    if repeated:
        lines.extend(["| English | 位置 | 当前译文 |", "|---|---|---|"])
        for items in sorted(repeated, key=lambda value: (-len(value), value[0][0])):
            source = items[0][1]["source_text"].replace("|", "\\|").replace("\n", " ")
            locations = ", ".join(f"p{page}-{region['id']}" for page, region in items)
            translations = " / ".join(
                dict.fromkeys(region["translation"] for _, region in items)
            ).replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {source} | {locations} | {translations} |")
    else:
        lines.append("无完全重复的英文表达。")

    lines.extend(["", "## 逐页英中译文", ""])
    for record in records:
        lines.extend([f"### PDF 第 {record['page']} 页", ""])
        for region in record.get("study", {}).get("regions", []):
            lines.extend(
                [
                    f"#### [p{record['page']}-{region['id']}] {region['type']}",
                    "",
                    f"English: {region['source_text']}",
                    "",
                    f"Chinese: {region['translation']}",
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def chapter_review_report(
    records: list[dict[str, Any]],
    parsed: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    region_map = {
        (record["page"], region["id"]): region
        for record in records
        for region in record.get("study", {}).get("regions", [])
    }
    prior_keys = {
        (record["page"], issue["id"])
        for record in records
        for issue in (record.get("translation_verifier_comparison") or {}).get(
            "issues", []
        )
    }
    chapter_keys = {(issue["page"], issue["id"]) for issue in parsed["issues"]}
    comparison = {
        "prior_candidate_regions": len(prior_keys),
        "chapter_candidate_regions": len(chapter_keys),
        "overlap_regions": len(prior_keys & chapter_keys),
        "chapter_only_regions": len(chapter_keys - prior_keys),
        "prior_only_regions": len(prior_keys - chapter_keys),
    }
    result = {**parsed, "comparison": comparison}
    lines = [
        "# 整章翻译盲审结果",
        "",
        f"- 整章确认问题：{len(parsed['issues'])}",
        f"- 全局一致性项：{len(parsed['global_consistency'])}",
        f"- 与逐页候选重合的区域：{comparison['overlap_regions']}",
        f"- 整章新增区域：{comparison['chapter_only_regions']}",
        f"- 仅逐页复核报告的区域：{comparison['prior_only_regions']}",
        "",
        "## 确认问题",
        "",
        "| 位置 | 类别 | English | 当前译文 | 建议译文 | 理由 |",
        "|---|---|---|---|---|---|",
    ]
    for issue in parsed["issues"]:
        region = region_map[(issue["page"], issue["id"])]
        cells = [
            f"p{issue['page']}-{issue['id']}",
            issue["category"],
            region["source_text"],
            region["translation"],
            issue["suggested_translation"] or "[未提供]",
            issue["reason"],
        ]
        lines.append(
            "| " + " | ".join(str(cell).replace("|", "\\|").replace("\n", " ") for cell in cells) + " |"
        )
    lines.extend(["", "## 全局一致性", ""])
    if parsed["global_consistency"]:
        for item in parsed["global_consistency"]:
            affected = ", ".join(
                f"p{entry['page']}-{entry['id']}" for entry in item["affected"]
            ) or "[未定位]"
            lines.extend(
                [
                    f"### {item['category']}: {item['item']}",
                    "",
                    f"- 首选译法：{item['preferred_translation'] or '[未提供]'}",
                    f"- 影响位置：{affected}",
                    f"- 理由：{item['reason']}",
                    "",
                ]
            )
    else:
        lines.append("未报告全局一致性问题。")
    return result, "\n".join(lines).rstrip() + "\n"


def chapter_model_comparison(
    reports: dict[str, dict[str, Any]],
    failures: dict[str, str],
) -> tuple[dict[str, Any], str]:
    region_sets = {
        name: {(item["page"], item["id"]) for item in report["issues"]}
        for name, report in reports.items()
    }
    pairwise: list[dict[str, Any]] = []
    names = sorted(region_sets)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            left_set = region_sets[left]
            right_set = region_sets[right]
            pairwise.append(
                {
                    "left": left,
                    "right": right,
                    "overlap": len(left_set & right_set),
                    "left_only": len(left_set - right_set),
                    "right_only": len(right_set - left_set),
                }
            )
    data = {
        "models": {
            name: {
                "issues": len(report["issues"]),
                "candidate_regions": len(region_sets[name]),
                "global_consistency": len(report["global_consistency"]),
            }
            for name, report in reports.items()
        },
        "pairwise": pairwise,
        "failures": failures,
    }
    lines = [
        "# 整章复核模型对比",
        "",
        "| 配置名 | 问题数 | 候选区域数 | 全局一致性项 |",
        "|---|---:|---:|---:|",
    ]
    for name, summary in data["models"].items():
        lines.append(
            f"| {name} | {summary['issues']} | "
            f"{summary['candidate_regions']} | {summary['global_consistency']} |"
        )
    lines.extend(
        [
            "",
            "## 两两重合",
            "",
            "| 模型A | 模型B | 重合区域 | 仅A | 仅B |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for item in pairwise:
        lines.append(
            f"| {item['left']} | {item['right']} | {item['overlap']} | "
            f"{item['left_only']} | {item['right_only']} |"
        )
    if failures:
        lines.extend(["", "## 失败的模型", ""])
        lines.extend(f"- {name}: {error}" for name, error in failures.items())
    return data, "\n".join(lines).rstrip() + "\n"


def build_summary(args: argparse.Namespace, records: list[dict[str, Any]]) -> None:
    manifest = {
        "source_pdf": str(args.pdf),
        "mode": args.mode,
        "pages": [record["page"] for record in records],
        "long_edge": args.long_edge,
        "pdftoppm_command": args.pdftoppm_command,
        "pdfinfo_command": args.pdfinfo_command,
        "verify_policy": args.verify,
        "printed_page_offset": args.printed_page_offset,
        "models": {
            "layout": args.layout_model,
            "primary": args.primary_model,
            "verifier": args.verifier_model,
            "translation": (
                {
                    "provider": "deepl_mcp",
                    "model": "DeepL Remote MCP",
                    "target_lang": args.deepl_target_lang,
                }
                if args.mode == "study"
                else None
            ),
        },
        "records": records,
    }
    atomic_write_json(args.output / "manifest.json", manifest)

    lines = [
        "# 小批量OCR汇总",
        "",
        f"- 源PDF：`{args.pdf}`",
        f"- 处理模式：`{args.mode}`",
        f"- PDF页码：{', '.join(str(r['page']) for r in records)}",
        f"- 复核策略：`{args.verify}`",
        f"- 需要人工复核：{sum(bool(r['review_required']) for r in records)}/{len(records)}",
        "",
        "| PDF页 | 版面OCR秒数 | 主OCR秒数 | 复核秒数 | 复核原因 | 人工复核 |",
        "|---:|---:|---:|---:|---|:---:|",
    ]
    for record in records:
        verifier = record.get("verifier")
        verifier_seconds = verifier["elapsed_seconds"] if verifier else "-"
        flags = ", ".join(record.get("review_reasons", [])) or "none"
        lines.append(
            f"| {record['page']} | {record['layout']['elapsed_seconds']} | "
            f"{record['primary']['elapsed_seconds']} | {verifier_seconds} | "
            f"{flags} | {'是' if record['review_required'] else '否'} |"
        )
    if args.mode == "study":
        lines.extend(
            [
                "",
                "学习模式只翻译英文自然语言。拟声词、纯数字表、纯公式、",
                "独立数字图形标注和页脚页码不会作为漏识别项。",
                "",
                "浏览 `study.html` 可对照整页原图查看英中翻译。",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "主OCR始终是文字权威来源。复核模型只提供独立比较结果，",
                "程序不会自动合并或覆盖主OCR文本。",
                "",
            ]
        )
    atomic_write_text(args.output / "summary.md", "\n".join(lines))
    if args.mode == "study":
        write_study_html(args, records)


def write_study_html(
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    output_name: str = "study.html",
    reviewed: bool = False,
) -> None:
    page_sections: list[str] = []
    for record in records:
        region_cards: list[str] = []
        translation_issues: dict[str, list[dict[str, str]]] = collections.defaultdict(list)
        for issue in record.get("translation_verifier_comparison", {}).get(
            "issues", []
        ):
            translation_issues[issue["id"]].append(issue)
        for index, region in enumerate(record.get("study", {}).get("regions", []), start=1):
            adjudication = region.get("codex_adjudication", {}) if reviewed else {}
            child_note = str(adjudication.get("child_note", ""))
            adjudication_html = ""
            if adjudication:
                decision_labels = {
                    "replace": "Codex已修正",
                    "normalize": "Codex已统一",
                    "discard": "Codex已复核保留",
                }
                label = decision_labels.get(
                    str(adjudication.get("decision", "")), "Codex已复核"
                )
                adjudication_html = (
                    '<div class="adjudication">'
                    f'<strong>{html.escape(label)}</strong>：'
                    f'{html.escape(str(adjudication.get("reason", "")))}'
                    "</div>"
                )
                if child_note:
                    adjudication_html += (
                        '<div class="child-note"><strong>学习提示：</strong>'
                        f'{html.escape(child_note)}</div>'
                    )
            issue_html = "".join(
                '<div class="translation-issue">'
                f'语义复核（{html.escape(issue["category"])}）：'
                f'{html.escape(issue["explanation"])}<br>'
                f'建议：{html.escape(issue["suggested_translation"] or "[未提供]")}'
                "</div>"
                for issue in translation_issues.get(region["id"], [])
            ) if not reviewed else ""
            region_cards.append(
                '<article class="region">'
                f'<div class="meta"><span class="number">{index}</span>'
                f'<span class="type">{html.escape(region["type"])}</span>'
                f'<span class="id">{html.escape(region["id"])}</span></div>'
                f'<div class="english">{html.escape(region["source_text"])}</div>'
                f'<div class="chinese">{html.escape(region["translation"] or "[缺少翻译]")}</div>'
                f"{adjudication_html}"
                f"{issue_html}"
                "</article>"
            )
        if not region_cards:
            region_cards.append('<p class="empty">本页没有需要翻译的英文自然语言。</p>')
        status_class = "review" if record["review_required"] else "pass"
        status_text = (
            "仍需人工复核"
            if reviewed and record["review_required"]
            else "Codex已完成裁决"
            if reviewed
            else "需要复核"
            if record["review_required"]
            else "已通过自动检查"
        )
        reasons = ", ".join(record.get("review_reasons", [])) or "none"
        page_sections.append(
            f'<section class="page" id="page-{record["page"]}">'
            '<div class="page-head">'
            f'<h2>PDF 第 {record["page"]} 页</h2>'
            f'<span class="status {status_class}">{status_text}</span>'
            "</div>"
            f'<div class="reasons">复核原因：{html.escape(reasons)}</div>'
            '<div class="columns">'
            f'<div class="image-panel"><img src="{html.escape(record["image"])}" '
            f'alt="PDF 第 {record["page"]} 页"></div>'
            f'<div class="translations">{"".join(region_cards)}</div>'
            "</div></section>"
        )

    title = f"{args.pdf.name} - {'Codex裁决版' if reviewed else '学习翻译'}"
    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root {{ color-scheme: light; --ink:#172033; --muted:#657086; --line:#d9dfeb; --paper:#fff; --bg:#f3f5f9; --blue:#2457d6; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:16px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",sans-serif; }}
header {{ position:sticky; top:0; z-index:5; padding:14px 24px; background:rgba(255,255,255,.94); border-bottom:1px solid var(--line); backdrop-filter:blur(10px); }}
header h1 {{ margin:0; font-size:18px; }} header p {{ margin:2px 0 0; color:var(--muted); font-size:13px; }}
main {{ max-width:1500px; margin:0 auto; padding:24px; }}
.page {{ margin:0 0 28px; padding:20px; background:var(--paper); border:1px solid var(--line); border-radius:14px; box-shadow:0 8px 24px rgba(31,42,68,.06); }}
.page-head {{ display:flex; align-items:center; gap:12px; }} .page-head h2 {{ margin:0; font-size:20px; }}
.status {{ padding:3px 9px; border-radius:999px; font-size:12px; }} .status.pass {{ color:#11663a; background:#e7f7ee; }} .status.review {{ color:#8a4515; background:#fff1df; }}
.reasons {{ margin:5px 0 14px; color:var(--muted); font-size:12px; }}
.columns {{ display:grid; grid-template-columns:minmax(360px,1.05fr) minmax(360px,.95fr); gap:22px; align-items:start; }}
.image-panel {{ position:sticky; top:76px; }} .image-panel img {{ display:block; width:100%; height:auto; border:1px solid var(--line); border-radius:8px; }}
.translations {{ display:grid; gap:11px; }} .region {{ padding:13px 15px; border:1px solid var(--line); border-radius:9px; }}
.meta {{ display:flex; gap:7px; align-items:center; margin-bottom:7px; color:var(--muted); font-size:11px; text-transform:uppercase; }}
.number {{ display:grid; place-items:center; width:22px; height:22px; border-radius:50%; color:#fff; background:var(--blue); font-weight:700; }}
.type {{ padding:2px 6px; border-radius:4px; background:#eef2fb; }} .id {{ margin-left:auto; }}
.english {{ font-weight:650; white-space:pre-wrap; }} .chinese {{ margin-top:6px; color:#243c78; white-space:pre-wrap; }} .empty {{ color:var(--muted); }}
.translation-issue {{ margin-top:9px; padding:8px 10px; color:#7b3d0e; background:#fff3e4; border-left:3px solid #dc7b2a; border-radius:4px; font-size:13px; }}
.adjudication {{ margin-top:9px; padding:8px 10px; color:#22543d; background:#edf9f1; border-left:3px solid #38a169; border-radius:4px; font-size:13px; }}
.child-note {{ margin-top:7px; padding:9px 11px; color:#5b4512; background:#fff9db; border:1px solid #f0d879; border-radius:6px; font-size:14px; }}
@media (max-width:850px) {{ main {{ padding:12px; }} .page {{ padding:13px; }} .columns {{ grid-template-columns:1fr; }} .image-panel {{ position:static; }} }}
</style>
</head>
<body>
<header><h1>{html.escape(title)}</h1><p>{'已应用Codex整章裁决；原始结果保持不变。' if reviewed else '整页原图与选择性英中翻译对照；不翻译拟声词、纯数字表、纯公式、独立数字标注和页脚页码。'}</p></header>
<main>{''.join(page_sections)}</main>
</body>
</html>
"""
    atomic_write_text(args.output / output_name, document)


ENV_API_KEY_NAMES = frozenset(
    {
        "SILICONFLOW_API_KEY",
        "DASHSCOPE_API_KEY",
        "DEEPL_OAUTH_CREDENTIALS",
    }
)


def load_env_file(env_path: pathlib.Path) -> None:
    """Load supported API keys from a UTF-8 .env file without overriding the process.

    Only credential variables used by this project are read. Existing process
    environment variables take precedence, allowing CI and shell-provided secrets to
    override a local .env file. DeepL OAuth credentials are generated and refreshed
    by the MCP helper; users should not edit their encoded value manually.
    """

    if not env_path.is_file():
        return
    try:
        lines = env_path.read_text(encoding="utf-8-sig").splitlines()
    except OSError as error:
        raise ValueError(f"Unable to read .env file: {env_path}") from error
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").lstrip()
        name, separator, raw_value = line.partition("=")
        name = name.strip()
        if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError(f"Invalid .env entry on line {line_number}")
        if name not in ENV_API_KEY_NAMES:
            continue
        value = raw_value.strip()
        if value[:1] in {"'", '"'}:
            quote = value[0]
            if len(value) < 2 or value[-1] != quote:
                raise ValueError(f"Unclosed quoted value in .env line {line_number}")
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        os.environ.setdefault(name, value)


def discover_config_path(argv: list[str]) -> pathlib.Path | None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=pathlib.Path)
    known, _ = pre_parser.parse_known_args(argv)
    if known.config:
        return known.config.expanduser().resolve()
    manual_task_options = {"--pdf", "--pages", "--output"}
    if not any(option in argv for option in manual_task_options):
        default_path = pathlib.Path(__file__).resolve().with_name("ocr_config.json")
        if default_path.is_file():
            return default_path
    return None


def strip_json_comments(source: str) -> str:
    """Remove // and /* */ comments without changing comment markers in strings."""

    output: list[str] = []
    index = 0
    in_string = False
    escaped = False
    while index < len(source):
        char = source[index]
        next_char = source[index + 1] if index + 1 < len(source) else ""
        if in_string:
            output.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
            output.append(char)
            index += 1
            continue
        if char == "/" and next_char == "/":
            index += 2
            while index < len(source) and source[index] not in "\r\n":
                index += 1
            continue
        if char == "/" and next_char == "*":
            index += 2
            closed = False
            while index < len(source):
                if source[index] == "\n":
                    output.append("\n")
                if (
                    source[index] == "*"
                    and index + 1 < len(source)
                    and source[index + 1] == "/"
                ):
                    index += 2
                    closed = True
                    break
                index += 1
            if not closed:
                raise ValueError("Unterminated /* */ comment in config")
            continue
        output.append(char)
        index += 1
    return "".join(output)


def load_config_defaults(config_path: pathlib.Path) -> dict[str, Any]:
    try:
        source = config_path.read_text(encoding="utf-8")
        data = json.loads(strip_json_comments(source))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON/JSONC config {config_path}: {error}") from error
    if not isinstance(data, dict):
        raise ValueError("Config root must be a JSON object")

    direct_fields = {
        "pdf",
        "pages",
        "output",
        "mode",
        "long_edge",
        "pdftoppm_command",
        "workers",
        "verify",
        "printed_page_offset",
        "resume",
        "redo_pages",
        "log_file",
    }
    if "api_keys" in data:
        raise ValueError(
            "config.api_keys is no longer supported; move keys to the .env file "
            "beside this config"
        )
    allowed_fields = direct_fields | {
        "translation",
        "model_profiles",
        "model_usage",
        "pdf_backfill",
        "pdf_translation_writer",
        "codex_page_review",
        "codex_book_review",
    }
    unknown = sorted(set(data) - allowed_fields)
    if unknown:
        raise ValueError(f"Unknown config fields: {', '.join(unknown)}")

    defaults = {field: data[field] for field in direct_fields if field in data}
    model_profiles = data.get("model_profiles")
    if model_profiles is not None:
        if not isinstance(model_profiles, dict):
            raise ValueError("config.model_profiles must be an object")
        defaults["model_profiles"] = model_profiles
    model_usage = data.get("model_usage")
    if model_usage is not None:
        if not isinstance(model_usage, dict):
            raise ValueError("config.model_usage must be an object")
        defaults["model_usage"] = model_usage

    pdf_backfill = data.get("pdf_backfill", {})
    if not isinstance(pdf_backfill, dict):
        raise ValueError("config.pdf_backfill must be an object")
    defaults["pdf_backfill"] = pdf_backfill

    codex_page_review = data.get("codex_page_review", {})
    if not isinstance(codex_page_review, dict):
        raise ValueError("config.codex_page_review must be an object")

    translation = data.get("translation", {})
    if not isinstance(translation, dict):
        raise ValueError("config.translation must be an object")
    if translation.get("provider", "deepl_mcp") != "deepl_mcp":
        raise ValueError("config.translation.provider must be 'deepl_mcp'")
    translation_fields = {
        "provider": "translation_provider",
        "endpoint": "deepl_mcp_endpoint",
        "source_lang": "deepl_source_lang",
        "target_lang": "deepl_target_lang",
        "formality": "deepl_formality",
        "glossary_id": "deepl_glossary_id",
        "style_id": "deepl_style_id",
        "context": "deepl_context",
        "custom_instructions": "deepl_custom_instructions",
        "oauth_callback_port": "deepl_oauth_callback_port",
        "oauth_keychain_service": "deepl_oauth_keychain_service",
        "oauth_keychain_account": "deepl_oauth_keychain_account",
        "node_command": "deepl_node_command",
        "bridge_script": "deepl_bridge_script",
    }
    unknown_translation = sorted(set(translation) - set(translation_fields))
    if unknown_translation:
        raise ValueError(
            f"Unknown translation fields: {', '.join(unknown_translation)}"
        )
    for config_name, argument_name in translation_fields.items():
        if config_name in translation:
            defaults[argument_name] = translation[config_name]

    base = config_path.parent
    for field in (
        "pdf",
        "output",
        "log_file",
        "pdftoppm_command",
        "deepl_bridge_script",
    ):
        value = defaults.get(field)
        if not value:
            continue
        path = pathlib.Path(value).expanduser()
        if not path.is_absolute():
            path = base / path
        defaults[field] = path.resolve()
    defaults["config"] = config_path
    return defaults


def build_parser(defaults: dict[str, Any] | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=pathlib.Path)
    parser.add_argument("--pdf", type=pathlib.Path)
    parser.add_argument("--pages", help="Example: 25,53,57-59")
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument(
        "--mode",
        choices=("exact", "study"),
        default="exact",
        help="exact transcribes everything; study selects and translates prose",
    )
    parser.add_argument("--long-edge", type=int, default=2048)
    parser.add_argument(
        "--pdftoppm-command",
        help="Optional path to a Poppler-compatible pdftoppm executable",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--verify", choices=("auto", "always", "never"), default="auto")
    parser.add_argument("--printed-page-offset", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--redo-pages",
        help="With --resume, rerun only these pages and reuse all other page results",
    )
    parser.add_argument("--layout-profile")
    parser.add_argument("--primary-profile")
    parser.add_argument("--verifier-profile")
    parser.add_argument("--deepl-mcp-endpoint", default=DEEPL_MCP_ENDPOINT)
    parser.add_argument("--deepl-source-lang", default="EN")
    parser.add_argument("--deepl-target-lang", default="ZH-HANS")
    parser.add_argument("--deepl-formality", default="")
    parser.add_argument("--deepl-glossary-id", default="")
    parser.add_argument("--deepl-style-id", default="")
    parser.add_argument("--deepl-context", default="")
    parser.add_argument("--deepl-custom-instructions", nargs="*", default=[])
    parser.add_argument("--deepl-oauth-callback-port", type=int, default=8765)
    parser.add_argument(
        "--deepl-oauth-keychain-service",
        default="Beast Academy OCR DeepL MCP",
    )
    parser.add_argument("--deepl-oauth-keychain-account", default="")
    parser.add_argument("--deepl-node-command", default="node")
    parser.add_argument(
        "--deepl-bridge-script",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().with_name("deepl_mcp_client.mjs"),
    )
    parser.add_argument("--log-file", type=pathlib.Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate resolved config, keys, PDF, pages, and tools without API calls",
    )
    parser.set_defaults(
        model_profiles={
            name: dict(profile) for name, profile in DEFAULT_MODEL_PROFILES.items()
        },
        model_usage=dict(DEFAULT_MODEL_USAGE),
        pdf_backfill={},
        translation_provider="deepl_mcp",
        translation_verify="never",
        reanalyze=False,
        retranslate=False,
        reverify_translation=False,
        export_chapter_review=False,
        chapter_review=False,
        build_reviewed_study=False,
        build_final_translation=False,
        export_pdf=None,
        adjudication_file=None,
    )
    if defaults:
        parser.set_defaults(**defaults)
    return parser


def resolve_model_configuration(args: argparse.Namespace) -> None:
    profiles = args.model_profiles
    usage = args.model_usage
    if not isinstance(profiles, dict):
        raise ValueError("model_profiles must be an object")
    if not isinstance(usage, dict):
        raise ValueError("model_usage must be an object")
    allowed_profile_fields = {
        "provider",
        "model",
        "api_key_name",
        "chat_endpoint",
        "responses_endpoint",
        "enable_thinking",
        "max_tokens",
    }
    key_values = {
        "dashscope": args.dashscope_key,
        "siliconflow": args.siliconflow_key,
    }
    resolved_profiles: dict[str, dict[str, Any]] = {}
    for name, raw in profiles.items():
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", str(name)):
            raise ValueError(f"Invalid model profile name: {name!r}")
        if not isinstance(raw, dict):
            raise ValueError(f"model_profiles.{name} must be an object")
        unknown = sorted(set(raw) - allowed_profile_fields)
        if unknown:
            raise ValueError(
                f"Unknown fields in model profile {name}: {', '.join(unknown)}"
            )
        model = str(raw.get("model", "")).strip()
        provider = str(raw.get("provider", "")).strip()
        key_name = str(raw.get("api_key_name", "")).strip()
        chat_endpoint = str(raw.get("chat_endpoint", "")).strip()
        responses_endpoint = str(raw.get("responses_endpoint", "")).strip()
        thinking = raw.get("enable_thinking", None)
        if not model or not provider:
            raise ValueError(f"Model profile {name} requires provider and model")
        if key_name not in key_values:
            raise ValueError(
                f"Model profile {name} uses unknown api_key_name: {key_name}"
            )
        for endpoint in (chat_endpoint, responses_endpoint):
            if endpoint and not endpoint.startswith("https://"):
                raise ValueError(f"Model profile {name} endpoints must use https://")
        if thinking is not None and not isinstance(thinking, bool):
            raise ValueError(
                f"Model profile {name} enable_thinking must be true, false, or null"
            )
        max_tokens = int(raw.get("max_tokens", 16384))
        if max_tokens < 1024 or max_tokens > 65536:
            raise ValueError(
                f"Model profile {name} max_tokens must be between 1024 and 65536"
            )
        resolved_profiles[str(name)] = {
            "name": str(name),
            "provider": provider,
            "model": model,
            "api_key_name": key_name,
            "api_key": key_values[key_name],
            "chat_endpoint": chat_endpoint,
            "responses_endpoint": responses_endpoint,
            "enable_thinking": thinking,
            "max_tokens": max_tokens,
        }

    allowed_usage = {
        "layout_ocr",
        "primary_ocr",
        "ocr_verifier",
    }
    unknown_usage = sorted(set(usage) - allowed_usage)
    if unknown_usage:
        raise ValueError(f"Unknown model_usage fields: {', '.join(unknown_usage)}")
    selected = dict(DEFAULT_MODEL_USAGE)
    selected.update(usage)
    cli_overrides = {
        "layout_ocr": args.layout_profile,
        "primary_ocr": args.primary_profile,
        "ocr_verifier": args.verifier_profile,
    }
    for role, value in cli_overrides.items():
        if value:
            selected[role] = value
    def profile_for(role: str) -> dict[str, Any]:
        name = selected.get(role)
        if not isinstance(name, str) or name not in resolved_profiles:
            raise ValueError(f"model_usage.{role} references unknown profile: {name}")
        return resolved_profiles[name]

    args.layout_profile_config = profile_for("layout_ocr")
    args.primary_profile_config = profile_for("primary_ocr")
    args.verifier_profile_config = profile_for("ocr_verifier")
    for role, profile in (
        ("layout_ocr", args.layout_profile_config),
        ("primary_ocr", args.primary_profile_config),
    ):
        if not profile["chat_endpoint"]:
            raise ValueError(f"Model profile for {role} has no chat_endpoint")
    if not args.verifier_profile_config["responses_endpoint"]:
        raise ValueError(
            "Model profile for ocr_verifier has no responses_endpoint"
        )
    args.resolved_chapter_review_models = []

    args.layout_model = args.layout_profile_config["model"]
    args.primary_model = args.primary_profile_config["model"]
    args.verifier_model = args.verifier_profile_config["model"]


def resolve_pdf_backfill_settings(args: argparse.Namespace) -> dict[str, Any]:
    raw = args.pdf_backfill
    if not isinstance(raw, dict):
        raise ValueError("pdf_backfill must be an object")
    allowed = {
        "output_filename",
        "final_translation_filename",
        "plan_filename",
        "report_filename",
        "spotcheck_filename",
        "font_file",
        "human_translation_overrides_filename",
        "layout_overrides_filename",
        "rules",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(
            f"Unknown pdf_backfill fields: {', '.join(unknown)}"
        )

    def output_path(field: str, default: str, *, optional: bool = False) -> pathlib.Path | None:
        value = str(raw.get(field, default)).strip()
        if optional and not value:
            return None
        if not value:
            raise ValueError(f"pdf_backfill.{field} cannot be empty")
        path = pathlib.Path(value).expanduser()
        if not path.is_absolute():
            path = args.output / path
        return path.resolve()

    font_value = str(
        raw.get(
            "font_file",
            "/System/Library/AssetsV2/com_apple_MobileAsset_Font7/"
            "eb257c12d1a51c8c661b89f30eec56cacf9b8987.asset/AssetData/STHEITI.ttf",
        )
    ).strip()
    if not font_value:
        raise ValueError("pdf_backfill.font_file cannot be empty")
    font_file = pathlib.Path(font_value).expanduser()
    if not font_file.is_absolute():
        font_file = pathlib.Path(__file__).resolve().parent / font_file
    from pdf_backfill import merge_rules

    return {
        "output_pdf": output_path(
            "output_filename", "beast-academy-3A-chapter1-zh-review.pdf"
        ),
        "final_translation": output_path(
            "final_translation_filename", "chapter_translation_final.json"
        ),
        "plan": output_path("plan_filename", "pdf_backfill_plan.json"),
        "report": output_path("report_filename", "pdf_backfill_report.json"),
        "spotcheck": output_path(
            "spotcheck_filename", "ai_spotcheck_request.json"
        ),
        "font_file": font_file.resolve(),
        "human_translation_overrides": output_path(
            "human_translation_overrides_filename",
            "",
            optional=True,
        ),
        "layout_overrides": output_path(
            "layout_overrides_filename",
            "",
            optional=True,
        ),
        "rules": merge_rules(raw.get("rules")),
    }


def build_final_translation_for_backfill(
    args: argparse.Namespace,
    pages: list[int],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    from pdf_backfill import (
        build_final_translation_snapshot,
        load_human_translation_overrides,
    )

    settings = resolve_pdf_backfill_settings(args)
    all_pages = discover_existing_study_pages(args.output)
    missing_pages = sorted(set(pages) - set(all_pages))
    if missing_pages:
        raise ValueError(
            "Requested PDF backfill pages are not present in the saved chapter: "
            + ", ".join(str(page) for page in missing_pages)
        )
    settings = scope_backfill_paths_for_subset(settings, pages, all_pages)
    chapter_records = load_existing_page_records(args.output, all_pages)
    adjudication = json.loads(
        args.adjudication_file.read_text(encoding="utf-8")
    )
    verify_adjudication_inputs(args.output, adjudication)
    all_reviewed_records, _application = apply_chapter_adjudication(
        chapter_records, adjudication
    )
    requested_pages = set(pages)
    reviewed_records = [
        record
        for record in all_reviewed_records
        if int(record["page"]) in requested_pages
    ]
    human_overrides_path = settings["human_translation_overrides"]
    human_overrides = {
        coordinate: override
        for coordinate, override in load_human_translation_overrides(
            human_overrides_path
        ).items()
        if coordinate[0] in requested_pages
    }
    page_record_hashes = {
        f"page-{page:04d}.json": sha256_file(
            args.output / "pages" / f"page-{page:04d}.json"
        )
        for page in pages
    }
    snapshot = build_final_translation_snapshot(
        reviewed_records,
        source_pdf=args.pdf,
        adjudication_file=args.adjudication_file,
        page_record_hashes=page_record_hashes,
        human_overrides=human_overrides,
        human_overrides_file=human_overrides_path,
    )
    atomic_write_json(settings["final_translation"], snapshot)
    return reviewed_records, snapshot, settings


def validate_args(args: argparse.Namespace) -> list[int]:
    if args.pdf is None:
        raise ValueError("--pdf is required (or set pdf in the config)")
    if not args.pages:
        raise ValueError("--pages is required (or set pages in the config)")
    if args.output is None:
        raise ValueError("--output is required (or set output in the config)")
    if not args.pdf.is_file():
        raise ValueError(f"PDF not found: {args.pdf}")
    args.pdftoppm_command = find_poppler_pdftoppm(args.pdftoppm_command)
    args.pdfinfo_command = find_matching_pdfinfo(args.pdftoppm_command)
    if args.long_edge < 512:
        raise ValueError("--long-edge must be at least 512")
    if args.workers < 1 or args.workers > 4:
        raise ValueError("--workers must be between 1 and 4")
    maintenance_modes = sum(
        bool(value)
        for value in (
            args.reanalyze,
            args.retranslate,
            args.reverify_translation,
            args.export_chapter_review,
            args.chapter_review,
            args.build_reviewed_study,
            args.build_final_translation,
            bool(args.export_pdf),
        )
    )
    if maintenance_modes > 1:
        raise ValueError(
            "Maintenance and chapter-review modes cannot be used together"
        )
    fresh_ocr = not maintenance_modes
    resolve_model_configuration(args)
    if fresh_ocr and not args.layout_profile_config["api_key"]:
        raise ValueError("API key is missing for the selected layout OCR profile")
    if fresh_ocr and not args.primary_profile_config["api_key"]:
        raise ValueError("API key is missing for the selected primary OCR profile")
    if (
        fresh_ocr
        and args.verify != "never"
        and not args.verifier_profile_config["api_key"]
    ):
        raise ValueError("API key is missing for the selected OCR verifier profile")
    if (
        args.retranslate
        and args.translation_provider == "qwen"
        and not args.translation_profile_config["api_key"]
    ):
        raise ValueError("API key is missing for the selected Qwen translation profile")
    if (
        args.mode == "study"
        and args.translation_verify == "always"
        and (fresh_ocr or args.retranslate)
        and not args.translation_verifier_profile_config["api_key"]
    ):
        raise ValueError("API key is missing for the translation verifier profile")
    if (
        args.reverify_translation
        and not args.translation_verifier_profile_config["api_key"]
    ):
        raise ValueError(
            "API key is missing for the translation verifier profile"
        )
    if args.chapter_review:
        missing_profiles = [
            profile["name"]
            for profile in args.resolved_chapter_review_models
            if not profile["api_key"]
        ]
        if missing_profiles:
            raise ValueError(
                "API key is missing for chapter review profiles: "
                + ", ".join(missing_profiles)
            )
    needs_translation = (
        args.mode == "study"
        and (fresh_ocr or args.retranslate)
    )
    if needs_translation and args.translation_provider == "deepl_mcp":
        if not args.deepl_mcp_endpoint.startswith("https://"):
            raise ValueError("DeepL MCP endpoint must use https://")
        if not args.deepl_target_lang:
            raise ValueError("DeepL target language is required")
        if args.deepl_glossary_id and not args.deepl_source_lang:
            raise ValueError("DeepL glossary use requires source_lang")
        if len(args.deepl_context) > 300:
            raise ValueError("DeepL context must be at most 300 characters")
        if not isinstance(args.deepl_custom_instructions, list):
            raise ValueError("DeepL custom_instructions must be a JSON array")
        if len(args.deepl_custom_instructions) > 10:
            raise ValueError("DeepL supports at most 10 custom instructions")
        if any(
            not isinstance(item, str) or not item.strip() or len(item) > 300
            for item in args.deepl_custom_instructions
        ):
            raise ValueError(
                "Each DeepL custom instruction must be a non-empty string of at most 300 characters"
            )
        if not 1024 <= args.deepl_oauth_callback_port <= 65535:
            raise ValueError("DeepL OAuth callback port must be between 1024 and 65535")
        resolved_node = shutil.which(args.deepl_node_command)
        if resolved_node is None:
            raise RuntimeError(
                f"DeepL MCP Node command not found: {args.deepl_node_command}"
            )
        args.deepl_node_command = resolved_node
        if not args.deepl_bridge_script.is_file():
            raise ValueError(
                f"DeepL MCP bridge script not found: {args.deepl_bridge_script}"
            )
        sdk_path = args.deepl_bridge_script.parent / "node_modules" / "@modelcontextprotocol" / "sdk"
        if not sdk_path.is_dir():
            raise RuntimeError(
                "DeepL MCP SDK is missing; run npm install in the ocr-demo directory"
            )
        if not args.deepl_oauth_keychain_account:
            args.deepl_oauth_keychain_account = args.deepl_mcp_endpoint
    pages = parse_pages(args.pages)
    args.redo_pages = set(parse_pages(args.redo_pages)) if args.redo_pages else set()
    if args.redo_pages and not args.resume:
        raise ValueError("--redo-pages requires --resume")
    if not args.redo_pages.issubset(pages):
        raise ValueError("--redo-pages must be a subset of --pages")
    count = pdf_page_count(args.pdf, args.pdfinfo_command)
    if pages[-1] > count:
        raise ValueError(f"PDF has {count} pages; requested page {pages[-1]}")
    if maintenance_modes and not args.output.is_dir():
        print(
            "Maintenance mode requires an existing output directory; "
            "if this is a new run, create the output directory first"
        )
        #raise ValueError("Maintenance mode requires an existing output directory")
    if args.build_reviewed_study or args.build_final_translation or args.export_pdf:
        if args.adjudication_file is None:
            args.adjudication_file = (
                args.output / "chapter_review-codex-adjudication.json"
            )
        if not args.adjudication_file.is_file():
            raise ValueError(
                f"Codex adjudication file not found: {args.adjudication_file}"
            )
    if args.build_final_translation or args.export_pdf:
        resolve_pdf_backfill_settings(args)
    if (
        args.output.exists()
        and any(args.output.iterdir())
        and not args.resume
        and not args.reanalyze
        and not args.retranslate
        and not args.reverify_translation
        and not args.export_chapter_review
        and not args.chapter_review
        and not args.build_reviewed_study
        and not args.build_final_translation
        and not args.export_pdf
    ):
        print("Output directory is not empty; use --resume or a new directory")
        #raise ValueError("Output directory is not empty; use --resume or a new directory")
    args.output.mkdir(parents=True, exist_ok=True)
    return pages


def main() -> int:
    config_path = discover_config_path(sys.argv[1:])
    env_path = (
        config_path.parent if config_path else pathlib.Path(__file__).resolve().parent
    ) / ".env"
    try:
        load_env_file(env_path)
    except ValueError as error:
        build_parser().error(str(error))
    defaults: dict[str, Any] = {}
    if config_path is not None:
        try:
            defaults = load_config_defaults(config_path)
        except Exception as error:
            build_parser().error(str(error))
    parser = build_parser(defaults)
    args = parser.parse_args()
    if args.pdf is not None:
        args.pdf = args.pdf.expanduser().resolve()
    if args.output is not None:
        args.output = args.output.expanduser().resolve()
    if args.log_file is not None:
        args.log_file = args.log_file.expanduser().resolve()
    if args.adjudication_file is not None:
        args.adjudication_file = args.adjudication_file.expanduser().resolve()
    args.deepl_bridge_script = args.deepl_bridge_script.expanduser().resolve()
    args.deepl_env_file = env_path.resolve()
    args.siliconflow_key = getattr(args, "siliconflow_key", "") or os.environ.get(
        "SILICONFLOW_API_KEY", ""
    )
    args.dashscope_key = getattr(args, "dashscope_key", "") or os.environ.get(
        "DASHSCOPE_API_KEY", ""
    )
    with TeeLogging(args.log_file):
        try:
            pages = validate_args(args)
            print(
                json.dumps(
                    {
                        "event": "configuration_validated",
                        "config": str(config_path) if config_path else None,
                        "pdf": str(args.pdf),
                        "pages": args.pages,
                        "output": str(args.output),
                        "mode": args.mode,
                        "workers": args.workers,
                        "verify": args.verify,
                        "translation_provider": args.translation_provider,
                        "build_final_translation": args.build_final_translation,
                        "export_pdf": args.export_pdf,
                        "model_usage": {
                            "layout_ocr": args.layout_profile_config["name"],
                            "primary_ocr": args.primary_profile_config["name"],
                            "ocr_verifier": args.verifier_profile_config["name"],
                            "qwen_translation": args.translation_profile_config["name"],
                            "translation_verifier": args.translation_verifier_profile_config["name"],
                            "chapter_review": [
                                profile["name"]
                                for profile in args.resolved_chapter_review_models
                            ],
                        },
                        "siliconflow_key_configured": bool(args.siliconflow_key),
                        "dashscope_key_configured": bool(args.dashscope_key),
                        "deepseek_key_configured": bool(args.deepseek_key),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if args.dry_run:
                print("dry_run=ok")
                return 0
            if args.build_final_translation:
                _records, snapshot, settings = build_final_translation_for_backfill(
                    args, pages
                )
                print(
                    f"final_translation_built={snapshot['region_count']} "
                    f"pages={snapshot['page_count']} "
                    f"output={settings['final_translation']}"
                )
                return 0
            if args.export_pdf == "chinese":
                from pdf_backfill import (
                    build_ai_spotcheck_request,
                    build_backfill_plan,
                    export_chinese_pdf,
                    load_layout_overrides,
                    program_check_pdf,
                )

                reviewed_records, snapshot, settings = (
                    build_final_translation_for_backfill(args, pages)
                )
                plan = build_backfill_plan(snapshot, reviewed_records)
                atomic_write_json(settings["plan"], plan)
                layout_overrides = load_layout_overrides(
                    settings["layout_overrides"]
                )
                export_result = export_chinese_pdf(
                    source_pdf=args.pdf,
                    pages=pages,
                    plan=plan,
                    rules=settings["rules"],
                    layout_overrides=layout_overrides,
                    font_file=settings["font_file"],
                    output_pdf=settings["output_pdf"],
                )
                program_check = program_check_pdf(
                    source_pdf=args.pdf,
                    output_pdf=settings["output_pdf"],
                    pages=pages,
                    snapshot=snapshot,
                    plan=plan,
                    export_result=export_result,
                )
                report = {
                    "schema_version": 1,
                    "status": program_check["status"],
                    "visual_review": "not_performed_human_required",
                    "final_translation_file": str(settings["final_translation"]),
                    "plan_file": str(settings["plan"]),
                    "export": export_result,
                    "program_check": program_check,
                }
                atomic_write_json(settings["report"], report)
                spotcheck = build_ai_spotcheck_request(plan, settings["rules"])
                atomic_write_json(settings["spotcheck"], spotcheck)
                policy_skipped = sum(
                    int(export_result["status_counts"].get(status, 0))
                    for status in (
                        "skipped_low_confidence",
                        "skipped_ambiguous_mapping",
                        "skipped_geometry_label",
                    )
                )
                print(
                    f"pdf_backfill_status={program_check['status']} "
                    f"written={export_result['status_counts'].get('written', 0)} "
                    f"unmatched={plan['unmapped_count']} "
                    f"policy_skipped={policy_skipped} "
                    f"output={settings['output_pdf']} "
                    "visual_review=human_required"
                )
                return 0 if program_check["status"] == "program_checked" else 1
            if args.build_reviewed_study:
                chapter_records = load_existing_page_records(args.output, pages)
                adjudication = json.loads(
                    args.adjudication_file.read_text(encoding="utf-8")
                )
                verify_adjudication_inputs(args.output, adjudication)
                reviewed_records, application = apply_chapter_adjudication(
                    chapter_records, adjudication
                )
                application["source_adjudication"] = str(args.adjudication_file)
                atomic_write_json(
                    args.output / "chapter_review-applied.json", application
                )
                write_study_html(
                    args,
                    reviewed_records,
                    output_name="study-reviewed.html",
                    reviewed=True,
                )
                print(
                    f"reviewed_study_built={len(reviewed_records)} "
                    f"applied={application['applied_count']} "
                    f"human_review={application['human_review_count']} "
                    f"output={args.output / 'study-reviewed.html'}"
                )
                return 0
            if args.export_chapter_review or args.chapter_review:
                chapter_records = load_existing_page_records(args.output, pages)
                review_input = build_chapter_review_input(args, chapter_records)
                input_path = args.output / "chapter_review_input.md"
                atomic_write_text(input_path, review_input)
                if args.export_chapter_review:
                    print(
                        f"chapter_review_exported={len(chapter_records)} "
                        f"characters={len(review_input)} output={input_path}"
                    )
                    return 0

                valid_regions = {
                    (record["page"], region["id"])
                    for record in chapter_records
                    for region in record.get("study", {}).get("regions", [])
                }
                reports: dict[str, dict[str, Any]] = {}
                failures: dict[str, str] = {}
                for model_config in args.resolved_chapter_review_models:
                    name = model_config["name"]
                    prefix = f"chapter_review-{name}"
                    api_result = call_chapter_translation_verifier(
                        args, review_input, model_config
                    )
                    atomic_write_json(
                        args.output / f"{prefix}-api.json",
                        api_result.as_dict(),
                    )
                    if api_result.error:
                        failures[name] = api_result.error
                        continue
                    try:
                        parsed = parse_chapter_review(
                            api_result.content, valid_regions
                        )
                    except ValueError as error:
                        failures[name] = str(error)
                        atomic_write_json(
                            args.output / f"{prefix}.json",
                            {
                                "error": str(error),
                                "issues": [],
                                "global_consistency": [],
                            },
                        )
                        continue
                    report_data, report_markdown = chapter_review_report(
                        chapter_records, parsed
                    )
                    report_data["profile"] = name
                    report_data["provider"] = api_result.provider
                    report_data["model"] = api_result.model
                    report_data["elapsed_seconds"] = api_result.elapsed_seconds
                    reports[name] = report_data
                    atomic_write_json(
                        args.output / f"{prefix}.json", report_data
                    )
                    atomic_write_text(
                        args.output / f"{prefix}.md", report_markdown
                    )
                    print(
                        f"chapter_review_profile={name} "
                        f"issues={len(parsed['issues'])} "
                        f"global_consistency={len(parsed['global_consistency'])}",
                        flush=True,
                    )
                comparison_data, comparison_markdown = chapter_model_comparison(
                    reports, failures
                )
                atomic_write_json(
                    args.output / "chapter_review-comparison.json",
                    comparison_data,
                )
                atomic_write_text(
                    args.output / "chapter_review-comparison.md",
                    comparison_markdown,
                )
                print(
                    f"chapter_reviewed={len(chapter_records)} "
                    f"models_succeeded={len(reports)} "
                    f"models_failed={len(failures)} "
                    f"output={args.output}"
                )
                return 1 if failures else 0
            records: list[dict[str, Any]] = []
            if args.reanalyze:
                records = [reanalyze_page(args, page) for page in pages]
            elif args.retranslate:
                with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
                    futures = {
                        executor.submit(retranslate_page, args, page): page
                        for page in pages
                    }
                    for future in concurrent.futures.as_completed(futures):
                        page = futures[future]
                        try:
                            records.append(future.result())
                        except Exception as error:
                            print(
                                f"page {page} failed: {type(error).__name__}: {error}",
                                file=sys.stderr,
                            )
                            return 1
            elif args.reverify_translation:
                with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
                    futures = {
                        executor.submit(reverify_translation_page, args, page): page
                        for page in pages
                    }
                    for future in concurrent.futures.as_completed(futures):
                        page = futures[future]
                        try:
                            records.append(future.result())
                        except Exception as error:
                            print(
                                f"page {page} failed: {type(error).__name__}: {error}",
                                file=sys.stderr,
                            )
                            return 1
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
                    futures = {
                        executor.submit(process_page, args, page): page
                        for page in pages
                    }
                    for future in concurrent.futures.as_completed(futures):
                        page = futures[future]
                        try:
                            records.append(future.result())
                        except Exception as error:
                            print(
                                f"page {page} failed: {type(error).__name__}: {error}",
                                file=sys.stderr,
                            )
                            return 1
            updated_count = len(records)
            if args.reanalyze or args.retranslate or args.reverify_translation:
                records = merge_with_existing_records(args.output, records)
            records.sort(key=lambda record: record["page"])
            build_summary(args, records)
            if args.reanalyze:
                operation = "reanalyzed"
            elif args.retranslate:
                operation = "retranslated"
            elif args.reverify_translation:
                operation = "translation_reverified"
            else:
                operation = "completed"
            print(
                f"{operation}={updated_count} aggregate_pages={len(records)} "
                f"output={args.output}"
            )
            return 0
        except Exception as error:
            parser.error(str(error))
            return 2
        finally:
            close_deepl_bridges()


if __name__ == "__main__":
    raise SystemExit(main())
