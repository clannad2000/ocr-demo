from __future__ import annotations

import json
import pathlib
from typing import Any


class ConfigError(ValueError):
    """Raised when the shared pipeline configuration is invalid."""


ALLOWED_ROOT_FIELDS = {
    "pdf",
    "pages",
    "toc_pages",
    "long_edge",
    "pdftoppm_command",
    "workers",
    "verify",
    "resume",
    "redo_pages",
    "printed_page_offset",
    "codex_review",
    "pdf_writer",
    "model_profiles",
    "model_usage",
    "translation",
}


def strip_json_comments(source: str) -> str:
    output: list[str] = []
    index = 0
    in_string = False
    escaped = False
    while index < len(source):
        char = source[index]
        following = source[index + 1] if index + 1 < len(source) else ""
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
        if char == "/" and following == "/":
            index += 2
            while index < len(source) and source[index] not in "\r\n":
                index += 1
            continue
        if char == "/" and following == "*":
            index += 2
            while index + 1 < len(source) and source[index : index + 2] != "*/":
                if source[index] == "\n":
                    output.append("\n")
                index += 1
            if index + 1 >= len(source):
                raise ConfigError("Unterminated block comment in configuration")
            index += 2
            continue
        output.append(char)
        index += 1
    return "".join(output)


def load_config(path: pathlib.Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    try:
        value = json.loads(strip_json_comments(path.read_text(encoding="utf-8")))
    except OSError as error:
        raise ConfigError(f"Cannot read configuration {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise ConfigError(f"Invalid JSONC configuration {path}: {error}") from error
    if not isinstance(value, dict):
        raise ConfigError("Configuration root must be an object")
    unknown = sorted(set(value) - ALLOWED_ROOT_FIELDS)
    if unknown:
        raise ConfigError("Unknown configuration fields: " + ", ".join(unknown))
    for field in ("codex_review", "pdf_writer", "model_profiles", "model_usage", "translation"):
        if field in value and not isinstance(value[field], dict):
            raise ConfigError(f"Configuration field '{field}' must be an object")
    return value
