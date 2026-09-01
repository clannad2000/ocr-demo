import contextlib
import io
import json
import os
import pathlib
import subprocess
import tempfile
import unittest
from unittest import mock

import codex_page_review as review


class CodexPageReviewTests(unittest.TestCase):
    def make_fixture(self, root: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
        config = root / "ocr_config.json"
        config.write_text(
            """{
              // Local ChatGPT-authenticated Codex settings.
              "codex_page_review": {
                "command": "codex",
                "model": "gpt-5.6-terra",
                "reasoning_effort": "high",
                "timeout_seconds": 120,
                "image_mode": "on_demand",
                "require_chatgpt_login": true
              }
            }
            """,
            encoding="utf-8",
        )
        pages = root / "runs" / "batch" / "pages"
        pages.mkdir(parents=True)
        page_json = pages / "page-0053.json"
        page_json.write_text(
            json.dumps(
                {
                    "page": 53,
                    "mode": "study",
                    "study": {
                        "regions": [
                            {
                                "id": "r001",
                                "type": "question",
                                "source_text": "What pattern do we get?",
                                "translation": "我们会得到什么规律？",
                            },
                            {
                                "id": "r002",
                                "type": "dialogue",
                                "source_text": "Stripes!",
                                "translation": "条纹！",
                            },
                        ],
                        "skipped": [
                            {"id": "r999", "source_text": "DO_NOT_SEND_SKIPPED"}
                        ],
                    },
                    "issues": [{"message": "DO_NOT_SEND_ISSUES"}],
                    "translation_verifier_comparison": {
                        "issues": [
                            {
                                "id": "r001",
                                "suggested_translation": "DO_NOT_SEND_VERIFIER",
                            }
                        ]
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        page_md = pages / "page-0053.md"
        page_md.write_text(
            """# PDF第53页 - 学习翻译

## 1. question (r001)

**英文：** What pattern do we get?

**中文：** 我们会得到什么规律？

## 2. dialogue (r002)

**英文：** Stripes!

**中文：** 条纹！

## 翻译语义复核建议（未自动改写）

```json
[{"id":"r001","suggested_translation":"我们会得到什么图案？"}]
```
""",
            encoding="utf-8",
        )
        (pages / "page-0053.png").write_bytes(b"fake-png")
        return config, page_json

    @staticmethod
    def valid_response() -> dict:
        return {
            "page": 53,
            "reviewed_region_ids": ["r001", "r002"],
            "decisions": [
                {
                    "page": 53,
                    "id": "r001",
                    "decision": "replace",
                    "current_translation": "我们会得到什么规律？",
                    "final_translation": "我们会得到什么图案？",
                    "reason": "pattern在此处指涂色形成的图案。",
                    "confidence": "high",
                    "child_note": "",
                },
            ],
            "human_review": [],
            "summary": "复核2条，修正1条。",
        }

    def test_load_settings_uses_configured_terra_high(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config, _page_json = self.make_fixture(pathlib.Path(temporary))
            settings = review.load_settings(config)
        self.assertEqual(settings.model, "gpt-5.6-terra")
        self.assertEqual(settings.reasoning_effort, "high")
        self.assertEqual(settings.image_mode, "on_demand")
        self.assertTrue(settings.require_chatgpt_login)

    def test_page_markdown_is_not_read_or_hashed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _config, page_json = self.make_fixture(pathlib.Path(temporary))
            page_md = page_json.with_suffix(".md")
            page_md.write_text(
                "这份 Markdown 已过期，而且包含不应进入模型的建议。\n",
                encoding="utf-8",
            )
            located_json = review.resolve_page_json_argument(
                page_json=None, page_markdown=page_md
            )
            inputs = review.load_page_inputs(located_json, include_image=True)
            self.assertNotIn(page_md.name, inputs.hashes)
            self.assertIn(page_json.name, inputs.hashes)

    def test_validate_response_requires_complete_unique_region_set(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _config, page_json = self.make_fixture(pathlib.Path(temporary))
            inputs = review.load_page_inputs(page_json, include_image=True)
            response = self.valid_response()
            response["reviewed_region_ids"] = ["r001"]
            with self.assertRaisesRegex(
                review.CodexPageReviewError, "did not review every region"
            ):
                review.validate_codex_response(inputs, response)

    def test_unchanged_decision_must_be_omitted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _config, page_json = self.make_fixture(pathlib.Path(temporary))
            inputs = review.load_page_inputs(page_json, include_image=True)
            response = self.valid_response()
            response["decisions"][0]["final_translation"] = "我们会得到什么规律？"
            with self.assertRaisesRegex(
                review.CodexPageReviewError, "must be omitted"
            ):
                review.validate_codex_response(inputs, response)

    def test_invoke_codex_uses_saved_chatgpt_login_and_no_api_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config, page_json = self.make_fixture(pathlib.Path(temporary))
            settings = review.load_settings(config)
            inputs = review.load_page_inputs(page_json, include_image=True)
            calls: list[tuple[list[str], dict]] = []

            def fake_run(args, **kwargs):
                calls.append((args, kwargs))
                self.assertNotIn("OPENAI_API_KEY", kwargs["env"])
                self.assertNotIn("CODEX_API_KEY", kwargs["env"])
                if args[1:3] == ["login", "status"]:
                    return subprocess.CompletedProcess(
                        args, 0, stdout="Logged in using ChatGPT\n", stderr=""
                    )
                schema_path = pathlib.Path(args[args.index("--output-schema") + 1])
                schema = json.loads(schema_path.read_text(encoding="utf-8"))
                self.assertEqual(schema["properties"]["page"]["const"], 53)
                self.assertIn("--ephemeral", args)
                self.assertEqual(args[args.index("--sandbox") + 1], "read-only")
                self.assertEqual(args[args.index("--model") + 1], "gpt-5.6-terra")
                self.assertIn('model_reasoning_effort="high"', args)
                self.assertIn("--json", args)
                self.assertNotIn("翻译语义复核建议", kwargs["input"])
                self.assertNotIn("suggested_translation", kwargs["input"])
                self.assertNotIn("DO_NOT_SEND_SKIPPED", kwargs["input"])
                self.assertNotIn("DO_NOT_SEND_ISSUES", kwargs["input"])
                self.assertNotIn("DO_NOT_SEND_VERIFIER", kwargs["input"])
                self.assertIn("What pattern do we get?", kwargs["input"])
                self.assertIn("page JSON's study.regions", kwargs["input"])
                result_path = pathlib.Path(
                    args[args.index("--output-last-message") + 1]
                )
                result_path.write_text(
                    json.dumps(self.valid_response(), ensure_ascii=False),
                    encoding="utf-8",
                )
                return subprocess.CompletedProcess(
                    args,
                    0,
                    stdout="\n".join(
                        [
                            json.dumps(
                                {"type": "thread.started", "thread_id": "test-thread"}
                            ),
                            json.dumps(
                                {
                                    "type": "turn.completed",
                                    "usage": {
                                        "input_tokens": 1200,
                                        "cached_input_tokens": 200,
                                        "output_tokens": 300,
                                        "reasoning_output_tokens": 100,
                                    },
                                }
                            ),
                        ]
                    ),
                    stderr="progress only",
                )

            with (
                mock.patch.dict(
                    os.environ,
                    {"OPENAI_API_KEY": "must-not-leak", "CODEX_API_KEY": "must-not-leak"},
                ),
                mock.patch("codex_page_review.shutil.which", return_value="codex.exe"),
                mock.patch("codex_page_review.subprocess.run", side_effect=fake_run),
            ):
                response, elapsed, login_status, usage = review.invoke_codex(
                    settings, inputs
                )

            validated = review.validate_codex_response(inputs, response)
            self.assertEqual(len(calls), 2)
            self.assertGreaterEqual(elapsed, 0)
            self.assertIn("ChatGPT", login_status)
            self.assertEqual(len(validated["decisions"]), 1)
            self.assertEqual(usage["input_tokens"], 1200)
            self.assertEqual(usage["cached_input_tokens"], 200)
            self.assertEqual(usage["non_cached_input_tokens"], 1000)
            self.assertEqual(usage["output_tokens"], 300)
            self.assertEqual(usage["reasoning_output_tokens"], 100)
            self.assertEqual(usage["total_tokens"], 1500)

    def test_resolve_codex_finds_windows_desktop_cli_when_conda_hides_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            local_app_data = pathlib.Path(temporary)
            executable = (
                local_app_data
                / "OpenAI"
                / "Codex"
                / "bin"
                / "version-hash"
                / "codex.exe"
            )
            executable.parent.mkdir(parents=True)
            executable.write_bytes(b"stub")
            with (
                mock.patch("codex_page_review.shutil.which", return_value=None),
                mock.patch("codex_page_review.os.name", "nt"),
                mock.patch.dict(
                    os.environ,
                    {"LOCALAPPDATA": str(local_app_data)},
                    clear=False,
                ),
            ):
                resolved = review.resolve_codex_command("codex")
            self.assertEqual(pathlib.Path(resolved), executable.resolve())

    def test_input_hash_change_is_detected_before_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _config, page_json = self.make_fixture(pathlib.Path(temporary))
            inputs = review.load_page_inputs(page_json, include_image=True)
            inputs.json_path.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(
                review.CodexPageReviewError, "Input changed while Codex was reviewing"
            ):
                review.verify_inputs_unchanged(inputs)

    def test_dry_run_does_not_invoke_codex_or_write_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config, page_json = self.make_fixture(pathlib.Path(temporary))
            output = page_json.with_name("custom-review.json")
            output.write_text('{"existing":true}\n', encoding="utf-8")
            stdout = io.StringIO()
            with (
                mock.patch(
                    "codex_page_review.invoke_codex",
                    side_effect=AssertionError("must not invoke Codex"),
                ),
                contextlib.redirect_stdout(stdout),
            ):
                status = review.main(
                    [
                        "--config",
                        str(config),
                        "--page-json",
                        str(page_json),
                        "--output",
                        str(output),
                        "--dry-run",
                    ]
                )
            summary = json.loads(stdout.getvalue())
            self.assertEqual(status, 0)
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8")), {"existing": True}
            )
            self.assertEqual(summary["model"], "gpt-5.6-terra")
            self.assertEqual(summary["reasoning_effort"], "high")
            self.assertEqual(summary["review_input_source"], "page_json.study.regions")
            self.assertFalse(summary["page_markdown_used"])
            self.assertFalse(summary["would_invoke_codex"])

    def test_main_writes_separate_validated_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config, page_json = self.make_fixture(pathlib.Path(temporary))
            output = page_json.with_name("page-0053-codex-review.json")
            stdout = io.StringIO()
            with (
                mock.patch(
                    "codex_page_review.invoke_codex",
                    return_value=(
                        self.valid_response(),
                        1.25,
                        "Logged in using ChatGPT",
                        {
                            "available": True,
                            "input_tokens": 1200,
                            "cached_input_tokens": 200,
                            "non_cached_input_tokens": 1000,
                            "output_tokens": 300,
                            "reasoning_output_tokens": 100,
                            "total_tokens": 1500,
                        },
                    ),
                ),
                contextlib.redirect_stdout(stdout),
            ):
                status = review.main(
                    [
                        "--config",
                        str(config),
                        "--page-json",
                        str(page_json),
                    ]
                )
            saved = json.loads(output.read_text(encoding="utf-8"))
            original = json.loads(page_json.read_text(encoding="utf-8"))
            self.assertEqual(status, 0)
            self.assertEqual(saved["schema_version"], 4)
            self.assertEqual(saved["review_scope"], "page")
            self.assertEqual(saved["review_input"]["source"], "page_json.study.regions")
            self.assertFalse(saved["review_input"]["page_markdown_used"])
            self.assertFalse(
                saved["review_input"]["translation_verifier_comparison_used"]
            )
            self.assertEqual(set(saved["inputs"]["files"]), {page_json.name})
            self.assertEqual(saved["inputs"]["region_ids"], ["r001", "r002"])
            self.assertEqual(saved["codex"]["model"], "gpt-5.6-terra")
            self.assertEqual(saved["codex"]["reasoning_effort"], "high")
            self.assertEqual(saved["codex"]["image_mode"], "on_demand")
            self.assertFalse(saved["codex"]["image_attached"])
            self.assertEqual(len(saved["codex"]["stages"]), 1)
            self.assertEqual(saved["codex"]["stages"][0]["name"], "text")
            self.assertTrue(saved["codex"]["chatgpt_login_confirmed"])
            self.assertEqual(saved["codex"]["usage"]["total_tokens"], 1500)
            self.assertEqual([item["id"] for item in saved["decisions"]], ["r001"])
            self.assertEqual(original["study"]["regions"][0]["translation"], "我们会得到什么规律？")

    def test_on_demand_mode_adds_image_stage_only_for_image_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config, page_json = self.make_fixture(pathlib.Path(temporary))
            settings = review.load_settings(config)
            inputs = review.load_page_inputs(page_json, include_image=True)
            text_response = self.valid_response()
            text_response["human_review"] = [
                {
                    "page": 53,
                    "id": "r002",
                    "reason": "需要确认对白对应的视觉图案。",
                    "evidence_needed": "查看页面图中的涂色图案。",
                    "evidence_type": "image",
                }
            ]
            image_response = {
                "page": 53,
                "reviewed_region_ids": ["r002"],
                "decisions": [
                    {
                        "page": 53,
                        "id": "r002",
                        "decision": "replace",
                        "current_translation": "条纹！",
                        "final_translation": "竖条纹！",
                        "reason": "页面图显示的是竖向条纹。",
                        "confidence": "high",
                        "child_note": "",
                    }
                ],
                "human_review": [],
                "summary": "图片证据足以完成裁决。",
            }
            usage_one = {
                "available": True,
                "input_tokens": 1000,
                "cached_input_tokens": 0,
                "non_cached_input_tokens": 1000,
                "output_tokens": 200,
                "reasoning_output_tokens": 50,
                "total_tokens": 1200,
            }
            usage_two = {
                "available": True,
                "input_tokens": 3000,
                "cached_input_tokens": 0,
                "non_cached_input_tokens": 3000,
                "output_tokens": 100,
                "reasoning_output_tokens": 25,
                "total_tokens": 3100,
            }
            calls = []

            def fake_invoke(_settings, stage_inputs, region_ids=None, **kwargs):
                calls.append((stage_inputs.image_path, region_ids, kwargs))
                if stage_inputs.image_path is None:
                    self.assertTrue(kwargs["allow_image_followup"])
                    return text_response, 1.0, "Logged in using ChatGPT", usage_one
                self.assertEqual(region_ids, ["r002"])
                self.assertFalse(kwargs["allow_image_followup"])
                return image_response, 2.0, "Logged in using ChatGPT", usage_two

            with mock.patch(
                "codex_page_review.invoke_codex", side_effect=fake_invoke
            ):
                validated, elapsed, _login, usage, stages, used_inputs = (
                    review.run_review_stages(settings, inputs)
                )

            self.assertEqual(len(calls), 2)
            self.assertIsNone(calls[0][0])
            self.assertIsNotNone(calls[1][0])
            self.assertEqual([item["id"] for item in validated["decisions"]], ["r001", "r002"])
            self.assertEqual(validated["human_review"], [])
            self.assertEqual(elapsed, 3.0)
            self.assertEqual(usage["total_tokens"], 4300)
            self.assertEqual([stage["name"] for stage in stages], ["text", "image"])
            self.assertIsNotNone(used_inputs.image_path)

    def test_parse_codex_usage_returns_unavailable_when_cli_omits_usage(self) -> None:
        usage = review.parse_codex_usage(
            json.dumps({"type": "thread.started", "thread_id": "test-thread"})
        )
        self.assertFalse(usage["available"])


if __name__ == "__main__":
    unittest.main()
