import json
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ocr_demo import (  # noqa: E402
    ApiResult,
    analyze_risk,
    analyze_study_risk,
    apply_thinking_setting,
    apply_chapter_adjudication,
    build_study_output,
    build_chapter_review_input,
    build_translation_context,
    call_deepl_mcp_translation,
    call_chapter_translation_verifier,
    compare_translation_verifier,
    compare_verifier,
    compact_page_label,
    collect_review_reasons,
    deepseek_regions,
    numeric_tokens,
    merge_with_existing_records,
    parse_pages,
    parse_chapter_review,
    parse_study_result,
    reverify_translation_page,
    resolve_model_configuration,
    strip_json_comments,
    write_study_html,
)
from pdf_backfill import (  # noqa: E402
    build_ai_spotcheck_request,
    build_backfill_plan,
    build_final_translation_snapshot,
    export_chinese_pdf,
    map_page_regions,
    mapping_skip_reason,
    merge_rules,
    program_check_pdf,
)


def result(content: str) -> ApiResult:
    return ApiResult("test", "test", "test", 200, 1.0, {}, content)


class OcrDemoTests(unittest.TestCase):
    def test_final_translation_snapshot_uses_adjudication_then_human_override(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source_pdf = root / "book.pdf"
            adjudication_file = root / "adjudication.json"
            source_pdf.write_bytes(b"pdf-placeholder")
            adjudication_file.write_text("{}", encoding="utf-8")
            records = [
                {
                    "page": 3,
                    "study": {
                        "regions": [
                            {
                                "id": "r001",
                                "type": "dialogue",
                                "source_text": "Hello.",
                                "translation": "你好。",
                            },
                            {
                                "id": "r002",
                                "type": "heading",
                                "source_text": "Shapes",
                                "translation": "形状",
                                "codex_adjudication": {
                                    "decision": "replace",
                                    "reason": "统一标题",
                                    "child_note": "",
                                },
                            },
                        ]
                    },
                }
            ]
            snapshot = build_final_translation_snapshot(
                records,
                source_pdf=source_pdf,
                adjudication_file=adjudication_file,
                page_record_hashes={"page-0003.json": "a" * 64},
                human_overrides={
                    (3, "r002"): {
                        "final_translation": "图形",
                        "reason": "人工最终确认",
                    }
                },
            )
            by_id = {item["id"]: item for item in snapshot["regions"]}
            self.assertEqual(by_id["r001"]["translation_source"], "saved_page_translation")
            self.assertEqual(by_id["r002"]["translation_source"], "human_override")
            self.assertEqual(by_id["r002"]["final_translation"], "图形")
            self.assertTrue(all(item["locked"] for item in snapshot["regions"]))

    def test_page_mapping_assigns_repeated_labels_in_order(self):
        regions = [
            {
                "id": "r001",
                "type": "caption",
                "source_text": "Player 1",
                "final_translation": "玩家1",
                "translation_source": "saved_page_translation",
            },
            {
                "id": "r002",
                "type": "caption",
                "source_text": "Player 1",
                "final_translation": "玩家1",
                "translation_source": "saved_page_translation",
            },
        ]
        layout = (
            "<|ref|>Player 1<|/ref|><|det|>[[100,100,180,120]]<|/det|>\n"
            "<|ref|>Player 1<|/ref|><|det|>[[500,500,580,520]]<|/det|>"
        )
        mappings = map_page_regions(8, regions, layout)
        self.assertEqual(mappings[0]["placement_box"], [100.0, 100.0, 180.0, 120.0])
        self.assertEqual(mappings[1]["placement_box"], [500.0, 500.0, 580.0, 520.0])
        self.assertEqual([item["confidence"] for item in mappings], ["high", "high"])

    def test_backfill_plan_keeps_unmatched_region_as_explicit_warning(self):
        snapshot = {
            "status": "locked_final_translation",
            "regions": [
                {
                    "page": 9,
                    "id": "r001",
                    "type": "heading",
                    "source_text": "RECESS",
                    "final_translation": "课间休息",
                    "translation_source": "saved_page_translation",
                }
            ],
        }
        records = [{"page": 9, "layout": {"content": ""}}]
        plan = build_backfill_plan(snapshot, records)
        self.assertEqual(plan["mapped_count"], 0)
        self.assertEqual(plan["unmapped_count"], 1)
        self.assertEqual(plan["mappings"][0]["warnings"], ["layout_not_found"])

    def test_page_mapping_never_reuses_one_layout_item(self):
        regions = [
            {
                "id": "r001",
                "type": "dialogue",
                "source_text": "Professor Grok is gone",
                "final_translation": "格罗克教授不见了",
                "translation_source": "saved_page_translation",
            },
            {
                "id": "r002",
                "type": "dialogue",
                "source_text": "Professor Grok is locked in a room",
                "final_translation": "格罗克教授被锁在房间里",
                "translation_source": "saved_page_translation",
            },
        ]
        layout = (
            "<|ref|>Professor Grok is<|/ref|><|det|>[[100,100,260,120]]<|/det|>\n"
            "<|ref|>gone<|/ref|><|det|>[[110,122,160,140]]<|/det|>\n"
            "<|ref|>locked in a room<|/ref|><|det|>[[600,100,760,120]]<|/det|>"
        )
        mappings = map_page_regions(8, regions, layout)
        first = set(mappings[0]["layout_item_indexes"])
        second = set(mappings[1]["layout_item_indexes"])
        self.assertFalse(first & second)
        plan = build_backfill_plan(
            {"status": "locked_final_translation", "regions": [dict(item, page=8) for item in regions]},
            [{"page": 8, "layout": {"content": layout}}],
        )
        self.assertEqual(plan["layout_item_ownership_conflict_count"], 0)

    def test_compact_page_label(self):
        self.assertEqual(compact_page_label([18, 19, 23, 26, 27]), "18-19_23_26-27")

    def test_backfill_policy_skips_only_risky_or_dense_geometry_labels(self):
        rules = merge_rules({})
        low = {
            "type": "dialogue",
            "source_text": "Hello",
            "match_coverage": 0.5,
            "warnings": [],
        }
        self.assertEqual(
            mapping_skip_reason(low, [low], rules),
            "skipped_low_confidence",
        )
        ambiguous = {
            "type": "dialogue",
            "source_text": "Hello there",
            "match_coverage": 0.8,
            "warnings": ["ambiguous_layout_candidates"],
        }
        self.assertEqual(
            mapping_skip_reason(ambiguous, [ambiguous], rules),
            "skipped_ambiguous_mapping",
        )
        geometry_page = [
            {
                "type": "heading" if index == 0 else "definition",
                "source_text": text,
                "match_coverage": 1.0,
                "warnings": [],
            }
            for index, text in enumerate(
                (
                    "Triangles",
                    "Equilateral triangle",
                    "Acute angle",
                    "Isosceles triangle",
                    "Right angle",
                    "Scalene triangle",
                )
            )
        ]
        self.assertIsNone(mapping_skip_reason(geometry_page[1], geometry_page, rules))
        geometry_filter_rules = merge_rules({"skip_dense_geometry_labels": True})
        self.assertIsNone(
            mapping_skip_reason(
                geometry_page[0], geometry_page, geometry_filter_rules
            )
        )
        self.assertEqual(
            mapping_skip_reason(
                geometry_page[1], geometry_page, geometry_filter_rules
            ),
            "skipped_geometry_label",
        )

    def test_ai_spotcheck_is_only_a_ranked_request(self):
        plan = {
            "mappings": [
                {"page": 5, "confidence": "missing", "warnings": ["layout_not_found"]},
                {"page": 6, "confidence": "medium", "warnings": []},
            ]
        }
        rules = merge_rules({"ai_check": "spot", "ai_sample_count": 1})
        request = build_ai_spotcheck_request(plan, rules)
        self.assertEqual(request["pages"][0]["page"], 5)
        self.assertIn("Human visual acceptance", request["instruction"])

    def test_small_pdf_backfill_and_program_check(self):
        try:
            import fitz
        except ImportError:
            self.skipTest("PyMuPDF is not installed")
        font_file = pathlib.Path(
            "/System/Library/AssetsV2/com_apple_MobileAsset_Font7/"
            "eb257c12d1a51c8c661b89f30eec56cacf9b8987.asset/AssetData/STHEITI.ttf"
        )
        if not font_file.is_file():
            self.skipTest("STHeiti font is not available")
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source_pdf = root / "source.pdf"
            output_pdf = root / "output.pdf"
            source = fitz.open()
            page = source.new_page(width=300, height=400)
            page.draw_rect(page.rect, fill=(0.95, 0.95, 0.9), color=None)
            page.insert_text((60, 90), "Hello", fontsize=14)
            source.save(source_pdf)
            source.close()
            plan = {
                "region_count": 1,
                "unmapped_count": 0,
                "mappings": [
                    {
                        "page": 1,
                        "id": "r001",
                        "type": "dialogue",
                        "source_text": "Hello",
                        "final_translation": "这是一段需要根据文本框大小自动缩小字号的较长中文译文",
                        "translation_source": "codex_adjudication",
                        "placement_box": [180, 170, 380, 240],
                        "erase_boxes": [
                            [180, 170, 380, 205],
                            [180, 205, 380, 240],
                        ],
                        "match_coverage": 1.0,
                        "confidence": "high",
                        "warnings": [],
                    }
                ],
            }
            rules = merge_rules(
                {
                    "ai_check": "none",
                    "minimum_font_size": 7.0,
                    "maximum_vertical_overflow_ratio": 1.1,
                }
            )
            with mock.patch(
                "pdf_backfill._sample_background",
                side_effect=[(1.0, 0.0, 0.0), (0.0, 0.0, 1.0)],
            ):
                exported = export_chinese_pdf(
                    source_pdf=source_pdf,
                    pages=[1],
                    plan=plan,
                    rules=rules,
                    layout_overrides={},
                    font_file=font_file,
                    output_pdf=output_pdf,
                )
            snapshot = {
                "status": "locked_final_translation",
                "region_count": 1,
                "regions": [{"locked": True}],
            }
            checked = program_check_pdf(
                source_pdf=source_pdf,
                output_pdf=output_pdf,
                pages=[1],
                snapshot=snapshot,
                plan=plan,
                export_result=exported,
            )
            self.assertEqual(exported["status_counts"], {"written": 1})
            self.assertEqual(exported["regions"][0]["erase_fill_color"], "#800080")
            self.assertLess(exported["regions"][0]["font_size"], 11.0)
            self.assertEqual(checked["status"], "program_checked")
            self.assertEqual(checked["visual_review"], "not_performed_human_required")

    def test_json_comments_preserve_urls_inside_strings(self):
        source = '''{
          // 这是一行注释
          "endpoint": "https://mcp.deepl.com/v1/mcp",
          /* 这是块注释 */
          "enabled": true
        }'''
        parsed = json.loads(strip_json_comments(source))
        self.assertEqual(parsed["endpoint"], "https://mcp.deepl.com/v1/mcp")
        self.assertTrue(parsed["enabled"])

    def test_parse_pages(self):
        self.assertEqual(parse_pages("3,1-2,3,5"), [1, 2, 3, 5])

    def test_parse_pages_rejects_reverse_range(self):
        with self.assertRaises(ValueError):
            parse_pages("5-3")

    def test_deepseek_regions(self):
        content = "<|ref|>7 feet<|/ref|><|det|>[[1,2,3,4]]<|/det|>"
        self.assertEqual(
            deepseek_regions(content),
            [{"text": "7 feet", "box": [[1, 2, 3, 4]]}],
        )

    def test_page_number_risk(self):
        layout = result("<|ref|>text<|/ref|><|det|>[[1,2,3,4]]<|/det|>")
        primary = result("1. text\n2. 72+2. I know!")
        risk = analyze_risk(layout, primary, page=73, printed_page_offset=-1)
        self.assertIn("suspicious_page_number_merge", risk["flags"])

    def test_formula_is_not_page_number_merge(self):
        layout = result("<|ref|>7+7=14<|/ref|><|det|>[[1,2,3,4]]<|/det|>")
        primary = result("1. 7+7=14\n2. 72")
        risk = analyze_risk(layout, primary, page=73, printed_page_offset=-1)
        self.assertNotIn("suspicious_page_number_merge", risk["flags"])

    def test_numeric_tokens_ignore_numbered_list_markers(self):
        self.assertEqual(numeric_tokens("1. 7 feet\n2. 8 feet"), {"7", "8"})

    def test_verifier_numeric_disagreement(self):
        comparison = compare_verifier(result("1. 7 feet"), result("1. 8 feet"))
        self.assertTrue(comparison["needs_human_review"])
        self.assertEqual(comparison["primary_only_numbers"], ["7"])
        self.assertEqual(comparison["verifier_only_numbers"], ["8"])

    def test_small_verifier_hallucination_requires_review(self):
        comparison = compare_verifier(
            result("1. Urrrrggh! No! No! No!"),
            result("1. Urrrr1. Urrrrggh! No! No! No!"),
        )
        reasons = collect_review_reasons({"flags": []}, comparison)
        self.assertIn("primary_verifier_text_disagreement", reasons)

    def test_parse_study_json_from_fenced_response(self):
        parsed = parse_study_result(
            '```json\n{"regions":[{"id":"r001","type":"dialogue",'
            '"source_text":"How many?"}],"skipped":{}}\n```'
        )
        self.assertEqual(parsed["regions"][0]["source_text"], "How many?")

    def test_study_risk_ignores_excluded_visual_content(self):
        layout = result(
            "<|ref|>BAM!<|/ref|><|det|>[[1,2,3,4]]<|/det|>"
            "<|ref|>7+7=14<|/ref|><|det|>[[5,6,7,8]]<|/det|>"
            "<|ref|>100<|/ref|><|det|>[[9,10,11,12]]<|/det|>"
        )
        primary = result('{"regions":[],"skipped":{"numeric_tables":true}}')
        risk = analyze_study_risk(layout, primary)
        self.assertEqual(risk["flags"], [])

    def test_study_risk_flags_missing_natural_language(self):
        layout = result(
            "<|ref|>What is the area?<|/ref|><|det|>[[1,2,3,4]]<|/det|>"
        )
        primary = result('{"regions":[],"skipped":{}}')
        risk = analyze_study_risk(layout, primary)
        self.assertIn("possible_translatable_text_omission", risk["flags"])

    def test_translation_is_merged_by_region_id(self):
        primary = result(
            '{"regions":[{"id":"r001","type":"question",'
            '"source_text":"How many?"}],"skipped":{}}'
        )
        translation = result(
            '{"translations":[{"id":"r001","translation":"有多少个？"}]}'
        )
        study, reasons = build_study_output(primary, translation)
        self.assertEqual(study["regions"][0]["translation"], "有多少个？")
        self.assertEqual(reasons, [])

    def test_translation_expansion_requires_review(self):
        primary = result(
            '{"regions":[{"id":"r001","type":"instruction",'
            '"source_text":"Use the hundred chart to count by five."}],"skipped":{}}'
        )
        translation = result(
            '{"translations":[{"id":"r001","translation":"'
            + "这是不应该出现的额外说明。" * 20
            + '"}]}'
        )
        _, reasons = build_study_output(primary, translation)
        self.assertIn("translation_suspicious_expansion", reasons)

    def test_deepl_mcp_translation_preserves_region_ids(self):
        class FakeBridge:
            def __init__(self):
                self.requests = []

            def request(self, payload):
                self.requests.append(payload)
                return {
                    "detectedSourceLanguage": "EN",
                    "text": f"译文:{payload['text']}",
                }

        bridge = FakeBridge()
        args = types.SimpleNamespace(
            deepl_node_command="node",
            deepl_bridge_script=pathlib.Path("deepl_mcp_client.mjs"),
            deepl_mcp_endpoint="https://mcp.deepl.com/v1/mcp",
            deepl_oauth_callback_port=8765,
            deepl_oauth_keychain_service="test",
            deepl_oauth_keychain_account="test",
            deepl_source_lang="EN",
            deepl_target_lang="ZH-HANS",
            deepl_formality="",
            deepl_glossary_id="",
            deepl_style_id="",
            deepl_context="math book",
            deepl_custom_instructions=[],
        )
        with mock.patch("ocr_demo.get_deepl_bridge", return_value=bridge):
            translated = call_deepl_mcp_translation(
                args,
                [
                    {"id": "r001", "source_text": "How many?"},
                    {"id": "r002", "source_text": "Try again."},
                ],
            )
        content = json.loads(translated.content)
        self.assertIsNone(translated.error)
        self.assertEqual(
            [item["id"] for item in content["translations"]],
            ["r001", "r002"],
        )
        self.assertEqual(len(bridge.requests), 2)
        self.assertIn("r002: Try again.", bridge.requests[0]["context"])

    def test_translation_context_is_page_aware_and_bounded(self):
        regions = [
            {"id": "r001", "source_text": "Acute"},
            {"id": "r002", "source_text": "Triangle angle types"},
        ]
        context = build_translation_context("Grade 3 math", regions, 0, limit=120)
        self.assertIn("Acute", context)
        self.assertIn("Triangle angle types", context)
        self.assertLessEqual(len(context), 120)

    def test_translation_verifier_flags_advisory_issue(self):
        regions = [
            {
                "id": "r001",
                "type": "caption",
                "source_text": "Acute",
                "translation": "急性",
            }
        ]
        verifier = result(
            '{"issues":[{"id":"r001","category":"math_term",'
            '"explanation":"几何术语错误","suggested_translation":"锐角"}]}'
        )
        comparison = compare_translation_verifier(regions, verifier)
        self.assertIsNone(comparison["error"])
        self.assertEqual(comparison["issues"][0]["suggested_translation"], "锐角")

    def test_translation_verifier_keeps_issue_without_suggestion(self):
        regions = [
            {
                "id": "r001",
                "type": "caption",
                "source_text": "Acute",
                "translation": "急性",
            }
        ]
        verifier = result(
            '{"issues":[{"id":"r001","category":"math_term",'
            '"explanation":"术语需要人工判断"}]}'
        )
        comparison = compare_translation_verifier(regions, verifier)
        self.assertIsNone(comparison["error"])
        self.assertEqual(len(comparison["issues"]), 1)
        self.assertEqual(comparison["issues"][0]["suggested_translation"], "")

    def test_subset_maintenance_preserves_existing_aggregate_pages(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = pathlib.Path(temporary)
            pages_dir = output / "pages"
            pages_dir.mkdir()
            for page in (13, 14, 15):
                (pages_dir / f"page-{page:04d}.json").write_text(
                    json.dumps({"page": page, "value": f"old-{page}"}),
                    encoding="utf-8",
                )
            (output / "manifest.json").write_text(
                json.dumps({"pages": [13, 14, 15]}), encoding="utf-8"
            )
            records = merge_with_existing_records(
                output, [{"page": 14, "value": "updated"}]
            )
            self.assertEqual([record["page"] for record in records], [13, 14, 15])
            self.assertEqual(records[1]["value"], "updated")

    def test_translation_reverify_does_not_change_saved_translation(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = pathlib.Path(temporary)
            pages_dir = output / "pages"
            pages_dir.mkdir()
            primary = result(
                '{"regions":[{"id":"r001","type":"caption",'
                '"source_text":"Acute"}],"skipped":{}}'
            )
            translation = result(
                '{"translations":[{"id":"r001","translation":"急性"}]}'
            )
            record = {
                "page": 23,
                "mode": "study",
                "image": "pages/page-0023.png",
                "layout": result("").as_dict(),
                "primary": primary.as_dict(),
                "risk": {"flags": []},
                "verifier": None,
                "translation": translation.as_dict(),
                "study": {
                    "regions": [
                        {
                            "id": "r001",
                            "type": "caption",
                            "source_text": "Acute",
                            "translation": "急性",
                        }
                    ],
                    "skipped": {},
                    "error": None,
                },
                "verifier_comparison": {"available": False},
                "review_reasons": [],
                "review_required": False,
            }
            page_json = pages_dir / "page-0023.json"
            page_json.write_text(json.dumps(record), encoding="utf-8")
            verifier = result(
                '{"issues":[{"id":"r001","category":"math_term",'
                '"explanation":"几何术语错误","suggested_translation":"锐角"}]}'
            )
            args = types.SimpleNamespace(output=output)
            with mock.patch(
                "ocr_demo.call_translation_verifier", return_value=verifier
            ):
                updated = reverify_translation_page(args, 23)
            self.assertEqual(
                updated["study"]["regions"][0]["translation"], "急性"
            )
            self.assertIn("translation_semantic_issue", updated["review_reasons"])
            self.assertEqual(
                updated["translation_verifier_comparison"]["issues"][0][
                    "suggested_translation"
                ],
                "锐角",
            )

    def test_chapter_review_input_is_blind_and_contains_saved_pairs(self):
        args = types.SimpleNamespace(
            deepl_custom_instructions=["Keep Beast Academy unchanged."]
        )
        records = [
            {
                "page": 18,
                "mode": "study",
                "translation_verifier_comparison": {
                    "issues": [{"id": "r001", "reason": "must stay hidden"}]
                },
                "study": {
                    "regions": [
                        {
                            "id": "r001",
                            "type": "heading",
                            "source_text": "Acute Angles",
                            "translation": "锐角",
                        },
                        {
                            "id": "r002",
                            "type": "dialogue",
                            "source_text": "Professor Grok is here.",
                            "translation": "格罗克教授在这里。",
                        },
                    ]
                },
            }
        ]
        review_input = build_chapter_review_input(args, records)
        self.assertIn("[p18-r001]", review_input)
        self.assertIn("English: Acute Angles", review_input)
        self.assertIn("Chinese: 锐角", review_input)
        self.assertIn("Professor Grok", review_input)
        self.assertNotIn("must stay hidden", review_input)

    def test_parse_chapter_review_validates_page_and_region(self):
        content = json.dumps(
            {
                "issues": [
                    {
                        "page": 18,
                        "id": "r003",
                        "category": "wordplay",
                        "suggested_translation": "肥角",
                        "reason": "原译丢失语音双关",
                    }
                ],
                "global_consistency": [
                    {
                        "category": "term",
                        "item": "side",
                        "preferred_translation": "边",
                        "reason": "统一术语",
                        "affected": [{"page": 18, "id": "r003"}],
                    }
                ],
            },
            ensure_ascii=False,
        )
        parsed = parse_chapter_review(content, {(18, "r003")})
        self.assertEqual(parsed["issues"][0]["category"], "wordplay")
        self.assertEqual(
            parsed["global_consistency"][0]["affected"],
            [{"page": 18, "id": "r003"}],
        )
        with self.assertRaises(ValueError):
            parse_chapter_review(content, {(18, "r999")})

    def test_chapter_review_uses_independent_model_argument(self):
        captured = {}

        def fake_post(endpoint, payload, api_key):
            captured.update(payload)
            return 200, {
                "choices": [
                    {
                        "message": {
                            "content": '{"issues":[],"global_consistency":[]}'
                        }
                    }
                ],
                "usage": {},
            }

        args = types.SimpleNamespace()
        model_config = {
            "name": "test-profile",
            "provider": "test-provider",
            "model": "qwen-test-long-context",
            "endpoint": "https://example.test/chat/completions",
            "api_key": "test-key",
            "enable_thinking": False,
            "max_tokens": 4096,
        }
        with mock.patch("ocr_demo.post_json", side_effect=fake_post):
            response = call_chapter_translation_verifier(
                args, "review input", model_config
            )
        self.assertIsNone(response.error)
        self.assertEqual(captured["model"], "qwen-test-long-context")
        self.assertEqual(response.model, "qwen-test-long-context")

    def test_thinking_setting_uses_provider_specific_request_field(self):
        deepseek_payload = {}
        apply_thinking_setting(deepseek_payload, "deepseek", False)
        self.assertEqual(
            deepseek_payload,
            {"thinking": {"type": "disabled"}},
        )

        qwen_payload = {}
        apply_thinking_setting(qwen_payload, "dashscope", True)
        self.assertEqual(qwen_payload, {"enable_thinking": True})

        omitted_payload = {}
        apply_thinking_setting(omitted_payload, "deepseek", None)
        self.assertEqual(omitted_payload, {})

    def test_model_profiles_are_separate_from_ocr_and_review_usage(self):
        profiles = {
            "layout-a": {
                "provider": "layout-provider",
                "model": "layout-model-id",
                "api_key_name": "siliconflow",
                "chat_endpoint": "https://layout.test/chat",
                "enable_thinking": False,
                "max_tokens": 4096,
            },
            "text-a": {
                "provider": "text-provider",
                "model": "text-model-id",
                "api_key_name": "dashscope",
                "chat_endpoint": "https://text.test/chat",
                "responses_endpoint": "https://text.test/responses",
                "enable_thinking": True,
                "max_tokens": 8192,
            },
            "review-b": {
                "provider": "review-provider",
                "model": "review-model-id",
                "api_key_name": "deepseek",
                "chat_endpoint": "https://review.test/chat",
                "responses_endpoint": "https://review.test/responses",
                "enable_thinking": None,
                "max_tokens": 16384,
            },
        }
        args = types.SimpleNamespace(
            model_profiles=profiles,
            model_usage={
                "layout_ocr": "layout-a",
                "primary_ocr": "text-a",
                "ocr_verifier": "review-b",
                "qwen_translation": "text-a",
                "translation_verifier": "review-b",
                "chapter_review": ["text-a", "review-b"],
            },
            layout_profile=None,
            primary_profile=None,
            verifier_profile=None,
            translation_profile=None,
            translation_verifier_profile=None,
            chapter_review_profile=None,
            siliconflow_key="layout-key",
            dashscope_key="text-key",
            deepseek_key="review-key",
        )
        resolve_model_configuration(args)
        self.assertEqual(args.layout_profile_config["name"], "layout-a")
        self.assertEqual(args.primary_profile_config["name"], "text-a")
        self.assertEqual(args.verifier_profile_config["name"], "review-b")
        self.assertEqual(
            [item["name"] for item in args.resolved_chapter_review_models],
            ["text-a", "review-b"],
        )
        self.assertEqual(args.primary_profile_config["api_key"], "text-key")
        self.assertEqual(
            args.resolved_chapter_review_models[1]["api_key"], "review-key"
        )

    def test_study_html_contains_image_and_translation(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = pathlib.Path(temporary)
            args = types.SimpleNamespace(
                pdf=pathlib.Path("book.pdf"), output=output
            )
            records = [
                {
                    "page": 25,
                    "image": "pages/page-0025.png",
                    "review_required": False,
                    "review_reasons": [],
                    "study": {
                        "regions": [
                            {
                                "id": "r001",
                                "type": "dialogue",
                                "source_text": "How many?",
                                "translation": "有多少个？",
                            }
                        ]
                    },
                }
            ]
            write_study_html(args, records)
            rendered = (output / "study.html").read_text(encoding="utf-8")
            self.assertIn("pages/page-0025.png", rendered)
            self.assertIn("有多少个？", rendered)

    def test_adjudication_builds_copy_without_changing_saved_record(self):
        records = [
            {
                "page": 18,
                "review_required": True,
                "review_reasons": ["translation_semantic_issue"],
                "study": {
                    "regions": [
                        {
                            "id": "r003",
                            "type": "dialogue",
                            "source_text": "I think they're called obese angles.",
                            "translation": "我觉得它们被称为钝角。",
                        }
                    ]
                },
            }
        ]
        adjudication = {
            "schema_version": 1,
            "decisions": [
                {
                    "page": 18,
                    "id": "r003",
                    "decision": "replace",
                    "final_translation": "我觉得它们叫肥角。",
                    "child_note": "这是一个误说笑点。",
                    "reason": "保留双关。",
                    "confidence": 0.98,
                }
            ],
            "human_review": [],
        }
        revised, application = apply_chapter_adjudication(records, adjudication)
        self.assertEqual(
            records[0]["study"]["regions"][0]["translation"],
            "我觉得它们被称为钝角。",
        )
        self.assertEqual(
            revised[0]["study"]["regions"][0]["translation"],
            "我觉得它们叫肥角。",
        )
        self.assertFalse(revised[0]["review_required"])
        self.assertEqual(application["applied_count"], 1)

    def test_reviewed_html_contains_adjudication_and_child_note(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = pathlib.Path(temporary)
            args = types.SimpleNamespace(
                pdf=pathlib.Path("book.pdf"), output=output
            )
            records = [
                {
                    "page": 18,
                    "image": "pages/page-0018.png",
                    "review_required": False,
                    "review_reasons": [],
                    "study": {
                        "regions": [
                            {
                                "id": "r003",
                                "type": "dialogue",
                                "source_text": "obese angles",
                                "translation": "肥角",
                                "codex_adjudication": {
                                    "decision": "replace",
                                    "reason": "保留双关。",
                                    "confidence": 0.98,
                                    "child_note": "后面会纠正为钝角。",
                                },
                            }
                        ]
                    },
                }
            ]
            write_study_html(
                args,
                records,
                output_name="study-reviewed.html",
                reviewed=True,
            )
            rendered = (output / "study-reviewed.html").read_text(encoding="utf-8")
            self.assertIn("Codex已修正", rendered)
            self.assertIn("后面会纠正为钝角", rendered)
            self.assertIn("肥角", rendered)


if __name__ == "__main__":
    unittest.main()
