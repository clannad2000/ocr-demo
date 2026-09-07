import json
import pathlib
import tempfile
import unittest

from pipeline import codex_review as book
from pipeline import page_review as page


class CodexBookReviewTests(unittest.TestCase):
    def test_parse_page_spec(self) -> None:
        self.assertEqual(book.parse_page_spec("6,5-6,8"), [5, 6, 8])
        with self.assertRaisesRegex(book.BookReviewError, "Invalid page range"):
            book.parse_page_spec("6-5")

    def test_shared_codex_review_settings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = pathlib.Path(temporary) / "pipeline.json"
            config.write_text(
                """{
                  "codex_review": {
                    "model": "gpt-5.6-terra",
                    "reasoning_effort": "high",
                    "image_mode": "on_demand",
                    "exclude_back_matter": true,
                    "toc_image_detail": "high"
                  }
                }""",
                encoding="utf-8",
            )
            settings, _data = book.load_settings(config)
        self.assertEqual(settings.model, "gpt-5.6-terra")
        self.assertEqual(settings.reasoning_effort, "high")
        self.assertTrue(settings.exclude_back_matter)

    @staticmethod
    def toc_response() -> dict:
        return {
            "pdf_page_count": 104,
            "toc_pdf_pages": [5, 6],
            "printed_to_pdf_offset": 1,
            "entries": [
                {
                    "kind": "front_matter",
                    "title": "Characters",
                    "printed_start_page": 6,
                    "source_toc_pdf_page": 5,
                    "parent_chapter_number": None,
                    "confidence": "high",
                },
                {
                    "kind": "front_matter",
                    "title": "How to Use This Book",
                    "printed_start_page": 8,
                    "source_toc_pdf_page": 5,
                    "parent_chapter_number": None,
                    "confidence": "high",
                },
                {
                    "kind": "chapter",
                    "title": "Chapter 1: Shapes",
                    "printed_start_page": 12,
                    "source_toc_pdf_page": 5,
                    "parent_chapter_number": 1,
                    "confidence": "high",
                },
                {
                    "kind": "section",
                    "title": "Angles",
                    "printed_start_page": 14,
                    "source_toc_pdf_page": 5,
                    "parent_chapter_number": 1,
                    "confidence": "high",
                },
                {
                    "kind": "chapter",
                    "title": "Chapter 2: Skip-Counting",
                    "printed_start_page": 42,
                    "source_toc_pdf_page": 5,
                    "parent_chapter_number": 2,
                    "confidence": "high",
                },
                {
                    "kind": "chapter",
                    "title": "Chapter 3: Perimeter and Area",
                    "printed_start_page": 66,
                    "source_toc_pdf_page": 5,
                    "parent_chapter_number": 3,
                    "confidence": "high",
                },
                {
                    "kind": "section",
                    "title": "Grogg's Notes",
                    "printed_start_page": 99,
                    "source_toc_pdf_page": 6,
                    "parent_chapter_number": 3,
                    "confidence": "high",
                },
                {
                    "kind": "index",
                    "title": "Index",
                    "printed_start_page": 100,
                    "source_toc_pdf_page": 6,
                    "parent_chapter_number": None,
                    "confidence": "high",
                },
            ],
            "human_review": [],
            "summary": "Three chapters and an index.",
        }

    def test_build_plan_maps_printed_pages_and_excludes_index(self) -> None:
        response = book.validate_toc_response(
            self.toc_response(), pdf_pages=104, toc_pages=[5, 6]
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            pdf = root / "book.pdf"
            pdf.write_bytes(b"pdf")
            image5 = root / "page-0005.png"
            image6 = root / "page-0006.png"
            image5.write_bytes(b"five")
            image6.write_bytes(b"six")
            plan = book.build_book_plan(
                response,
                pdf_path=pdf.resolve(),
                toc_images=[image5, image6],
                exclude_preliminary=False,
                exclude_back_matter=True,
            )
        self.assertEqual(
            [(task["pdf_start_page"], task["pdf_end_page"]) for task in plan["tasks"]],
            [(1, 6), (7, 12), (13, 42), (43, 66), (67, 100)],
        )
        self.assertEqual(plan["tasks"][0]["task_id"], "preliminary")
        self.assertFalse(plan["policy"]["exclude_preliminary"])
        self.assertNotIn("preliminary", [item["kind"] for item in plan["excluded_ranges"]])
        self.assertEqual(plan["excluded_ranges"][-1]["kind"], "index")
        self.assertEqual(plan["excluded_ranges"][-1]["pdf_start_page"], 101)
        self.assertEqual(plan["excluded_ranges"][-1]["pdf_end_page"], 104)

    def test_page_user_input_contains_only_study_regions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            page_json = pathlib.Path(temporary) / "page-0013.json"
            page_json.write_text(
                json.dumps(
                    {
                        "page": 13,
                        "mode": "study",
                        "study": {
                            "regions": [
                                {
                                    "id": "r001",
                                    "type": "question",
                                    "source_text": "What shape?",
                                    "translation": "什么形状？",
                                }
                            ],
                            "skipped": [{"source_text": "DO_NOT_SEND_SKIPPED"}],
                        },
                        "issues": ["DO_NOT_SEND_ISSUES"],
                        "translation_verifier_comparison": {
                            "issues": ["DO_NOT_SEND_VERIFIER"]
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            inputs = page.load_page_inputs(page_json, include_image=False)
            items = book.page_user_input(inputs, include_image=False)
        text = items[0]["text"]
        self.assertIn("What shape?", text)
        self.assertNotIn("DO_NOT_SEND_SKIPPED", text)
        self.assertNotIn("DO_NOT_SEND_ISSUES", text)
        self.assertNotIn("DO_NOT_SEND_VERIFIER", text)

    def test_normalize_usage_preserves_cache_breakdown(self) -> None:
        usage = book.normalize_usage(
            {
                "inputTokens": 1000,
                "cachedInputTokens": 700,
                "cacheWriteInputTokens": 20,
                "outputTokens": 100,
                "reasoningOutputTokens": 40,
                "totalTokens": 1100,
            }
        )
        self.assertEqual(usage["cached_input_tokens"], 700)
        self.assertEqual(usage["non_cached_input_tokens"], 300)

    def test_run_turn_reads_structured_message_and_usage(self) -> None:
        class FakeServer(book.CodexAppServer):
            def __init__(self) -> None:
                super().__init__("codex", timeout_seconds=5)
                self.events = collections.deque()

            def request(self, method, params, **_kwargs):
                self.assert_request = (method, params)
                return {"turn": {"id": "turn-1"}}

            def _next_event(self, _timeout_seconds=None):
                return self.events.popleft()

        import collections

        server = FakeServer()
        server.events.extend(
            [
                {
                    "method": "thread/tokenUsage/updated",
                    "params": {
                        "threadId": "thread-1",
                        "turnId": "turn-1",
                        "tokenUsage": {
                            "last": {
                                "inputTokens": 100,
                                "cachedInputTokens": 50,
                                "outputTokens": 10,
                                "reasoningOutputTokens": 4,
                                "totalTokens": 110,
                            }
                        },
                    },
                },
                {
                    "method": "item/completed",
                    "params": {
                        "threadId": "thread-1",
                        "turnId": "turn-1",
                        "completedAtMs": 1,
                        "item": {
                            "id": "message-1",
                            "type": "agentMessage",
                            "text": '{"ok":true}',
                        },
                    },
                },
                {
                    "method": "turn/completed",
                    "params": {
                        "threadId": "thread-1",
                        "turn": {"id": "turn-1", "status": "completed"},
                    },
                },
            ]
        )
        result = server.run_turn(
            thread_id="thread-1",
            input_items=[{"type": "text", "text": "test"}],
            output_schema={"type": "object"},
            effort="high",
        )
        self.assertEqual(result.response, {"ok": True})
        self.assertEqual(result.usage["cached_input_tokens"], 50)


if __name__ == "__main__":
    unittest.main()
