import json
import pathlib
import tempfile
import unittest

import codex_book_review as book
import codex_page_review as page


class CodexBookReviewTests(unittest.TestCase):
    @staticmethod
    def make_page_inputs(
        root: pathlib.Path, page_number: int, region_count: int
    ) -> page.PageReviewInputs:
        page_json = root / f"page-{page_number:04d}.json"
        page_json.write_text(
            json.dumps(
                {
                    "page": page_number,
                    "mode": "study",
                    "study": {
                        "regions": [
                            {
                                "id": f"r{index:03d}",
                                "type": "question",
                                "source_text": f"Question {index}",
                                "translation": f"问题{index}",
                            }
                            for index in range(1, region_count + 1)
                        ]
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return page.load_page_inputs(page_json, include_image=False)

    def test_parse_page_spec(self) -> None:
        self.assertEqual(book.parse_page_spec("6,5-6,8"), [5, 6, 8])
        with self.assertRaisesRegex(book.BookReviewError, "Invalid page range"):
            book.parse_page_spec("6-5")

    def test_settings_inherit_page_review_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = pathlib.Path(temporary) / "ocr_config.json"
            config.write_text(
                """{
                  "codex_page_review": {
                    "model": "gpt-5.6-terra",
                    "reasoning_effort": "high",
                    "image_mode": "on_demand"
                  },
                  "codex_book_review": {
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

    def test_example_config_parses_annotated_batching_settings(self) -> None:
        config = pathlib.Path(book.__file__).resolve().with_name(
            "ocr_config.example.json"
        )
        settings, _data = book.load_settings(config)
        self.assertTrue(settings.batching_enabled)
        self.assertEqual(settings.batch_target_pages, 6)
        self.assertEqual(settings.thread_target_batches, 4)
        self.assertEqual(settings.thread_max_batches, 6)
        self.assertEqual(settings.model_context_window_tokens, 1050000)
        self.assertEqual(settings.context_hard_ratio, 0.90)
        self.assertTrue(settings.fixed_translation_instructions)

    def test_content_batches_use_soft_targets_and_preserve_gaps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            inputs = [
                self.make_page_inputs(root, 1, 3),
                self.make_page_inputs(root, 2, 3),
                self.make_page_inputs(root, 4, 1),
            ]
            settings = book.BookSettings(
                batch_target_pages=6,
                batch_target_regions=5,
                batch_target_tokens=100000,
            )
            batches = book.plan_task_batches(
                {"task_id": "chapter-01", "title": "Shapes"},
                inputs,
                settings,
            )
        self.assertEqual([item["pages"] for item in batches], [[1, 2], [4]])
        self.assertEqual(batches[0]["region_count"], 6)

    def test_nine_batches_are_balanced_into_five_and_four(self) -> None:
        settings = book.BookSettings(
            thread_target_batches=4,
            thread_max_batches=6,
        )
        batches = [{"batch_id": str(index)} for index in range(9)]
        groups = book.group_batches_for_threads(batches, settings)
        self.assertEqual([len(group) for group in groups], [5, 4])

    def test_batch_response_is_validated_and_split_by_page(self) -> None:
        class FakeServer:
            def run_turn(self, **kwargs):
                schema = kwargs["output_schema"]
                batch_id = schema["properties"]["batch_id"]["const"]
                page_results = {}
                for key, page_schema in schema["properties"]["page_results"][
                    "properties"
                ].items():
                    page_number = page_schema["properties"]["page"]["const"]
                    region_ids = page_schema["properties"]["reviewed_region_ids"][
                        "items"
                    ]["enum"]
                    page_results[key] = {
                        "page": page_number,
                        "reviewed_region_ids": region_ids,
                        "decisions_by_id": (
                            {
                                "r001": {
                                    "decision": "normalize",
                                    "final_translation": "统一后的问题1",
                                    "reason": "保持术语一致。",
                                    "confidence": "high",
                                    "child_note": "",
                                }
                            }
                            if page_number == 10
                            else {}
                        ),
                        "human_review": [],
                        "summary": "acceptable",
                    }
                return book.TurnResult(
                    thread_id=kwargs["thread_id"],
                    turn_id="turn-1",
                    response={
                        "batch_id": batch_id,
                        "page_results": page_results,
                        "consistency_summary": "Use 直角 consistently.",
                    },
                    usage={
                        "available": True,
                        "input_tokens": 100,
                        "cached_input_tokens": 0,
                        "cache_write_input_tokens": 0,
                        "non_cached_input_tokens": 100,
                        "output_tokens": 10,
                        "reasoning_output_tokens": 2,
                        "total_tokens": 110,
                    },
                    elapsed_seconds=0.1,
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            inputs = [
                self.make_page_inputs(root, 10, 1),
                self.make_page_inputs(root, 11, 1),
            ]
            settings = book.BookSettings(image_mode="never")
            task = {"task_id": "chapter-01", "title": "Shapes"}
            descriptor = book.make_batch_descriptor(task, inputs, settings)
            results, batch_record, summary = book.review_page_batch(
                FakeServer(),
                thread_id="thread-1",
                task=task,
                descriptor=descriptor,
                settings=settings,
                bootstrap_context=True,
                previous_consistency_summary="",
            )
        self.assertEqual([result["page"] for result in results], [10, 11])
        self.assertEqual(results[0]["codex"]["usage_scope"], "shared_batch")
        self.assertEqual(results[0]["decisions"][0]["id"], "r001")
        self.assertEqual(results[0]["decisions"][0]["current_translation"], "问题1")
        self.assertEqual(batch_record["pages"], [10, 11])
        self.assertEqual(summary, "Use 直角 consistently.")

    def test_execute_review_locks_batches_and_never_crosses_tasks(self) -> None:
        class FakeServer:
            def __init__(self):
                self.thread_number = 0

            def start_thread(self, **_kwargs):
                self.thread_number += 1
                return f"thread-{self.thread_number}"

            def resume_thread(self, thread_id):
                return thread_id

            def run_turn(self, **kwargs):
                schema = kwargs["output_schema"]
                batch_id = schema["properties"]["batch_id"]["const"]
                page_results = {}
                for key, page_schema in schema["properties"]["page_results"][
                    "properties"
                ].items():
                    page_results[key] = {
                        "page": page_schema["properties"]["page"]["const"],
                        "reviewed_region_ids": page_schema["properties"][
                            "reviewed_region_ids"
                        ]["items"]["enum"],
                        "decisions_by_id": {},
                        "human_review": [],
                        "summary": "acceptable",
                    }
                return book.TurnResult(
                    thread_id=kwargs["thread_id"],
                    turn_id=f"turn-{batch_id}",
                    response={
                        "batch_id": batch_id,
                        "page_results": page_results,
                        "consistency_summary": "stable terms",
                    },
                    usage={
                        "available": True,
                        "input_tokens": 100,
                        "cached_input_tokens": 0,
                        "cache_write_input_tokens": 0,
                        "non_cached_input_tokens": 100,
                        "output_tokens": 10,
                        "reasoning_output_tokens": 2,
                        "total_tokens": 110,
                    },
                    elapsed_seconds=0.1,
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            pages_dir = root / "pages"
            pages_dir.mkdir()
            for page_number in range(1, 7):
                self.make_page_inputs(pages_dir, page_number, 1)
            plan = {
                "tasks": [
                    {
                        "task_id": "chapter-01",
                        "title": "One",
                        "pdf_start_page": 1,
                        "pdf_end_page": 4,
                    },
                    {
                        "task_id": "chapter-02",
                        "title": "Two",
                        "pdf_start_page": 5,
                        "pdf_end_page": 6,
                    },
                ]
            }
            plan_path = root / "book-review-plan.json"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            output_dir = root / "output"
            settings = book.BookSettings(
                image_mode="never",
                batch_target_pages=2,
                batch_target_regions=100,
                batch_target_tokens=100000,
                thread_target_batches=1,
                thread_max_batches=1,
            )
            summary = book.execute_review(
                FakeServer(),
                plan=plan,
                plan_path=plan_path,
                pages_dir=pages_dir,
                output_dir=output_dir,
                settings=settings,
                workspace=root,
                force=False,
            )
            batch_plan = json.loads(
                (output_dir / "book-review-batch-plan.json").read_text(
                    encoding="utf-8"
                )
            )
            initial_batch_pages = [
                batch["pages"]
                for task in batch_plan["tasks"]
                for segment in task["segments"]
                for batch in segment["batches"]
            ]
            batch_plan["schema_version"] = 1
            batch_plan["settings"].pop("batch_response_schema_version")
            (output_dir / "book-review-batch-plan.json").write_text(
                json.dumps(batch_plan), encoding="utf-8"
            )
            migrated_summary = book.execute_review(
                FakeServer(),
                plan=plan,
                plan_path=plan_path,
                pages_dir=pages_dir,
                output_dir=output_dir,
                settings=settings,
                workspace=root,
                force=False,
            )
            batch_plan = json.loads(
                (output_dir / "book-review-batch-plan.json").read_text(
                    encoding="utf-8"
                )
            )
        self.assertEqual(summary["pages_reviewed"], 6)
        self.assertEqual(migrated_summary["pages_reviewed"], 6)
        self.assertEqual(summary["batches"], 3)
        self.assertEqual(batch_plan["schema_version"], book.BATCH_PLAN_SCHEMA_VERSION)
        self.assertEqual(
            [task["task_id"] for task in batch_plan["tasks"]],
            ["chapter-01", "chapter-02"],
        )
        self.assertEqual(
            initial_batch_pages,
            [[1, 2], [3, 4], [5, 6]],
        )
        self.assertEqual(
            [task["reused_pages"] for task in batch_plan["tasks"]],
            [[1, 2, 3, 4], [5, 6]],
        )

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
