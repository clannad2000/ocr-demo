import json
import pathlib
import tempfile
import unittest

import codex_book_finalize as finalize
import codex_page_review as page_review


class CodexBookFinalizeTests(unittest.TestCase):
    def make_fixture(self, root: pathlib.Path, *, human_review: bool = False):
        pdf = (root / "book.pdf").resolve()
        pdf.write_bytes(b"pdf-placeholder")
        pages_dir = root / "pages"
        reviews_root = root / "book-review"
        reviews_dir = reviews_root / "chapters"
        pages_dir.mkdir()
        plan_path = reviews_root / "book-review-plan.json"
        plan_path.parent.mkdir()
        plan = {
            "schema_version": 1,
            "kind": "codex_book_review_plan",
            "pdf": {
                "path": str(pdf),
                "sha256": page_review.sha256_file(pdf),
                "page_count": 1,
            },
            "tasks": [
                {
                    "task_id": "preliminary",
                    "title": "Preliminary",
                    "pdf_start_page": 1,
                    "pdf_end_page": 1,
                }
            ],
        }
        plan_path.write_text(json.dumps(plan), encoding="utf-8")
        page_json = pages_dir / "page-0001.json"
        page_record = {
            "page": 1,
            "mode": "study",
            "layout": {
                "content": "<|ref|>Hello<|/ref|><|det|>[[100,100,300,180]]<|/det|>"
            },
            "study": {
                "regions": [
                    {
                        "id": "r001",
                        "type": "dialogue",
                        "source_text": "Hello",
                        "translation": "你好",
                    }
                ]
            },
        }
        page_json.write_text(json.dumps(page_record, ensure_ascii=False), encoding="utf-8")
        review_dir = reviews_dir / "preliminary"
        review_dir.mkdir(parents=True)
        review_path = review_dir / "page-0001-codex-review.json"
        review = {
            "schema_version": 5,
            "review_scope": "book_chapter_page",
            "page": 1,
            "inputs": {
                "files": {page_json.name: page_review.sha256_file(page_json)},
                "region_ids": ["r001"],
            },
            "codex": {"thread_id": "thread-preliminary"},
            "decisions": [
                {
                    "page": 1,
                    "id": "r001",
                    "decision": "replace",
                    "current_translation": "你好",
                    "final_translation": "你好！",
                    "reason": "保留问候语气。",
                    "confidence": "high",
                    "child_note": "",
                }
            ],
            "human_review": (
                [
                    {
                        "page": 1,
                        "id": "r001",
                        "reason": "ambiguous",
                        "evidence_needed": "image",
                        "evidence_type": "image",
                    }
                ]
                if human_review
                else []
            ),
            "book_task": {
                "task_id": "preliminary",
                "title": "Preliminary",
                "plan_sha256": page_review.sha256_file(plan_path),
            },
        }
        review_path.write_text(json.dumps(review, ensure_ascii=False), encoding="utf-8")
        return pdf, pages_dir, reviews_dir, plan_path, plan

    def test_merge_reviews_applies_only_sparse_decisions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            pdf, pages_dir, reviews_dir, plan_path, plan = self.make_fixture(root)
            records, aggregate = finalize.merge_book_reviews(
                plan=plan,
                plan_path=plan_path,
                pages_dir=pages_dir,
                reviews_dir=reviews_dir,
                source_pdf=pdf,
            )
        region = records[0]["study"]["regions"][0]
        self.assertEqual(region["translation"], "你好！")
        self.assertEqual(region["codex_adjudication"]["decision"], "replace")
        self.assertEqual(aggregate["decision_count"], 1)
        self.assertEqual(
            aggregate["persistent_threads"], {"preliminary": ["thread-preliminary"]}
        )

    def test_unresolved_human_review_blocks_finalization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            pdf, pages_dir, reviews_dir, plan_path, plan = self.make_fixture(
                root, human_review=True
            )
            with self.assertRaisesRegex(finalize.FinalizeError, "blocks finalization"):
                finalize.merge_book_reviews(
                    plan=plan,
                    plan_path=plan_path,
                    pages_dir=pages_dir,
                    reviews_dir=reviews_dir,
                    source_pdf=pdf,
                )


if __name__ == "__main__":
    unittest.main()
