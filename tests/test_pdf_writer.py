import json
import pathlib
import tempfile
import unittest

from pipeline import page_review
from pipeline import pdf_writer as writer


class PdfTranslationWriterTests(unittest.TestCase):
    def test_writer_uses_cleaned_image_without_erasure(self) -> None:
        import pymupdf
        from PIL import Image

        font = pathlib.Path("../assets/fonts/simhei.ttf")
        if not font.is_file():
            self.skipTest("Chinese test font is unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source_pdf = (root / "source.pdf").resolve()
            source = pymupdf.open()
            source.new_page(width=300, height=400)
            source.save(source_pdf)
            source.close()
            cleaned = root / "page-0001.cleaned.png"
            Image.new("RGB", (300, 400), (245, 245, 240)).save(cleaned)
            adjudication = root / "book-codex-adjudication.json"
            adjudication.write_text(
                json.dumps(
                    {
                        "source_pdf_sha256": page_review.sha256_file(source_pdf)
                    }
                ),
                encoding="utf-8",
            )
            snapshot_path = root / "chapter_translation_final.json"
            snapshot = {
                "schema_version": 1,
                "status": "locked_final_translation",
                "source_pdf": str(source_pdf),
                "pages": [1],
                "page_count": 1,
                "region_count": 1,
                "inputs": {
                    "adjudication_file": str(adjudication),
                    "adjudication_sha256": page_review.sha256_file(adjudication),
                },
                "regions": [
                    {
                        "page": 1,
                        "id": "r001",
                        "type": "dialogue",
                        "source_text": "Hello",
                        "final_translation": "你好！",
                        "translation_source": "codex_adjudication",
                        "decision": "replace",
                        "reason": "test",
                        "child_note": "",
                        "locked": True,
                    }
                ],
            }
            snapshot_path.write_text(
                json.dumps(snapshot, ensure_ascii=False), encoding="utf-8"
            )
            plan = {
                "schema_version": 1,
                "layout_item_ownership_conflict_count": 0,
                "layout_item_ownership_conflicts": [],
                "mappings": [
                    {
                        "page": 1,
                        "id": "r001",
                        "type": "dialogue",
                        "source_text": "Hello",
                        "final_translation": "你好！",
                        "translation_source": "codex_adjudication",
                        "placement_box": [100, 100, 400, 220],
                        "match_coverage": 1.0,
                        "confidence": "high",
                        "warnings": [],
                    }
                ],
            }
            output_pdf = root / "output.pdf"
            snapshot_regions = writer.validate_snapshot(
                snapshot, source_pdf=source_pdf, snapshot_path=snapshot_path
            )
            writer.validate_plan(plan, snapshot_regions)
            exported = writer.write_pdf(
                source_pdf=source_pdf,
                snapshot=snapshot,
                plan=plan,
                cleaned_pages={1: cleaned},
                font_file=font,
                output_pdf=output_pdf,
                rules=writer.merge_rules(None),
                overrides={},
            )
            checked = writer.program_check(
                source_pdf=source_pdf,
                output_pdf=output_pdf,
                snapshot=snapshot,
                plan=plan,
                export=exported,
            )
            reopened = pymupdf.open(output_pdf)
            try:
                self.assertEqual(reopened.page_count, 1)
            finally:
                reopened.close()
        self.assertEqual(exported["status_counts"], {"written": 1})
        self.assertFalse(exported["cleaned_pages"]["writer_performed_erasure"])
        self.assertEqual(checked["status"], "program_checked")


if __name__ == "__main__":
    unittest.main()
