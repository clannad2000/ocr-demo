import json
import pathlib
import tempfile
import types
import unittest
from unittest import mock

from pipeline import ocr


class OcrPipelineTests(unittest.TestCase):
    def test_removed_legacy_modes_are_not_public_cli_options(self):
        parser = ocr.build_parser()
        args = parser.parse_args([])
        self.assertEqual(args.translation_verify, "never")
        self.assertNotIn("--chapter-review", parser._option_string_actions)

    def test_config_rejects_removed_translation_semantic_review_switch(self):
        with tempfile.TemporaryDirectory() as temporary:
            config_path = pathlib.Path(temporary) / "pipeline.json"
            config_path.write_text(
                '{"translation_verify": "always"}', encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "translation_verify"):
                ocr.load_config_defaults(config_path)

    def test_page_writer_creates_json_without_markdown(self):
        with tempfile.TemporaryDirectory() as temporary:
            pages = pathlib.Path(temporary) / "pages"
            json_path = pages / "page-0053.json"
            record = {"page": 53, "mode": "study"}

            ocr.write_page_json(json_path, record)

            self.assertEqual(
                json.loads(json_path.read_text(encoding="utf-8")), record
            )
            self.assertFalse((pages / "page-0053.md").exists())

    def test_aggregate_markdown_and_html_are_still_writable(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = pathlib.Path(temporary)
            summary_md = output / "summary.md"
            study_html = output / "study.html"

            ocr.atomic_write_text(summary_md, "summary")
            ocr.atomic_write_text(study_html, "study")

            self.assertEqual(summary_md.read_text(encoding="utf-8"), "summary")
            self.assertEqual(study_html.read_text(encoding="utf-8"), "study")

    def test_resume_loads_page_json_without_page_markdown(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = pathlib.Path(temporary)
            pages = output / "pages"
            pages.mkdir()
            record = {"page": 53, "mode": "study"}
            (pages / "page-0053.json").write_text(
                json.dumps(record), encoding="utf-8"
            )
            args = types.SimpleNamespace(
                output=output, resume=True, redo_pages=set()
            )

            result = ocr.process_page(args, 53)

            self.assertEqual(result, record)

    def test_redo_page_does_not_reuse_existing_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = pathlib.Path(temporary)
            pages = output / "pages"
            pages.mkdir()
            (pages / "page-0053.json").write_text("{}", encoding="utf-8")
            args = types.SimpleNamespace(
                output=output,
                resume=True,
                redo_pages={53},
                pdf=pathlib.Path("book.pdf"),
                long_edge=2048,
                pdftoppm_command="pdftoppm",
            )

            with mock.patch.object(
                ocr, "render_page", side_effect=RuntimeError("rerun")
            ) as render:
                with self.assertRaisesRegex(RuntimeError, "rerun"):
                    ocr.process_page(args, 53)

            render.assert_called_once()


if __name__ == "__main__":
    unittest.main()
