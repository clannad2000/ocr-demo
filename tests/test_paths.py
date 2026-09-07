import pathlib
import tempfile
import unittest

from pipeline.config import ConfigError, load_config
from pipeline.paths import BookPaths, book_slug, discover_pdfs


class PipelinePathTests(unittest.TestCase):
    def test_book_slug_matches_confirmed_layout(self) -> None:
        pdf = pathlib.Path("beast academy math guide 3A.pdf")
        self.assertEqual(book_slug(pdf), "beast_academy_math_guide_3A")
        paths = BookPaths.for_pdf(pdf.resolve())
        self.assertEqual(paths.root.name, "beast_academy_math_guide_3A")
        self.assertEqual(paths.pages.relative_to(paths.root).as_posix(), "01-ocr/pages")
        self.assertEqual(paths.review.name, "02-codex-review")
        self.assertEqual(paths.final.name, "03-final")
        self.assertEqual(paths.erased.name, "04-erased")
        self.assertEqual(paths.pdf_output.name, "05-pdf")

    def test_pdf_directory_discovery_is_sorted_and_non_recursive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            (root / "B.pdf").write_bytes(b"pdf")
            (root / "a.PDF").write_bytes(b"pdf")
            nested = root / "nested"
            nested.mkdir()
            (nested / "ignored.pdf").write_bytes(b"pdf")
            discovered = discover_pdfs({"pdf": str(root)})
        self.assertEqual([path.name for path in discovered], ["a.PDF", "B.pdf"])

    def test_derived_output_paths_are_rejected_in_shared_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = pathlib.Path(temporary) / "pipeline.json"
            config.write_text(
                '{"pdf": "book.pdf", "output": "runs/manual"}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConfigError, "output"):
                load_config(config)


if __name__ == "__main__":
    unittest.main()
