import json
import pathlib
import tempfile
import types
import unittest
from unittest import mock

from pipeline import __main__ as cli
from pipeline.paths import BookPaths


class PipelineCliTests(unittest.TestCase):
    def test_directory_config_dispatches_each_pdf_to_its_own_book_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            books = root / "books"
            books.mkdir()
            (books / "Book B.pdf").write_bytes(b"pdf")
            (books / "Book A.pdf").write_bytes(b"pdf")
            config = root / "pipeline.json"
            config.write_text(json.dumps({"pdf": str(books)}), encoding="utf-8")
            seen: list[BookPaths] = []

            def fake_runner(book, _config_path, _args, _config):
                seen.append(book)
                return 0

            with mock.patch.dict(cli.RUNNERS, {"ocr": fake_runner}):
                result = cli.main(["--config", str(config), "ocr"])

        self.assertEqual(result, 0)
        self.assertEqual([book.slug for book in seen], ["Book_A", "Book_B"])
        self.assertTrue(all(book.root.parent.name == "book" for book in seen))

    def test_review_derives_plan_and_page_directories(self) -> None:
        book = BookPaths.for_pdf(pathlib.Path("unit test review.pdf").resolve())
        args = types.SimpleNamespace(force=False, dry_run=False)
        calls: list[list[str]] = []

        def fake_main(arguments):
            calls.append(arguments)
            return 0

        with mock.patch.object(cli.codex_review, "main", side_effect=fake_main):
            result = cli.run_review(
                book,
                pathlib.Path("config/pipeline.json").resolve(),
                args,
                {"toc_pages": "5-6"},
            )

        self.assertEqual(result, 0)
        self.assertEqual(calls[0][0], "plan")
        self.assertEqual(calls[1][0], "review")
        self.assertIn(str(book.pages), calls[0])
        self.assertIn(str(book.review), calls[1])


if __name__ == "__main__":
    unittest.main()
