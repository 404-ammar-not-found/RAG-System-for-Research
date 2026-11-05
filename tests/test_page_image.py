import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock


def _toy_pdf(path: Path) -> None:
    """Two pages; the 'Figure 1:' caption sits on page 1, not page 0."""
    import pymupdf

    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "Introduction prose")
    doc.new_page().insert_text((72, 72), "Figure 1: results over time")
    doc.save(path)
    doc.close()


class TestPageImage(unittest.TestCase):
    def test_traversal_reduces_to_a_basename(self):
        from src.api.web_api import _safe_pdf_name

        self.assertEqual(_safe_pdf_name("../../etc/passwd"), "passwd.pdf")
        self.assertEqual(_safe_pdf_name("paper.pdf"), "paper.pdf")

    def test_renders_png_and_finds_the_caption_page(self):
        from src.api import web_api

        with TemporaryDirectory() as td:
            _toy_pdf(Path(td) / "toy.pdf")
            with mock.patch.object(web_api, "DATA_DIR", Path(td)):
                web_api._render_page.cache_clear()
                page0 = web_api._render_page("toy.pdf", 0, "")
                with_caption = web_api._render_page("toy.pdf", 0, "Figure 1:")
                self.assertTrue(page0.startswith(b"\x89PNG"))
                self.assertTrue(with_caption.startswith(b"\x89PNG"))
                # The caption search must move the render off the section's
                # first page to the caption's page.
                self.assertNotEqual(page0, with_caption)

    def test_out_of_range_page_clamps(self):
        from src.api import web_api

        with TemporaryDirectory() as td:
            _toy_pdf(Path(td) / "toy.pdf")
            with mock.patch.object(web_api, "DATA_DIR", Path(td)):
                web_api._render_page.cache_clear()
                self.assertTrue(
                    web_api._render_page("toy.pdf", 99, "").startswith(b"\x89PNG")
                )
                self.assertTrue(
                    web_api._render_page("toy.pdf", -3, "").startswith(b"\x89PNG")
                )


if __name__ == "__main__":
    unittest.main()
