import unittest
from pathlib import Path

from src.pipeline.parsing import EPISODE_MAX_CHARS, parse_paper, section_to_chunks

REPO_ROOT = Path(__file__).resolve().parents[1]
PDF = REPO_ROOT / "data" / "1706.03762v7.pdf"


@unittest.skipUnless(PDF.exists(), "sample PDF not present")
class TestParsePaper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sections = parse_paper(PDF)

    def test_finds_multiple_named_sections(self):
        self.assertGreater(len(self.sections), 2)
        self.assertTrue(all(s.section.strip() for s in self.sections))

    def test_title_is_not_the_arxiv_stamp(self):
        title = self.sections[0].paper_title
        self.assertNotIn("arXiv:", title)
        self.assertIn("Attention", title)

    def test_no_section_exceeds_episode_cap(self):
        for s in self.sections:
            self.assertLessEqual(len(s.text), EPISODE_MAX_CHARS, s.section)

    def test_episode_name_joins_stem_and_section(self):
        s = self.sections[0]
        self.assertEqual(s.episode_name, f"{PDF.stem}::{s.section}")

    def test_chunks_carry_context_header_not_file_path(self):
        section = next(s for s in self.sections if len(s.text) > 500)
        chunks = section_to_chunks(section, chunk_size=1200, chunk_overlap=150)
        self.assertTrue(chunks)
        for c in chunks:
            # Semantic header, not the old "[data/foo.pdf#chunk3]" prefix.
            self.assertTrue(c["text"].startswith(section.paper_title))
            self.assertNotIn(".pdf#chunk", c["text"])
            self.assertEqual(c["metadata"]["section"], section.section)


class TestSplitLong(unittest.TestCase):
    def test_hard_wraps_a_single_giant_paragraph(self):
        from src.pipeline.parsing import _split_long

        pieces = _split_long("x" * 9500, 4000)
        self.assertTrue(all(len(p) <= 4000 for p in pieces))
        self.assertEqual(sum(len(p) for p in pieces), 9500)


if __name__ == "__main__":
    unittest.main()
