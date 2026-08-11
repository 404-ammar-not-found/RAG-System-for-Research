import unittest

from src.pipeline.scholarly import parse_arxiv_id


class ParseArxivId(unittest.TestCase):
    def test_accepts_the_shapes_people_paste(self):
        for text, expected in [
            ("https://arxiv.org/abs/1706.03762", "1706.03762"),
            ("http://arxiv.org/pdf/1706.03762v5", "1706.03762v5"),
            ("arxiv.org/pdf/1706.03762.pdf", "1706.03762"),
            ("https://arxiv.org/html/2311.05232v2", "2311.05232v2"),
            ("  arXiv:1706.03762  ", "1706.03762"),
            ("1706.03762", "1706.03762"),
            ("https://arxiv.org/abs/cs/0501001", "cs/0501001"),
        ]:
            self.assertEqual(parse_arxiv_id(text), expected, text)

    def test_rejects_non_papers(self):
        for text in ["", "how does attention work", "https://example.com/paper"]:
            self.assertIsNone(parse_arxiv_id(text), text)


if __name__ == "__main__":
    unittest.main()
