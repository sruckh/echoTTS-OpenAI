import unittest
from app.text_chunking import split_text_into_chunks

class TestChunking(unittest.TestCase):
    def test_basic_split(self):
        text = "Hello world. This is a test."
        chunks = list(split_text_into_chunks(text, target_word_count=5))
        # "Hello world." (2 words) -> keeps adding -> "This is a test." (4 words) -> total 6 > 5?
        # Logic: 
        # 1. "Hello world." (2 words)
        # 2. Add "This is a test." (4 words). Total 6. 6 > 5.
        #    Yield "Hello world." (if overlap 0)
        #    Current buffer: "This is a test."
        # Yield remainder.
        
        # Let's see actual implementation behavior:
        # if len(current) + len(next) > target:
        #   yield current
        #   current = next
        
        # So: "Hello world." (2) + "This is a test." (4) = 6 > 5.
        # Yield "Hello world."
        # Buffer "This is a test."
        # End -> yield "This is a test."
        
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0], "Hello world.")
        self.assertEqual(chunks[1], "This is a test.")

    def test_overlap(self):
        text = "One two. Three four."
        # Target 2.
        # "One two." (2) + "Three four." (2) = 4 > 2.
        # Yield "One two."
        # Overlap 1 -> keep "two."
        # Buffer "two." + "Three four." -> "two. Three four."
        # End -> yield "two. Three four."
        
        chunks = list(split_text_into_chunks(text, target_word_count=2, overlap_words=1))
        # Actually logic says: if overlap > 0 and len >= overlap: keep last N.
        # current_chunk was ["One", "two."]
        # keep last 1: ["two."]
        # Extend with ["Three", "four."]
        # New chunk: "two. Three four."
        
        self.assertEqual(chunks[0], "One two.")
        self.assertEqual(chunks[1], "two. Three four.")

if __name__ == '__main__':
    unittest.main()
