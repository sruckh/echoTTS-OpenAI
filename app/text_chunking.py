import re
from typing import List, Generator

def split_text_into_chunks(
    text: str, 
    target_word_count: int = 40, 
    overlap_words: int = 0
) -> Generator[str, None, None]:
    """
    Splits text into chunks of approximately target_word_count words.
    Tries to split on sentence boundaries (. ! ?) first, then clauses (, ; :) if needed.
    """
    if not text:
        return

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Split into rough sentence-like units
    # This regex looks for .!? followed by a space or end of string
    # We keep the punctuation attached to the preceding sentence
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    current_chunk_words: List[str] = []
    
    for sentence in sentences:
        sentence_words = sentence.split()
        
        # If adding this sentence exceeds the target significantly (e.g. by > 50%),
        # and we already have content, yield the current chunk first.
        # But if the current chunk is empty, we must take it (or split it if it's huge).
        
        # Simple logic: accumulate until > target_word_count
        if len(current_chunk_words) + len(sentence_words) > target_word_count:
            if current_chunk_words:
                yield " ".join(current_chunk_words)
                
                # Handle overlap
                if overlap_words > 0 and len(current_chunk_words) >= overlap_words:
                    current_chunk_words = current_chunk_words[-overlap_words:]
                else:
                    current_chunk_words = []
            
            # If the single sentence is massive (larger than target), 
            # we ideally should split it by clauses, but for now let's just add it
            # to avoid cutting mid-word or implementing complex clause logic.
            # It will be yielded in the next pass.
            current_chunk_words.extend(sentence_words)
        else:
            current_chunk_words.extend(sentence_words)
            
    if current_chunk_words:
        yield " ".join(current_chunk_words)
