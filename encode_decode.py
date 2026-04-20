from typing import Iterable, Iterator

import regex as re
from collections import defaultdict
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenization:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []


        self.merges_priority_map = {pair: i for i, pair in enumerate(self.merges)}
        self.vocab_inv = {v: k for k, v in self.vocab.items()}

    def get_bpe_merges(self, word_bytes: bytes) -> list[bytes]:
        parts = [bytes([b]) for b in word_bytes]
        while len(parts) > 1:
            pairs = set()
            for i in range(len(parts) - 1):
                pair = (parts[i], parts[i+1])
                if pair in self.merges_priority_map:
                    pairs.add(pair)
            if not pairs:
                break
            best_pair = min(pairs, key=lambda pair: self.merges_priority_map[pair])

            new_parts = []
            i = 0
            while i < len(parts):
                if i < len(parts) - 1 and (parts[i],parts[i + 1]) == best_pair:
                    new_parts.append(parts[i] + parts[i + 1])
                    i += 2
                else:
                    new_parts.append(parts[i])
                    i += 1
            parts = new_parts
        return parts

    def encode(self, text: str) -> list[int]:
        
        sorted_special_tokens = []
        
        sorted_special_tokens = sorted(self.special_tokens, key=lambda x: (len(x), x), reverse=True) 
        special_token_pattern = '|'.join(map(re.escape, sorted_special_tokens))
        
        if self.special_tokens:
            chunks = re.split(f'({special_token_pattern})', text)
        else:
            chunks = [text] 
        
        final_ids = []
        for chunk in chunks:
            if not chunk:
                continue
            if chunk in self.special_tokens:
                final_ids.append(self.vocab_inv[chunk.encode("utf-8")])
            else:
                words = re.findall(PAT,chunk)
                for word in words:
                    merged_pieces = self.get_bpe_merges(word.encode('utf-8'))
                    for pieci in merged_pieces:
                        final_ids.append(self.vocab_inv[pieci])
        return final_ids
    
    def decode(self,ids: list[int]) -> str:
        bytes_list = [self.vocab[id] for id in ids]
        return b''.join(bytes_list).decode('utf-8', errors='ignore')
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)