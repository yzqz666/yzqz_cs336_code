import regex as re
from collections import Counter

class BpeTokenizer:
    def __init__(self):
        """
            初始化
            vocab merges 为最终需要的结果
            vocab_size 为当前vocab的大小，初始为0
            PAT为预tokenizer的正则表达式，后续会在add_special_tokens中更新为包含special token的正则表达式
        """
        self.vocab: dict[int, bytes] = {}
        self.vocab_size = 0
        for i in range(256):
            self.vocab[self.vocab_size] = bytes([i])
            self.vocab_size += 1
        self.merges: list[tuple[bytes, bytes]] = []

        self.vocab_inv: dict[bytes, int] = {} 
        self.special_token_pattern: re.Pattern | None = None
       
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.PAT = re.compile(self.PAT)

    def add_special_tokens(self, special_tokens : list[str]):
        """
            添加special token到vacab中，并且在tokenizer中记录special token的正则表达式
        """
        special_tokens.sort(key=lambda x: (len(x), x), reverse=True)

        escaped_strs =[]

        for special_token in special_tokens:
            self.vocab[self.vocab_size] = special_token.encode("utf-8")
            self.vocab_size += 1
            escaped_strs.append(re.escape(special_token))
        
        if escaped_strs:
            self.special_token_pattern = re.compile("|".join(escaped_strs))
            

    def count_pre_tokens(self, input_file : str) -> Counter:
        word_freqs = Counter()

        mini_chunk = 4096 * 4096

        with open(input_file, "rb") as f:

            while 1 :
                chunk = f.read(mini_chunk)

                if chunk == b"":
                    break

                splits = self.special_token_pattern.split(chunk.decode("utf-8",errors="ignore"))

                for split in splits:
                    words = self.PAT.findall(split) 
                    for word in words:
                        word_freqs[word] += 1    

            final_freqs = Counter()
            for word, freq in word_freqs.items():
                word_bytes = tuple(bytes([b]) for b in word.encode("utf-8"))
                final_freqs[word_bytes] = freq
        return final_freqs

    def merge_tokens(self, word_freqs: Counter, token1: bytes, token2: bytes, new_token_bytes: bytes) -> Counter:
        new_word_freqs = Counter()
        
        for words, freq in word_freqs.items():
            # 修复 3: 使用 += 而不是 =
            if len(words) < 2:
                new_word_freqs[words] += freq
                continue
            if token1 not in words:
                new_word_freqs[words] += freq
                continue

            new_word = []
            i = 0
            while i < len(words):
                if i < len(words) - 1 and words[i] == token1 and words[i+1] == token2:
                    new_word.append(new_token_bytes) # 直接填入合并后的 bytes
                    i += 2
                else:
                    new_word.append(words[i])
                    i += 1
            new_word_freqs[tuple(new_word)] += freq

        return new_word_freqs

    def train(self, input_file: str, target_vocab_size: int) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        """
            训练
        """
        word_freqs = self.count_pre_tokens(input_file)

        while self.vocab_size < target_vocab_size:
            pair_freqs = Counter()

            for words,freq in word_freqs.items():
                for word in range(len(words) - 1):
                    pair_freqs[(words[word], words[word + 1])] += freq

            if not pair_freqs:
                break

            best_pair = max(pair_freqs, key=pair_freqs.get)

            max_freq = pair_freqs[best_pair]
            candidate = []
            for pair, freq in pair_freqs.items():
                if freq == max_freq:
                    candidate.append(pair)
            best_pair = max(candidate)

            part1_bytes = best_pair[0]
            part2_bytes = best_pair[1]

            new_token_bytes = part1_bytes + part2_bytes
            
            self.vocab[self.vocab_size] = new_token_bytes
            self.merges.append((part1_bytes, part2_bytes))
            self.vocab_size += 1

            word_freqs = self.merge_tokens(word_freqs, best_pair[0], best_pair[1], new_token_bytes)

            self.vocab_inv: dict[bytes, int] = {v: k for k, v in self.vocab.items()} 

        return self.vocab, self.merges
    
    