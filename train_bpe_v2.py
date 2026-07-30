import os
from collections import Counter
from contextlib import nullcontext

import regex as re
from tqdm import tqdm

class BpeTokenizer:
    def __init__(self):
        """
            初始化
            vocab merges 为最终需要的结果
            vocab_size 为当前vocab的大小，初始为0
            PAT为预tokenizer的正则表达式，后续会在add_special_tokens中更新为包含special token的正则表达式
        """
        self.vocab: dict[int, bytes] = {}
        self.pair_to_pretoken_ids: dict[tuple[bytes,bytes],set[int]] = dict()
        self.pair_freqs : Counter[tuple[bytes,bytes]] = Counter()
        self.vocab_size = 0
        self.id_to_words: dict[int, tuple[tuple[bytes, ...], int]] = dict()
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
            

    def count_pre_tokens(self, input_file : str, show_progress: bool = False) -> Counter:
        word_freqs = Counter()

        mini_chunk = 4096 * 4096

        progress = (
            tqdm(
                total=os.path.getsize(input_file),
                desc="Pre-tokenizing",
                unit="B",
                unit_scale=True,
            )
            if show_progress
            else nullcontext()
        )

        with open(input_file, "rb") as f, progress as progress_bar:

            while 1 :
                chunk = f.read(mini_chunk)

                if chunk == b"":
                    break

                if progress_bar is not None:
                    progress_bar.update(len(chunk))

                text = chunk.decode("utf-8", errors="ignore")
                splits = self.special_token_pattern.split(text) if self.special_token_pattern else [text]

                for split in splits:
                    words = self.PAT.findall(split) 
                    for word in words:
                        word_freqs[word] += 1    

            final_freqs = Counter()
            for word, freq in word_freqs.items():
                word_bytes = tuple(bytes([b]) for b in word.encode("utf-8"))
                final_freqs[word_bytes] = freq
        return final_freqs

    def merge_tokens(self, token1: bytes, token2: bytes, new_token_bytes: bytes) -> None:
        pair = (token1, token2)
        affected_ids = self.pair_to_pretoken_ids.pop(pair, set())

        for pretoken_id in affected_ids:
            words, freq = self.id_to_words[pretoken_id]
            old_pair_counts = Counter(zip(words, words[1:]))

            new_words_list = []
            index = 0
            while index < len(words):
                if (
                    index + 1 < len(words)
                    and words[index] == token1
                    and words[index + 1] == token2
                ):
                    new_words_list.append(new_token_bytes)
                    index += 2
                else:
                    new_words_list.append(words[index])
                    index += 1

            new_words = tuple(new_words_list)
            new_pair_counts = Counter(zip(new_words, new_words[1:]))

            changed_pairs = old_pair_counts.keys() | new_pair_counts.keys()
            for changed_pair in changed_pairs:
                count_delta = (
                    new_pair_counts[changed_pair] - old_pair_counts[changed_pair]
                ) * freq

                if count_delta:
                    self.pair_freqs[changed_pair] += count_delta
                    if self.pair_freqs[changed_pair] <= 0:
                        self.pair_freqs.pop(changed_pair, None)

                if new_pair_counts[changed_pair] > 0:
                    self.pair_to_pretoken_ids.setdefault(changed_pair, set()).add(
                        pretoken_id
                    )
                else:
                    pair_ids = self.pair_to_pretoken_ids.get(changed_pair)
                    if pair_ids is not None:
                        pair_ids.discard(pretoken_id)
                        if not pair_ids:
                            self.pair_to_pretoken_ids.pop(changed_pair, None)

            self.id_to_words[pretoken_id] = (new_words, freq)
                

        

    def train(
        self,
        input_file: str,
        target_vocab_size: int,
        show_progress: bool = False,
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        """
            训练
        """
        word_freqs = self.count_pre_tokens(input_file, show_progress=show_progress)
        cnt = 0

        for words,freq in word_freqs.items():
                for word in range(len(words) - 1):
                    self.pair_freqs[(words[word], words[word + 1])] += freq
                    if (words[word], words[word + 1]) not in self.pair_to_pretoken_ids :
                         self.pair_to_pretoken_ids[(words[word], words[word + 1])] = set()
                    self.pair_to_pretoken_ids[(words[word], words[word + 1])].add(cnt)
                self.id_to_words[cnt] = (words,freq)    
                cnt = cnt + 1
    

        progress = (
            tqdm(
                total=max(0, target_vocab_size - self.vocab_size),
                desc="Learning merges (v2)",
                unit="merge",
            )
            if show_progress
            else nullcontext()
        )

        with progress as progress_bar:
            while self.vocab_size < target_vocab_size:

                if not self.pair_freqs:
                    break

                best_pair = max(self.pair_freqs, key=self.pair_freqs.get)

                max_freq = self.pair_freqs[best_pair]
                candidate = []
                for pair, freq in self.pair_freqs.items():
                    if freq == max_freq:
                        candidate.append(pair)
                best_pair = max(candidate)

                part1_bytes = best_pair[0]
                part2_bytes = best_pair[1]

                new_token_bytes = part1_bytes + part2_bytes

                self.vocab[self.vocab_size] = new_token_bytes
                self.merges.append((part1_bytes, part2_bytes))
                self.vocab_size += 1

                self.merge_tokens(best_pair[0], best_pair[1], new_token_bytes)
                if progress_bar is not None:
                    progress_bar.update(1)

                # self.vocab_inv: dict[bytes, int] = {v: k for k, v in self.vocab.items()}

        return self.vocab, self.merges
    
    
