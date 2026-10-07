import json
import os
import time
import random
from encode_decode import Tokenization
from train_tinystories_bpe import bytes_to_unicode

VOCAB_PATH = "/home/yzqz/lessons/cs336/assignment1-basics/artifacts/tinystories_bpe/train_bpe_v2/vocab.json"
MERGES_PATH = "/home/yzqz/lessons/cs336/assignment1-basics/artifacts/tinystories_bpe/train_bpe_v2/merges.txt"
VAILD_PATH = "/home/yzqz/lessons/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt"
SPECIAL_TOKEN = "<|endoftext|>"

def get_tokenizer_from_vocab_merges_path(
    vocab_path: str | os.PathLike[str],
    merges_path: str | os.PathLike[str],
    special_tokens: list[str] | None = None,
) -> Tokenization:
    gpt2_byte_decoder = {v: k for k, v in bytes_to_unicode().items()}

    with open(vocab_path) as vocab_f:
        tinystory_vocab = json.load(vocab_f)
    tinystory_merges = []
    with open(merges_path) as f:
        for line in f:
            cleaned_line = line.rstrip()
            if cleaned_line and len(cleaned_line.split(" ")) == 2:
                tinystory_merges.append(tuple(cleaned_line.split(" ")))

    vocab = {
        gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
        for gpt2_vocab_item, gpt2_vocab_index in tinystory_vocab.items()
    }

    if special_tokens:
        for special_token in special_tokens:
            byte_encoded_special_token = special_token.encode("utf-8")
            if byte_encoded_special_token not in set(vocab.values()):
                vocab[len(vocab)] = byte_encoded_special_token

    merges = [
        (
            bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
            bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
        )
        for merge_token_1, merge_token_2 in tinystory_merges
    ]

    return Tokenization(vocab, merges, special_tokens)

def main():
    tokenizer = get_tokenizer_from_vocab_merges_path(VOCAB_PATH,MERGES_PATH,[SPECIAL_TOKEN])

    with open(VAILD_PATH,encoding = "utf-8") as f:
        valid_text = f.read()

    docs = valid_text.split(SPECIAL_TOKEN)
    documents = []
    for doc in docs :
        if doc.strip():
            documents.append(doc)

    # 1. 取10篇文档
    sample_documents = random.sample(documents, 10)

    # 2. 原始字节数
    total_bytes = sum(
        len(doc.encode("utf-8"))
        for doc in sample_documents
    )

    # 3. 只对 tokenizer.encode() 计时
    start_time = time.perf_counter()

    encoded_documents = []
    for doc in sample_documents:
        ids = tokenizer.encode(doc)
        encoded_documents.append(ids)

    elapsed_seconds = time.perf_counter() - start_time

    # 4. token 总数
    total_tokens = sum(
        len(ids)
        for ids in encoded_documents
    )

    # 5. 压缩率：平均一个 token 表示多少字节
    compression_ratio = total_bytes / total_tokens

    # 6. 编码吞吐量
    throughput = total_bytes / elapsed_seconds
    throughput_mib = throughput / 1024**2

    # 7. 最大 token ID
    max_token_id = max(
        token_id
        for ids in encoded_documents
        for token_id in ids
    )

    assert max_token_id < 10_000

    print(f"文档数量: {len(sample_documents)}")
    print(f"原始字节数: {total_bytes:,}")
    print(f"token 数量: {total_tokens:,}")
    print(f"压缩率: {compression_ratio:.4f} bytes/token")
    print(f"编码耗时: {elapsed_seconds:.6f} 秒")
    print(f"吞吐量: {throughput_mib:.2f} MiB/s")
    print(f"最大 token ID: {max_token_id}")

if __name__ == "__main__":
    main()
