"""Encode a text dataset into a flat uint16 token-ID file."""

from __future__ import annotations

import argparse
import codecs
import os
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from tqdm import tqdm

from tokenizer_experiments import get_tokenizer_from_vocab_merges_path


DEFAULT_VOCAB_PATH = Path(
    "artifacts/tinystories_bpe/train_bpe_v2/vocab.json"
)
DEFAULT_MERGES_PATH = Path(
    "artifacts/tinystories_bpe/train_bpe_v2/merges.txt"
)
DEFAULT_INPUT_PATH = Path("data/TinyStoriesV2-GPT4-valid.txt")
DEFAULT_OUTPUT_PATH = Path("data/tinystories_valid.bin")
SPECIAL_TOKEN = "<|endoftext|>"


def iter_documents(
    input_path: Path,
    special_token: str,
    chunk_size_bytes: int,
) -> Iterator[str]:
    """Yield consecutive text segments while preserving every input character."""
    decoder = codecs.getincrementaldecoder("utf-8")()
    buffer = ""
    input_bytes = input_path.stat().st_size

    with input_path.open("rb") as input_file, tqdm(
        total=input_bytes,
        desc=f"Reading {input_path.name}",
        unit="B",
        unit_scale=True,
    ) as progress:
        while raw_chunk := input_file.read(chunk_size_bytes):
            progress.update(len(raw_chunk))
            buffer += decoder.decode(raw_chunk)

            pieces = buffer.split(special_token)
            for piece in pieces[:-1]:
                yield piece + special_token
            buffer = pieces[-1]

        buffer += decoder.decode(b"", final=True)

    # Preserve any text after the final special token, including whitespace.
    if buffer:
        yield buffer


def encode_dataset(
    tokenizer,
    input_path: Path,
    output_path: Path,
    chunk_size_bytes: int = 4 * 1024**2,
    overwrite: bool = False,
) -> None:
    """Stream-encode input_path and save the IDs as a flat uint16 array."""
    if not input_path.is_file():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output paths must be different")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Pass --overwrite to replace it."
        )
    if len(tokenizer.vocab) > np.iinfo(np.uint16).max + 1:
        raise ValueError("The tokenizer vocabulary does not fit in uint16")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_documents = 0
    total_tokens = 0
    max_token_id = -1
    started = time.perf_counter()

    with output_path.open("wb") as output_file:
        for text_segment in iter_documents(
            input_path,
            SPECIAL_TOKEN,
            chunk_size_bytes,
        ):
            token_ids = tokenizer.encode(text_segment)

            if token_ids:
                segment_max_id = max(token_ids)
                if segment_max_id >= len(tokenizer.vocab):
                    raise ValueError(
                        f"Token ID {segment_max_id} is outside a vocabulary of "
                        f"size {len(tokenizer.vocab)}"
                    )
                max_token_id = max(max_token_id, segment_max_id)

            np.asarray(token_ids, dtype=np.uint16).tofile(output_file)
            total_documents += text_segment.count(SPECIAL_TOKEN)
            total_tokens += len(token_ids)

    elapsed_seconds = time.perf_counter() - started
    input_bytes = input_path.stat().st_size
    output_bytes = output_path.stat().st_size
    expected_output_bytes = total_tokens * np.dtype(np.uint16).itemsize

    if output_bytes != expected_output_bytes:
        raise RuntimeError(
            f"Output size mismatch: expected {expected_output_bytes}, got {output_bytes}"
        )

    compression_ratio = input_bytes / total_tokens if total_tokens else 0.0
    throughput_mib = input_bytes / elapsed_seconds / 1024**2

    print(f"Documents: {total_documents:,}")
    print(f"Input bytes: {input_bytes:,}")
    print(f"Token count: {total_tokens:,}")
    print(f"Compression ratio: {compression_ratio:.4f} bytes/token")
    print(f"Maximum token ID: {max_token_id}")
    print(f"Elapsed time: {elapsed_seconds:.2f} seconds")
    print(f"Throughput: {throughput_mib:.2f} MiB/s")
    print(f"Output size: {output_bytes / 1024**2:.2f} MiB")
    print(f"Saved to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode a text dataset into a flat NumPy uint16 token file."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB_PATH)
    parser.add_argument("--merges", type=Path, default=DEFAULT_MERGES_PATH)
    parser.add_argument(
        "--chunk-size-mib",
        type=int,
        default=4,
        help="input read size in MiB (default: %(default)s)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace the output file if it already exists",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.chunk_size_mib <= 0:
        raise ValueError("--chunk-size-mib must be positive")

    tokenizer = get_tokenizer_from_vocab_merges_path(
        args.vocab,
        args.merges,
        [SPECIAL_TOKEN],
    )
    encode_dataset(
        tokenizer,
        args.input,
        args.output,
        chunk_size_bytes=args.chunk_size_mib * 1024**2,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
