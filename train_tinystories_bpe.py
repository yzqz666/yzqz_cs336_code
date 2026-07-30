"""Train both BPE implementations on TinyStories and save their artifacts."""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import threading
import time
from pathlib import Path

import psutil


IMPLEMENTATIONS = ("train_bpe", "train_bpe_v2")


class MemoryMonitor:
    """Sample this process's resident memory in a background thread."""

    def __init__(self, interval_seconds: float = 0.1) -> None:
        self.interval_seconds = interval_seconds
        self.process = psutil.Process()
        self.baseline_bytes = 0
        self.peak_bytes = 0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        rss = self.process.memory_info().rss
        self.peak_bytes = max(self.peak_bytes, rss)

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            self._sample()

    def start(self) -> None:
        self._sample()
        self.baseline_bytes = self.peak_bytes
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._sample()
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()

    @property
    def increase_bytes(self) -> int:
        return max(0, self.peak_bytes - self.baseline_bytes)


def mib(byte_count: int) -> float:
    return byte_count / 1024**2


def bytes_to_unicode() -> dict[int, str]:
    """Return GPT-2's reversible, printable byte-to-Unicode mapping."""
    byte_values = list(range(ord("!"), ord("~") + 1))
    byte_values += list(range(ord("¡"), ord("¬") + 1))
    byte_values += list(range(ord("®"), ord("ÿ") + 1))
    codepoints = byte_values[:]
    offset = 0
    for byte_value in range(256):
        if byte_value not in byte_values:
            byte_values.append(byte_value)
            codepoints.append(256 + offset)
            offset += 1
    return dict(zip(byte_values, map(chr, codepoints)))


def printable_token(token: bytes, encoder: dict[int, str]) -> str:
    return "".join(encoder[byte] for byte in token)


def save_artifacts(
    output_dir: Path,
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    metadata: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    encoder = bytes_to_unicode()

    # This is the same reversible printable representation used by GPT-2 files.
    serialized_vocab = {
        printable_token(token, encoder): token_id for token_id, token in vocab.items()
    }
    with (output_dir / "vocab.json").open("w", encoding="utf-8") as file:
        json.dump(serialized_vocab, file, ensure_ascii=False, indent=2)
        file.write("\n")

    with (output_dir / "merges.txt").open("w", encoding="utf-8") as file:
        for left, right in merges:
            file.write(f"{printable_token(left, encoder)} {printable_token(right, encoder)}\n")

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
        file.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train train_bpe.py and train_bpe_v2.py on TinyStories."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/TinyStoriesV2-GPT4-train.txt"),
        help="training corpus (default: %(default)s)",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=10_000,
        help="target vocabulary size including special tokens (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/tinystories_bpe"),
        help="artifact root directory (default: %(default)s)",
    )
    parser.add_argument(
        "--implementation",
        choices=("both", *IMPLEMENTATIONS),
        default="both",
        help="implementation to train (default: both)",
    )
    parser.add_argument(
        "--special-token",
        action="append",
        default=None,
        help="special token; repeat for multiple tokens (default: <|endoftext|>)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Training corpus not found: {input_path}")
    if args.vocab_size < 256:
        raise ValueError("--vocab-size must be at least 256")

    special_tokens = args.special_token or ["<|endoftext|>"]
    if args.vocab_size < 256 + len(special_tokens):
        raise ValueError("--vocab-size is smaller than the base and special-token vocabulary")
    implementations = (
        IMPLEMENTATIONS if args.implementation == "both" else (args.implementation,)
    )

    total_started = time.perf_counter()
    summaries: list[tuple[str, float, int, int, float, float, Path]] = []
    print(f"Corpus: {input_path} ({input_path.stat().st_size / 1024**3:.2f} GiB)")
    print(f"Target vocab size: {args.vocab_size:,}")

    total_memory = MemoryMonitor()
    total_memory.start()
    for index, module_name in enumerate(implementations, start=1):
        gc.collect()
        print(f"\n[{index}/{len(implementations)}] Training {module_name}.py", flush=True)
        trainer_class = importlib.import_module(module_name).BpeTokenizer
        trainer = trainer_class()
        trainer.add_special_tokens(special_tokens.copy())

        memory = MemoryMonitor()
        memory.start()
        started = time.perf_counter()
        vocab, merges = trainer.train(
            str(input_path), args.vocab_size, show_progress=True
        )
        elapsed = time.perf_counter() - started
        memory.stop()

        artifact_dir = args.output_dir / module_name
        metadata: dict[str, object] = {
            "implementation": f"{module_name}.py",
            "input": str(input_path),
            "input_bytes": input_path.stat().st_size,
            "target_vocab_size": args.vocab_size,
            "actual_vocab_size": len(vocab),
            "merge_count": len(merges),
            "special_tokens": special_tokens,
            "elapsed_seconds": elapsed,
            "ram_baseline_bytes": memory.baseline_bytes,
            "ram_peak_bytes": memory.peak_bytes,
            "ram_increase_bytes": memory.increase_bytes,
        }
        save_artifacts(artifact_dir, vocab, merges, metadata)
        summaries.append(
            (
                module_name,
                elapsed,
                len(vocab),
                len(merges),
                mib(memory.peak_bytes),
                mib(memory.increase_bytes),
                artifact_dir,
            )
        )
        print(
            f"Finished {module_name}.py in {elapsed:.2f}s; "
            f"peak RAM={mib(memory.peak_bytes):.2f} MiB "
            f"(+{mib(memory.increase_bytes):.2f} MiB); saved to {artifact_dir}"
        )
        del trainer, vocab, merges

    total_elapsed = time.perf_counter() - total_started
    total_memory.stop()
    print("\nTraining summary")
    for (
        module_name,
        elapsed,
        vocab_count,
        merge_count,
        peak_ram_mib,
        ram_increase_mib,
        artifact_dir,
    ) in summaries:
        print(
            f"  {module_name}.py: {elapsed:.2f}s, vocab={vocab_count:,}, "
            f"merges={merge_count:,}, peak RAM={peak_ram_mib:.2f} MiB "
            f"(+{ram_increase_mib:.2f} MiB), output={artifact_dir}"
        )
    print(f"Total elapsed time: {total_elapsed:.2f}s ({total_elapsed / 60:.2f} min)")
    print(
        f"Total peak RAM: {mib(total_memory.peak_bytes):.2f} MiB "
        f"(+{mib(total_memory.increase_bytes):.2f} MiB)"
    )


if __name__ == "__main__":
    main()
