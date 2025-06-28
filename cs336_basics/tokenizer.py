import os
from typing import BinaryIO
import regex as re
import multiprocessing as mp
from collections import defaultdict, Counter
from tqdm import tqdm

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PAT_COMPILED = re.compile(PAT)


def find_chunk_boundaries(file: BinaryIO, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pretokenize_chunks(args):
    input_path, start, end, special_tokens = args

    with open(input_path, "rb") as f:
        f.seek(start)
        raw_text = f.read(end - start).decode("utf-8", errors="ignore")
        pattern = "|".join(map(re.escape, special_tokens))
        chunks = re.split(pattern, raw_text)

    byte_seqs = []
    for chunk in chunks:
        tokens = PAT_COMPILED.findall(chunk)
        byte_seqs.extend([t.encode("utf-8") for t in tokens])

    return byte_seqs


def bpe_train(input_path: str, vocab_size: int, special_tokens: list[str]):
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, desired_num_chunks=8, split_special_token="<|endoftext|>".encode("utf-8"))

    tasks = [(input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]
    with mp.Pool(processes=8) as pool:
        results = pool.map(pretokenize_chunks, tasks)

    all_byte_seqs = [b for byte_seqs in results for b in byte_seqs]
    int_seqs = [list(b) for b in all_byte_seqs]

    vocab = {i: bytes([i]) for i in range(256)}
    merges = []
    cur_index = 256
    for special_token in special_tokens:
        vocab[cur_index] = special_token.encode("utf-8")
        cur_index += 1

    pair_freqs = Counter()
    for seq in int_seqs:
        for i in range(len(seq) - 1):
            pair_freqs[(seq[i], seq[i + 1])] += 1

    pair_positions = defaultdict(Counter)
    for seq_idx, seq in enumerate(int_seqs):
        for i in range(len(seq) - 1):
            pair_positions[(seq[i], seq[i + 1])][seq_idx] += 1

    numberOfTokensToMerge = vocab_size - len(vocab)

    # Start training
    for _ in tqdm(range(numberOfTokensToMerge), desc="Training BPE"):
        if not pair_freqs:
            print("No more pairs to merge. Stopping early.")
            break

        # Find the pair with highest frequency
        max_freq = max(pair_freqs.values())
        candidates = [(pair, freq) for pair, freq in pair_freqs.items() if freq == max_freq]
        best_pair = max(candidates, key=lambda x: (vocab[x[0][0]], vocab[x[0][1]]))[
            0
        ]  # find the pair with largest lexicographic order
        token1, token2 = best_pair

        new_token = vocab[token1] + vocab[token2]
        merges.append((vocab[token1], vocab[token2]))

        vocab[cur_index] = new_token
        # Update all sequences to use the new merged token
        changed_seqs = {}
        for seq_idx in pair_positions[best_pair]:
            if seq_idx in changed_seqs:
                continue

            changed_seqs[seq_idx] = True
            seq = int_seqs[seq_idx]

            new_seq = []
            i = 0

            while i < len(seq):
                if i < len(seq) - 1 and seq[i] == token1 and seq[i + 1] == token2:
                    new_seq.append(cur_index)
                    if i > 0:
                        pair_freqs[(seq[i - 1], cur_index)] += 1
                        pair_positions[(seq[i - 1], cur_index)][seq_idx] += 1
                    if i < len(seq) - 2:
                        pair_freqs[(cur_index, seq[i + 2])] += 1
                        pair_positions[(cur_index, seq[i + 2])][seq_idx] += 1

                    if i > 0:
                        pair_freqs[(seq[i - 1], seq[i])] -= 1

                        if pair_positions[(seq[i - 1], seq[i])][seq_idx] > 0:
                            pair_positions[(seq[i - 1], seq[i])][seq_idx] -= 1

                    if i < len(seq) - 2:
                        pair_freqs[(seq[i + 1], seq[i + 2])] -= 1
                        if pair_positions[(seq[i + 1], seq[i + 2])][seq_idx] > 0:
                            pair_positions[(seq[i + 1], seq[i + 2])][seq_idx] -= 1
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1

            seq[:] = new_seq

        cur_index += 1
        pair_freqs[best_pair] = 0
        del pair_positions[best_pair]

    print(merges)
    # print(vocab)
    return vocab, merges


# Main
def main():
    input_path = "../tests/fixtures/corpus.en"
    vocab_size = 500
    special_tokens = ["<|endoftext|>"]

    bpe_train(input_path, vocab_size, special_tokens)


if __name__ == "__main__":
    main()