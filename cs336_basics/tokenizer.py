from typing import Iterable
from pathlib import Path
import os
from typing import BinaryIO
from collections import defaultdict
import regex as re
import json
import pickle
from typing import Iterator

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.vocab_to_int = {v: k for k, v in self.vocab.items()}
        self.merges = merges
        #print("merges: ", merges)
        self.special_tokens = special_tokens if special_tokens is not None else []
        self.special_tokens.sort(key=len, reverse=True)
        self.pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.pat_compiled = re.compile(self.pat)

        ### Optimization
        self.merges_rank = {merges[idx]: idx for idx in range(len(merges))}
        self.token_cache = {} # list[bytes] -> list[int]
        self.cache_hits = 0
        self.cache_misses = 0

    @classmethod 
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        with open(vocab_filepath, "r") as f:
            vocab = json.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        #print('test: ', self.decode([8582, 247, 225]))
        #print("text to be encoded: ", text)
        pattern = "|".join(map(re.escape, self.special_tokens))
        #print("pattern: ", pattern)
        if self.special_tokens:
            chunks = re.split('(' + pattern + ')', text)
        else:
            chunks = [text]
        int_seqs = []
        #print("chunks: ", chunks)
        encoded_tokens_count = 0
        for chunk in chunks:
            if chunk in self.special_tokens:
                int_seqs.append(self.vocab_to_int[chunk.encode("utf-8")])
                continue
            tokens = self.pat_compiled.findall(chunk) # pre-tokenize
            #print("tokens: ", tokens)
            for token in tokens:
                #print("token: ", token)
                int_seqs.extend(self._encode_token(token))
                encoded_tokens_count += 1
                if encoded_tokens_count % 100 == 0:
                    print("encoded_tokens_count: ", encoded_tokens_count)
        
        # print("int_seqs: ", int_seqs)
        return int_seqs
    
    def _encode_token(self, token: str) -> list[int]:
        bytes_seq = token.encode("utf-8")
        if bytes_seq in self.vocab_to_int:
            return [self.vocab_to_int[bytes_seq]]
        if bytes_seq in self.token_cache:
            self.cache_hits += 1
            return self.token_cache[bytes_seq]
        
        bytes_seq_list = [bytes([byte]) for byte in bytes_seq]
        
        # print("bytes_seq: ", bytes_seq)
        #print("len(bytes_seq): ", len(bytes_seq))
        while True:
            #print("len(bytes_seq)123: ", len(bytes_seq))
            if len(bytes_seq_list) == 1:
                break
            merged = False
            best_rank, best_idx = None, None
            for i in range(len(bytes_seq_list) - 1):
                #print("i: ", i)
                #print("bytes_seq[i]: ", bytes_seq[i])
                pair = (bytes_seq_list[i], bytes_seq_list[i + 1])
                # print("pair: ", pair)
                if pair in self.merges_rank:
                    rank = self.merges_rank[pair]
                    if best_rank is None or rank < best_rank:
                        best_rank = rank
                        best_idx = i
                        merged = True
            
            if best_idx is not None:
                byte_concat = bytes_seq_list[best_idx] + bytes_seq_list[best_idx + 1]
                bytes_seq_list[best_idx] = byte_concat
                bytes_seq_list = bytes_seq_list[:best_idx+1] + bytes_seq_list[best_idx + 2:]
                merged = True
                continue
            
            if not merged:
                break
        #print("bytes_seq: ", bytes_seq)
        int_seq = []
        for byte in bytes_seq_list:
            int_seq.append(self.vocab_to_int[byte])
        self.token_cache[bytes_seq] = int_seq
        self.cache_misses += 1
        return int_seq
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        text = b""
        for id in ids:
            text += self.vocab[id]
        return text.decode("utf-8", errors="replace")
    

    
    

