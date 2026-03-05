
# Part 1: k-grams & exact Jaccard

import sys
import os
from itertools import combinations
import random

repo_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

def read_docs():
    all_text = {}
    for i in [1, 2, 3, 4]:
        txt_file = os.path.join(repo_path, "data",  f"D{i}.txt")
        with open(txt_file, 'r') as fp:
            all_text[f"D{i}"] = fp.read().strip()
    return all_text

def char_kgrams(text, k):
    grams = set()
    for i in range(len(text) - k + 1):
        grams.add(text[i:i+k])
    return grams

def word_kgrams(text, k):
    words = text.split()
    grams = set()
    for i in range(len(words) - k + 1):
        grams.add(tuple(words[i:i+k]))
    return grams

def jaccard(a,b):
    return len(a & b) / len(a | b)


def minhash_signature(items, t, m=20011):
    P = 2147483647
    hashes = []
    for idx in range(t):
        a = random.randint(1,P-1)
        b = random.randint(0,P-1)
        min_val = None
        for x in items:
            x_val = abs(hash(x))
            h = ((a*x_val + b) % P) % m
            if min_val is None or h < min_val:
                min_val = h
        hashes.append(min_val)
    return hashes


def approx_jaccard(sig1, sig2):

    same = 0
    for a,b in zip(sig1,sig2):
        if a == b:
            same += 1

    return same / len(sig1)


# -------------------------
# Main
# -------------------------
def main():
    docs = read_docs()

    # Build grams
    char2 = {}
    char3 = {}
    word2 = {}

    for name,text in docs.items():
        char2[name] = char_kgrams(text,2)
        char3[name] = char_kgrams(text,3)
        word2[name] = word_kgrams(text,2)

    print("\nExact Jaccard Similarity\n")

    for a,b in combinations(docs.keys(),2):

        print(f"{a}-{b} char2:", jaccard(char2[a],char2[b]))
        print(f"{a}-{b} char3:", jaccard(char3[a],char3[b]))
        print(f"{a}-{b} word2:", jaccard(word2[a],word2[b]))
        print()


    print("\nMinHash Approx Jaccard (D1 vs D2 using char3)\n")

    items1 = char3["D1"]
    items2 = char3["D2"]

    t_values = [20,60,150,300,600]

    for t in t_values:

        sig1 = minhash_signature(items1,t)
        sig2 = minhash_signature(items2,t)

        est = approx_jaccard(sig1,sig2)

        print("t =",t," -> ",est)


if __name__ == "__main__":
    main()