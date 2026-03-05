
# Part 3: LSH probability using S-curve

import sys  
import os
from itertools import combinations

repo_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

def read_docs():
    all_text = {}
    for i in [1, 2, 3, 4]:
        txt_file = os.path.join(repo_path, "data",  f"D{i}.txt")
        with open(txt_file, 'r') as fp:
            all_text[f"D{i}"] = fp.read().strip()
    return all_text


def char_grams(s, k=3):
    return {s[i:i+k] for i in range(len(s) - k + 1)}


def jaccard(a, b):
    return len(a & b) / len(a | b)


def f_prob(s, r, b):
    return 1 - (1 - (s ** b)) ** r


def main():
    docs = read_docs()

    # given in assignment: t = 160, tau = 0.7
    t = 160
    tau = 0.7

    # choose r*b = 160 (20*8 is a nice combo near tau~0.7)
    r = 20     # number of bands
    b = 8      # rows per band

    print("Part 3: LSH (S-curve probability)")
    print(f"t={t}, tau={tau}, chosen r={r}, b={b} (r*b={r*b})")
    print(f"f(tau) = {f_prob(tau, r, b):.6f}")
    print("-" * 55)

    grams = {name: char_grams(txt, 3) for name, txt in docs.items()}

    for a, c in combinations(["D1", "D2", "D3", "D4"], 2):
        s = jaccard(grams[a], grams[c])
        p = f_prob(s, r, b)
        print(f"{a}-{c}:  s={s:.6f}  ->  P(candidate)={p:.6f}")


if __name__ == "__main__":
    main()