
# Part 2: MinHashing (D1 vs D2) using character 3-grams

import sys
import os
import random
import time

repo_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


def read_txt(txt_file):
    txt_file = os.path.join(repo_path, "data", txt_file)
    with open(txt_file, 'r') as fp:
        return fp.read().strip()
    return None

def char_grams(s: str, k: int):
    out = set()
    for i in range(len(s) - k + 1):
        out.add(s[i:i+k])
    return out


def jaccard(a: set, b: set) -> float:
    u = a | b
    if not u:
        return 1.0
    return len(a & b) / len(u)


def gram_to_int(g: str) -> int:
    v = 0
    for ch in g:
        v = v * 257 + ord(ch)
    return v


def make_hash_params(t: int, m: int, seed: int):
    P = 2_147_483_647  
    rng = random.Random(seed)

    params = []
    for idx in range(t):
        a = rng.randrange(1, P - 1)
        b = rng.randrange(0, P - 1)
        params.append((a, b, P))
    return params


def minhash_signature(items_ints, hash_params, m):
    sig = []

    for (a, b, P) in hash_params:
        best = None
        for x in items_ints:
            hv = ((a * x + b) % P) % m
            if best is None or hv < best:
                best = hv
        sig.append(best)

    return sig


def sig_jaccard(sig1, sig2) -> float:
    same = 0
    for x, y in zip(sig1, sig2):
        if x == y:
            same += 1
    return same / len(sig1)


def run_once(d1_grams, d2_grams, t: int, m: int, seed: int):
    x1 = [gram_to_int(g) for g in d1_grams]
    x2 = [gram_to_int(g) for g in d2_grams]

    hp = make_hash_params(t=t, m=m, seed=seed)

    t0 = time.perf_counter()
    s1 = minhash_signature(x1, hp, m)
    s2 = minhash_signature(x2, hp, m)
    est = sig_jaccard(s1, s2)
    t1 = time.perf_counter()

    return est, (t1 - t0)


def main():

    d1 = read_txt("D1.txt")
    d2 = read_txt("D2.txt")

    k = 3
    g1 = char_grams(d1, k)
    g2 = char_grams(d2, k)

    exact = jaccard(g1, g2)

    m = 20011  

    print("Part 2: MinHashing (D1 vs D2)")
    print(f"Using character {k}-grams")
    print(f"m = {m}")
    print(f"|G(D1)|={len(g1)}, |G(D2)|={len(g2)}")
    print(f"Exact Jaccard (for reference): {exact:.6f}")
    print("-" * 60)

    # A) required t values
    t_list = [20, 60, 150, 300, 600]

    seed = 123

    print("A) Required runs (report these 5 numbers):")
    for t in t_list:
        est, secs = run_once(g1, g2, t=t, m=m, seed=seed)
        err = abs(est - exact)
        print(f"t={t:<3}  J_hat={est:.6f}   time={secs*1000:.1f} ms   |err|={err:.6f}")

    print("-" * 60)

    # B) extra experiments 
    extra_t = [10, 30, 80, 120, 200, 400, 800]

    print("B) Extra quick experiment (optional, helps choosing a good t):")
    for t in extra_t:
        est, secs = run_once(g1, g2, t=t, m=m, seed=seed)
        err = abs(est - exact)
        print(f"t={t:<3}  J_hat={est:.6f}   time={secs*1000:.1f} ms   |err|={err:.6f}")



if __name__ == "__main__":
    main()