
# Part 4: MinHashing on MovieLens 100k
import sys  
import os
import random
import time
from pathlib import Path

from minhash_lsh_utility import repo_path, load_user_movies

def jaccard(a, b):
    u = a | b
    if not u:
        return 1.0
    return len(a & b) / len(u)


def all_pairs(user_ids):
    ids = list(user_ids)
    for i in range(len(ids)):
        u = ids[i]
        for j in range(i + 1, len(ids)):
            v = ids[j]
            yield u, v


def exact_pairs_ge_thresh(users, thresh=0.5):
    hits = set()
    for u, v in all_pairs(users.keys()):
        s = jaccard(users[u], users[v])
        if s >= thresh:
            hits.add((u, v))
    return hits


def make_hashes(t, m, seed):
    # h(x) = (a*x + b) % P % m
    P = 2_147_483_647  # prime
    rng = random.Random(seed)
    params = []
    for _ in range(t):
        a = rng.randrange(1, P - 1)
        b = rng.randrange(0, P - 1)
        params.append((a, b, P))
    return params


def minhash_sig(items, params, m):
    sig = []
    items_list = list(items)

    for (a, b, P) in params:
        best = None
        for x in items_list:
            hv = ((a * x + b) % P) % m
            if best is None or hv < best:
                best = hv
        sig.append(best)

    return sig


def est_from_sigs(sig1, sig2):
    same = 0
    for x, y in zip(sig1, sig2):
        if x == y:
            same += 1
    return same / len(sig1)


def approx_pairs_ge_thresh(users, t, m, seed, thresh=0.5):
    params = make_hashes(t, m, seed)

    # signatures for all users
    sigs = {}
    for uid, movies in users.items():
        sigs[uid] = minhash_sig(movies, params, m)

    hits = set()
    for u, v in all_pairs(sigs.keys()):
        s_hat = est_from_sigs(sigs[u], sigs[v])
        if s_hat >= thresh:
            hits.add((u, v))
    return hits


def write_pairs(out_path, pairs):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for u, v in sorted(pairs):
            f.write(f"{u}\t{v}\n")


def main():
    repo = Path(__file__).resolve().parents[1]
    u_data = os.path.join(repo_path, "data", "ml-100k", "u.data")

    users = load_user_movies(u_data)
    user_ids = sorted(users.keys())

    print("MovieLens loaded.")
    print("Users:", len(user_ids))
    print("Example user 1 rated movies:", len(users[user_ids[0]]))

    thresh = 0.5

    # ---- exact ----
    print("\nComputing EXACT pairs with Jaccard >= 0.5 ...")
    t0 = time.perf_counter()
    exact_hits = exact_pairs_ge_thresh(users, thresh=thresh)
    t1 = time.perf_counter()

    print(f"Exact pairs >= {thresh}: {len(exact_hits)}")
    print(f"Exact calc time: {(t1 - t0):.2f} s")
    write_pairs(repo / "output" / "part4_exact_pairs_ge_0.5.txt", exact_hits)

    # ---- minhash ----
    m = 20011  # > 10000 as required
    t_values = [50, 100, 200]
    runs = 5

    print("\nMinHash runs (each t averaged over 5 runs):")
    for t in t_values:
        fp_list = []
        fn_list = []
        hit_counts = []
        time_list = []

        for run in range(runs):
            seed = 1000 + run
            s0 = time.perf_counter()
            approx_hits = approx_pairs_ge_thresh(users, t=t, m=m, seed=seed, thresh=thresh)
            s1 = time.perf_counter()

            fp = len(approx_hits - exact_hits)
            fn = len(exact_hits - approx_hits)

            fp_list.append(fp)
            fn_list.append(fn)
            hit_counts.append(len(approx_hits))
            time_list.append(s1 - s0)

            out_file = repo / "output" / f"part4_minhash_t{t}_run{run+1}_pairs_ge_0.5.txt"
            write_pairs(out_file, approx_hits)

        fp_avg = sum(fp_list) / runs
        fn_avg = sum(fn_list) / runs
        time_avg = sum(time_list) / runs

        print(f"\n--- t = {t} ---")
        print(f"avg predicted pairs >= {thresh}: {sum(hit_counts)/runs:.1f}")
        print(f"avg false positives: {fp_avg:.1f}")
        print(f"avg false negatives: {fn_avg:.1f}")
        print(f"avg time per run: {time_avg:.2f} s")


if __name__ == "__main__":
    main()