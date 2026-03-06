
# Part 5: LSH on MovieLens 100k

import sys  
import os
import random
import time
from minhash_lsh_utility import repo_path, load_user_movies

def jaccard(a, b):
    u = a | b
    if not u:
        return 1.0
    return len(a & b) / len(u)


def all_pairs(ids):
    ids = list(ids)
    for i in range(len(ids)):
        u = ids[i]
        for j in range(i + 1, len(ids)):
            v = ids[j]
            yield u, v


def exact_pairs_ge(users, thresh):
    hits = set()
    for u, v in all_pairs(users.keys()):
        if jaccard(users[u], users[v]) >= thresh:
            hits.add((u, v))
    return hits


def make_hashes(t, m, seed):
    P = 2_147_483_647
    rng = random.Random(seed)
    params = []
    for _ in range(t):
        a = rng.randrange(1, P - 1)
        b = rng.randrange(0, P - 1)
        params.append((a, b, P))
    return params


def minhash_sig(items, params, m):
    items_list = list(items)
    sig = []
    for (a, b, P) in params:
        best = None
        for x in items_list:
            hv = ((a * x + b) % P) % m
            if best is None or hv < best:
                best = hv
        sig.append(best)
    return sig


def est_sim(sig1, sig2):
    same = 0
    for x, y in zip(sig1, sig2):
        if x == y:
            same += 1
    return same / len(sig1)


def lsh_candidates(sigs, r, b):
    ids = list(sigs.keys())
    cand = set()

    for band_i in range(b):
        buckets = {} 
        start = band_i * r
        end = start + r

        for uid in ids:
            key = tuple(sigs[uid][start:end])
            buckets.setdefault(key, []).append(uid)

        for key, members in buckets.items():
            if len(members) < 2:
                continue
            members.sort()
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    cand.add((members[i], members[j]))

    return cand


def run_one(users, t, r, b, thresh, exact_hits, seed):
    m = 20011  
    params = make_hashes(t, m, seed)

    sigs = {}
    for uid, movies in users.items():
        sigs[uid] = minhash_sig(movies, params, m)

    cands = lsh_candidates(sigs, r=r, b=b)

    predicted = set()
    for u, v in cands:
        s_hat = est_sim(sigs[u], sigs[v])
        if s_hat >= thresh:
            predicted.add((u, v))

    fp = len(predicted - exact_hits)
    fn = len(exact_hits - predicted)
    return fp, fn, len(cands), len(predicted)


def avg_over_5_runs(users, t, r, b, thresh, exact_hits):
    fp_list, fn_list, cand_list, pred_list, time_list = [], [], [], [], []
    for run in range(5):
        seed = 5000 + run
        t0 = time.perf_counter()
        fp, fn, cands, preds = run_one(users, t, r, b, thresh, exact_hits, seed)
        t1 = time.perf_counter()

        fp_list.append(fp)
        fn_list.append(fn)
        cand_list.append(cands)
        pred_list.append(preds)
        time_list.append(t1 - t0)

    return {
        "fp_avg": sum(fp_list) / 5,
        "fn_avg": sum(fn_list) / 5,
        "cand_avg": sum(cand_list) / 5,
        "pred_avg": sum(pred_list) / 5,
        "time_avg": sum(time_list) / 5,
    }


def main():
    u_data = os.path.join(repo_path, "data", "ml-100k", "u.data")

    users = load_user_movies(u_data)
    print("Loaded users:", len(users))

    print("\nComputing exact pairs for threshold 0.6 ...")
    exact_06 = exact_pairs_ge(users, 0.6)
    print("Exact pairs >= 0.6:", len(exact_06))

    print("\nComputing exact pairs for threshold 0.8 ...")
    exact_08 = exact_pairs_ge(users, 0.8)
    print("Exact pairs >= 0.8:", len(exact_08))

    configs = [
        # (t, r, b)
        (50, 5, 10),
        (100, 5, 20),
        (200, 5, 40),
        (200, 10, 20),
    ]

    for thresh, exact_hits in [(0.6, exact_06), (0.8, exact_08)]:
        print("\n" + "=" * 70)
        print(f"LSH results for similarity threshold >= {thresh}")
        print("=" * 70)

        for (t, r, b) in configs:
            if t != r * b:
                print(f"Skipping invalid config t={t}, r={r}, b={b} (r*b != t)")
                continue

            res = avg_over_5_runs(users, t=t, r=r, b=b, thresh=thresh, exact_hits=exact_hits)

            print(f"\nConfig: t={t}, r={r}, b={b}")
            print(f"avg candidates: {res['cand_avg']:.1f}")
            print(f"avg predicted pairs >= {thresh}: {res['pred_avg']:.1f}")
            print(f"avg false positives: {res['fp_avg']:.1f}")
            print(f"avg false negatives: {res['fn_avg']:.1f}")
            print(f"avg time (5-run avg): {res['time_avg']:.2f} s")


if __name__ == "__main__":
    main()