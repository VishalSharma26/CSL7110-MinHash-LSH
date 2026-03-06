
import sys
import os

repo_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

def read_docs():
    all_text = {}
    for i in [1, 2, 3, 4]:
        txt_file = os.path.join(repo_path, "data",  f"D{i}.txt")
        with open(txt_file, 'r') as fp:
            all_text[f"D{i}"] = fp.read().strip()
    return all_text

def read_txt(txt_file):
    txt_file = os.path.join(repo_path, "data", txt_file)
    with open(txt_file, 'r') as fp:
        return fp.read().strip()
    return None

def load_user_movies(u_data_path):
    users = {}
    with open(u_data_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            uid = int(parts[0])
            mid = int(parts[1])
            users.setdefault(uid, set()).add(mid)
    return users