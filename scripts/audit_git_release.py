"""Scan reachable Git blobs and the index; report locations, never secret values."""

import json
from pathlib import PurePosixPath
import re
import subprocess


PATTERNS = {
    "private_key": rb"-----BEGIN (?:OPENSSH |RSA |EC |DSA |ENCRYPTED )?PRIVATE KEY-----",
    "github_token": rb"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b",
    "aws_access_key": rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b",
    "hf_token": rb"\bhf_[A-Za-z0-9]{30,}\b",
}


def git(*args):
    return subprocess.check_output(["git", *args])


def main():
    objects = {}
    for line in git("rev-list", "--objects", "--all").decode("utf-8").splitlines():
        oid, _, path = line.partition(" ")
        objects[oid] = path
    indexed = []
    for item in git("ls-files", "--stage", "-z").decode("utf-8").split("\0"):
        if not item:
            continue
        metadata, path = item.split("\t", 1)
        mode, oid, stage = metadata.split()
        objects[oid] = path
        indexed.append((mode, path))
    payload = ("\n".join(objects) + "\n").encode()
    listing = subprocess.check_output(["git", "cat-file", "--batch-check"], input=payload).decode().splitlines()
    blobs = [(line.split()[0], int(line.split()[2])) for line in listing if line.split()[1] == "blob"]
    stream = subprocess.Popen(["git", "cat-file", "--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    secrets, oversized = [], []
    for oid, size in blobs:
        if size >= 100 * 1024 * 1024:
            oversized.append({"path": objects[oid], "bytes": size})
        stream.stdin.write((oid + "\n").encode())
        stream.stdin.flush()
        stream.stdout.readline()
        data = stream.stdout.read(size)
        stream.stdout.read(1)
        if b"\0" not in data:
            for label, pattern in PATTERNS.items():
                if re.search(pattern, data):
                    secrets.append({"path": objects[oid], "kind": label, "blob": oid})
    stream.stdin.close()
    stream.wait()
    forbidden = []
    for mode, path in indexed:
        name = PurePosixPath(path).name.lower()
        if mode == "160000" or name in {".env", "kaggle.json", "credentials.json", "id_ed25519", "id_rsa"}:
            forbidden.append(path)
        if path.startswith(("results/", "logs/", "tmp/", ".venv/", "Mathematics/", "docs/")):
            forbidden.append(path)
        if path in {"PROJECT_GUIDE.md", "阅读顺序.md", "AGENTS.md", "feature_usage_plot.py",
                    "lamda_loss_plot_all.py", "shapley.py"}:
            forbidden.append(path)
        if PurePosixPath(path).suffix.lower() in {".pkl", ".pt", ".pth", ".npy", ".zip"}:
            forbidden.append(path)
    report = {"history_and_index_blobs": len(blobs), "indexed_files": len(indexed),
              "secret_pattern_findings": secrets, "oversized_blobs": oversized,
              "forbidden_index_paths": sorted(set(forbidden)),
              "limitations": "Pattern scan is not a guarantee; generic passwords and semantic disclosure require review."}
    print(json.dumps(report, indent=2))
    if secrets or oversized or forbidden:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
