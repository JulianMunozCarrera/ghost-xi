"""
Downloads the three open datasets into the data folder.

    python scripts/get_data.py

What you get (about 2 GB in total):
  data/metrica      3 matches tracked in the stadium         (Metrica Sports)
  data/skillcorner  20 A-League matches tracked from TV       (SkillCorner, MIT license)
  data/lastrow      19 Liverpool goals tracked from TV        (Last Row / Ricardo Tavares)

All three ask to be credited if you use them. See the README.

The SkillCorner tracking files are big, so GitHub stores them apart from
the rest of the repo (this is called Git LFS). We fetch those 20 files
one by one.
"""

import json
import os
import subprocess
import sys
import urllib.request

REPOS = {
    "metrica": "https://github.com/metrica-sports/sample-data",
    "skillcorner": "https://github.com/SkillCorner/opendata",
    "lastrow": "https://github.com/Friends-of-Tracking-Data-FoTD/Last-Row",
}
BIG_FILE = "https://media.githubusercontent.com/media/SkillCorner/opendata/master/data/matches/{id}/{id}_tracking_extrapolated.jsonl"


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "data"
    os.makedirs(root, exist_ok=True)

    for name, url in REPOS.items():
        target = os.path.join(root, name)
        if os.path.exists(target):
            print(name, "is already here")
            continue
        print("downloading", name)
        # GIT_LFS_SKIP_SMUDGE=1 means: do not fetch the big files yet. We do that below.
        subprocess.run(["git", "clone", "--depth", "1", url, target], check=True,
                       env={**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"})

    with open(os.path.join(root, "skillcorner", "data", "matches.json")) as f:
        ids = [m["id"] for m in json.load(f)]
    for n, match_id in enumerate(ids, start=1):
        path = os.path.join(root, "skillcorner", "data", "matches", str(match_id), f"{match_id}_tracking_extrapolated.jsonl")
        # Before the download this file is a tiny text note that points at the real one.
        if os.path.getsize(path) > 1_000_000:
            continue
        print(f"downloading tracking for match {n} of {len(ids)}")
        urllib.request.urlretrieve(BIG_FILE.format(id=match_id), path + ".part")
        os.replace(path + ".part", path)
    print("done")


if __name__ == "__main__":
    main()
