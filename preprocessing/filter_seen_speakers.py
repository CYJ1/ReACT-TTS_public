import json
from pathlib import Path

TABLE = Path("../data/meld_gradtts/speaker_embeddings_train/speaker_table.json")
ROOT = Path("../data/meld_gradtts")

with TABLE.open() as f:
    speakers = set(json.load(f))


def filter_file(src, dst):
    kept = []
    dropped = []

    with open(src, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            x = json.loads(line)

            if x["target_speaker_id"] in speakers:
                kept.append(x)
            else:
                dropped.append(x)

    Path(dst).parent.mkdir(parents=True, exist_ok=True)

    with open(dst, "w", encoding="utf-8") as f:
        for x in kept:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    print(
        f"{src}: kept={len(kept)}, dropped={len(dropped)}"
    )

    if dropped:
        print(
            "  unseen:",
            sorted(set(x["target_speaker_id"] for x in dropped))
        )


filter_file(
    ROOT / "stage_b/dev.jsonl",
    ROOT / "stage_b/dev_seen.jsonl",
)

filter_file(
    ROOT / "stage_c/dev.jsonl",
    ROOT / "stage_c/dev_seen.jsonl",
)

filter_file(
    ROOT / "stage_c/test.jsonl",
    ROOT / "stage_c/test_seen.jsonl",
)
