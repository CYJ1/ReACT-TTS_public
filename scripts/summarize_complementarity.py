from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("results/complementarity")
SEEDS = list(range(42, 52))

frames = []

for s in SEEDS:
    path = ROOT / f"seed{s}.csv"

    if not path.exists():
        print("MISSING:", path)
        continue

    df = pd.read_csv(path)
    df["seed"] = s
    frames.append(df)

all_df = pd.concat(frames, ignore_index=True)

print("\n========== GLOBAL 10-SEED SUMMARY ==========")

counts = all_df["category"].value_counts()

for cat in [
    "face_rescue",
    "face_hurt",
    "both_correct",
    "both_wrong",
]:
    n = int(counts.get(cat, 0))
    print(
        f"{cat:15s}: "
        f"{n:4d} "
        f"({100*n/len(all_df):.1f}%)"
    )

rescue = int(counts.get("face_rescue", 0))
hurt = int(counts.get("face_hurt", 0))

print(f"\nNet rescue = {rescue-hurt:+d}")


print("\n========== PER-SEED ==========")

for s in SEEDS:
    df = all_df[all_df["seed"] == s]

    rescue = (df["category"] == "face_rescue").sum()
    hurt = (df["category"] == "face_hurt").sum()

    print(
        f"seed {s}: "
        f"rescue={rescue:2d} "
        f"hurt={hurt:2d} "
        f"net={rescue-hurt:+d}"
    )


print("\n========== PER-EMOTION ==========")

emotion_stats = []

for emo, df in all_df.groupby("true_emotion"):
    rescue = (df["category"] == "face_rescue").sum()
    hurt = (df["category"] == "face_hurt").sum()

    n = len(df)

    emotion_stats.append(
        {
            "emotion": emo,
            "n": n,
            "rescue": rescue,
            "hurt": hurt,
            "net": rescue - hurt,
            "rescue_rate": rescue / n,
            "hurt_rate": hurt / n,
        }
    )

emo_df = pd.DataFrame(emotion_stats).sort_values(
    "net",
    ascending=False,
)

print(
    emo_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}",
    )
)


print("\n========== SAMPLE CONSENSUS ==========")

group_cols = [
    "sample_index",
    "dialogue_id",
    "utterance_id",
    "true_emotion",
    "context_text",
    "target_text",
]

rows = []

for keys, df in all_df.groupby(group_cols, dropna=False):

    rescue = (df["category"] == "face_rescue").sum()
    hurt = (df["category"] == "face_hurt").sum()
    both_correct = (df["category"] == "both_correct").sum()
    both_wrong = (df["category"] == "both_wrong").sum()

    rows.append({
        "sample_index": keys[0],
        "dialogue_id": keys[1],
        "utterance_id": keys[2],
        "true_emotion": keys[3],
        "context_text": keys[4],
        "target_text": keys[5],

        "rescue_seeds": rescue,
        "hurt_seeds": hurt,
        "both_correct_seeds": both_correct,
        "both_wrong_seeds": both_wrong,

        "net_rescue": rescue - hurt,

        "mean_true_prob_delta":
            df["true_prob_delta_face_minus_text"].mean(),

        "mean_gate_react":
            df["face_gate_react"].mean(),
    })

sample_df = pd.DataFrame(rows)

sample_df = sample_df.sort_values(
    ["net_rescue", "rescue_seeds"],
    ascending=False,
)

print("\nTop repeated FACE RESCUES:")

cols = [
    "sample_index",
    "true_emotion",
    "rescue_seeds",
    "hurt_seeds",
    "net_rescue",
    "mean_true_prob_delta",
    "mean_gate_react",
]

print(
    sample_df[cols]
    .head(20)
    .to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}",
    )
)


print("\nTop repeated FACE HURTS:")

print(
    sample_df
    .sort_values(
        ["net_rescue", "hurt_seeds"],
        ascending=[True, False],
    )[cols]
    .head(20)
    .to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}",
    )
)


print("\n========== GATE BY CATEGORY ==========")

for cat in [
    "face_rescue",
    "face_hurt",
    "both_correct",
    "both_wrong",
]:

    vals = all_df.loc[
        all_df["category"] == cat,
        "face_gate_react",
    ].dropna().values

    if len(vals) == 0:
        continue

    print(
        f"{cat:15s}: "
        f"{vals.mean():.4f} ± "
        f"{vals.std(ddof=1):.4f} "
        f"(n={len(vals)})"
    )


print("\n========== CONSISTENT SAMPLES ==========")

for threshold in [6, 7, 8, 9, 10]:

    n_rescue = (
        sample_df["rescue_seeds"] >= threshold
    ).sum()

    n_hurt = (
        sample_df["hurt_seeds"] >= threshold
    ).sum()

    print(
        f">={threshold}/10 seeds: "
        f"rescue samples={n_rescue}, "
        f"hurt samples={n_hurt}"
    )


out = ROOT / "consensus_10seed.csv"
sample_df.to_csv(out, index=False)

print("\nSaved:", out)