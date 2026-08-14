from pathlib import Path
import json
import numpy as np

ROOT = Path("results/test_stage_a")
EMOTIONS = [
    "anger", "disgust", "fear",
    "joy", "neutral", "sadness", "surprise"
]
LABELS = np.arange(7)
N_BOOT = 10000
BOOT_SEED = 2026


def load_json(condition, seed):
    path = ROOT / f"{condition}_seed{seed}.json"
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open() as f:
        return json.load(f)


def load_npz(condition, seed):
    path = ROOT / f"{condition}_seed{seed}.npz"
    if not path.exists():
        raise FileNotFoundError(path)

    return np.load(path)


def macro_f1(y, pred, num_classes=7):
    vals = []

    for c in range(num_classes):
        tp = np.sum((y == c) & (pred == c))
        fp = np.sum((y != c) & (pred == c))
        fn = np.sum((y == c) & (pred != c))

        denom = 2 * tp + fp + fn

        if denom == 0:
            f1 = 0.0
        else:
            f1 = 2 * tp / denom

        vals.append(f1)

    return float(np.mean(vals))


def accuracy(y, pred):
    return float(np.mean(y == pred))


def mean_std(vals):
    vals = np.asarray(vals, dtype=float)
    return vals.mean(), vals.std(ddof=1)


# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------

groups = {
    "temporal": list(range(42, 52)),
    "text_only": list(range(42, 52)),
    "static": list(range(42, 47)),
    "full": list(range(42, 47)),
}

data = {}

for cond, seeds in groups.items():
    data[cond] = {}

    for seed in seeds:
        data[cond][seed] = {
            "json": load_json(cond, seed),
            "npz": load_npz(cond, seed),
        }


# ------------------------------------------------------------
# SANITY CHECK
# ------------------------------------------------------------

reference_y = data["temporal"][42]["npz"]["y_true"]

assert len(reference_y) == 261

for cond, seeds in groups.items():
    for seed in seeds:
        y = data[cond][seed]["npz"]["y_true"]

        if not np.array_equal(reference_y, y):
            raise RuntimeError(
                f"Test ordering mismatch: {cond} seed{seed}"
            )

print("\nAll Test files use identical 261-sample ordering.")


# ------------------------------------------------------------
# MAIN SUMMARY
# ------------------------------------------------------------

print("\n============== TEST SUMMARY ==============")

for cond, seeds in groups.items():

    print(f"\n========== {cond.upper()} ==========")

    accs = []
    f1s = []
    cccs = []

    for seed in seeds:

        m = data[cond][seed]["json"]["metrics_correct"]

        acc = m["accuracy"]
        f1 = m["macro_f1"]
        ccc = m.get("ccc_mean", np.nan)

        accs.append(acc)
        f1s.append(f1)
        cccs.append(ccc)

        print(
            f"seed {seed} | "
            f"Acc={acc:.4f} | "
            f"MacroF1={f1:.4f} | "
            f"CCC={ccc:.4f}"
        )

    ma, sa = mean_std(accs)
    mf, sf = mean_std(f1s)
    mc, sc = mean_std(cccs)

    print("\nSummary:")
    print(f"Accuracy : {ma:.4f} ± {sa:.4f}")
    print(f"Macro-F1 : {mf:.4f} ± {sf:.4f}")
    print(f"CCC      : {mc:.4f} ± {sc:.4f}")


# ------------------------------------------------------------
# 10-SEED TEMPORAL VS TEXT
# ------------------------------------------------------------

print("\n\n============== TEMPORAL vs TEXT-ONLY ==============")

temp_f1 = []
text_f1 = []

temp_acc = []
text_acc = []

for seed in range(42, 52):

    tm = data["temporal"][seed]["json"]["metrics_correct"]
    tx = data["text_only"][seed]["json"]["metrics_correct"]

    df1 = tm["macro_f1"] - tx["macro_f1"]
    da = tm["accuracy"] - tx["accuracy"]

    temp_f1.append(tm["macro_f1"])
    text_f1.append(tx["macro_f1"])

    temp_acc.append(tm["accuracy"])
    text_acc.append(tx["accuracy"])

    print(
        f"seed {seed}: "
        f"Temporal={tm['macro_f1']:.4f} | "
        f"Text={tx['macro_f1']:.4f} | "
        f"DeltaF1={df1:+.4f} | "
        f"DeltaAcc={da:+.4f}"
    )

temp_f1 = np.asarray(temp_f1)
text_f1 = np.asarray(text_f1)

delta = temp_f1 - text_f1

print("\n10-seed summary:")

print(
    f"Temporal : "
    f"{temp_f1.mean():.4f} ± {temp_f1.std(ddof=1):.4f}"
)

print(
    f"Text-only: "
    f"{text_f1.mean():.4f} ± {text_f1.std(ddof=1):.4f}"
)

print(
    f"Paired Delta Macro-F1: "
    f"{delta.mean():+.4f} ± {delta.std(ddof=1):.4f}"
)

print(
    f"Temporal > Text: {(delta > 0).sum()}/10"
)

print(
    f"Temporal = Text: {np.isclose(delta, 0).sum()}/10"
)

print(
    f"Temporal < Text: {(delta < 0).sum()}/10"
)


# ------------------------------------------------------------
# FAIR 5-SEED ABLATION
# ------------------------------------------------------------

print("\n\n============== 5-SEED FAIR ABLATION ==============")

for cond in [
    "text_only",
    "static",
    "full",
    "temporal",
]:

    vals = []

    for seed in range(42, 47):

        vals.append(
            data[cond][seed]["json"]
            ["metrics_correct"]["macro_f1"]
        )

    vals = np.asarray(vals)

    print(
        f"{cond:12s}: "
        f"{vals.mean():.4f} ± {vals.std(ddof=1):.4f}"
    )


# ------------------------------------------------------------
# EMOTION-WISE F1
# ------------------------------------------------------------

print("\n\n============== EMOTION-WISE TEST F1 ==============")

print(
    f"{'Emotion':10s} "
    f"{'Temporal':>18s} "
    f"{'Text-only':>18s} "
    f"{'Delta':>10s} "
    f"{'Wins':>10s}"
)

print("-" * 76)

for idx, emo in enumerate(EMOTIONS):

    tvals = np.array([
        data["temporal"][s]["json"]
        ["metrics_correct"]["per_class_f1"][idx]
        for s in range(42, 52)
    ])

    xvals = np.array([
        data["text_only"][s]["json"]
        ["metrics_correct"]["per_class_f1"][idx]
        for s in range(42, 52)
    ])

    d = tvals - xvals

    wins = int((d > 0).sum())
    ties = int(np.isclose(d, 0).sum())
    losses = int((d < 0).sum())

    print(
        f"{emo:10s} "
        f"{tvals.mean():.4f}±{tvals.std(ddof=1):.4f} "
        f"{xvals.mean():.4f}±{xvals.std(ddof=1):.4f} "
        f"{d.mean():+9.4f} "
        f"{wins}/{ties}/{losses}"
    )

print("\nWins = Temporal / tie / Text-only")


# ------------------------------------------------------------
# CORRECT vs MISMATCHED LISTENER
# ------------------------------------------------------------

print("\n\n============== CORRECT vs MISMATCHED LISTENER ==============")

correct_f1 = []
mismatch_f1 = []

correct_acc = []
mismatch_acc = []

for seed in range(42, 52):

    j = data["temporal"][seed]["json"]

    c = j["metrics_correct"]
    m = j["metrics_mismatch"]

    df1 = c["macro_f1"] - m["macro_f1"]
    da = c["accuracy"] - m["accuracy"]

    correct_f1.append(c["macro_f1"])
    mismatch_f1.append(m["macro_f1"])

    correct_acc.append(c["accuracy"])
    mismatch_acc.append(m["accuracy"])

    print(
        f"seed {seed}: "
        f"Correct={c['macro_f1']:.4f} | "
        f"Mismatch={m['macro_f1']:.4f} | "
        f"DeltaF1={df1:+.4f} | "
        f"DeltaAcc={da:+.4f}"
    )

correct_f1 = np.asarray(correct_f1)
mismatch_f1 = np.asarray(mismatch_f1)

cf_delta = correct_f1 - mismatch_f1

print("\nCounterfactual summary:")

print(
    f"Correct   : "
    f"{correct_f1.mean():.4f} ± "
    f"{correct_f1.std(ddof=1):.4f}"
)

print(
    f"Mismatched: "
    f"{mismatch_f1.mean():.4f} ± "
    f"{mismatch_f1.std(ddof=1):.4f}"
)

print(
    f"Delta     : "
    f"{cf_delta.mean():+.4f} ± "
    f"{cf_delta.std(ddof=1):.4f}"
)

print(
    f"Correct > Mismatch: "
    f"{(cf_delta > 0).sum()}/10"
)

print(
    f"Correct = Mismatch: "
    f"{np.isclose(cf_delta, 0).sum()}/10"
)

print(
    f"Correct < Mismatch: "
    f"{(cf_delta < 0).sum()}/10"
)


# ------------------------------------------------------------
# BOOTSTRAP
#
# Resample the SAME Test sample indices across all seeds.
# For every bootstrap replicate:
#   1) recompute metric for each seed
#   2) compute paired condition difference
#   3) average that difference over seeds
#
# This estimates uncertainty from the finite Test set while
# preserving the paired evaluation design.
# ------------------------------------------------------------

rng = np.random.default_rng(BOOT_SEED)

n = len(reference_y)

temp_preds = np.stack([
    data["temporal"][s]["npz"]["y_pred_correct"]
    for s in range(42, 52)
])

text_preds = np.stack([
    data["text_only"][s]["npz"]["y_pred_correct"]
    for s in range(42, 52)
])

correct_preds = temp_preds

mismatch_preds = np.stack([
    data["temporal"][s]["npz"]["y_pred_mismatch"]
    for s in range(42, 52)
])


def bootstrap_delta(pred_a, pred_b, metric_fn):
    values = np.empty(N_BOOT, dtype=float)

    for b in range(N_BOOT):

        idx = rng.integers(
            0,
            n,
            size=n,
        )

        yb = reference_y[idx]

        seed_deltas = []

        for s in range(pred_a.shape[0]):

            a = metric_fn(
                yb,
                pred_a[s, idx],
            )

            bb = metric_fn(
                yb,
                pred_b[s, idx],
            )

            seed_deltas.append(
                a - bb
            )

        values[b] = np.mean(seed_deltas)

    return values


print("\n\n============== PAIRED BOOTSTRAP (10,000) ==============")


# Temporal vs Text-only Macro-F1
boot_tt_f1 = bootstrap_delta(
    temp_preds,
    text_preds,
    macro_f1,
)

lo, hi = np.percentile(
    boot_tt_f1,
    [2.5, 97.5],
)

print("\nTemporal - Text-only / Macro-F1")
print(f"Bootstrap mean Delta = {boot_tt_f1.mean():+.4f}")
print(f"95% CI = [{lo:+.4f}, {hi:+.4f}]")
print(
    f"P(Delta > 0) = "
    f"{np.mean(boot_tt_f1 > 0):.4f}"
)


# Temporal vs Text-only Accuracy
boot_tt_acc = bootstrap_delta(
    temp_preds,
    text_preds,
    accuracy,
)

lo_acc, hi_acc = np.percentile(
    boot_tt_acc,
    [2.5, 97.5],
)

print("\nTemporal - Text-only / Accuracy")
print(f"Bootstrap mean Delta = {boot_tt_acc.mean():+.4f}")
print(f"95% CI = [{lo_acc:+.4f}, {hi_acc:+.4f}]")


# Correct vs Mismatched Macro-F1
boot_cf_f1 = bootstrap_delta(
    correct_preds,
    mismatch_preds,
    macro_f1,
)

lo_cf, hi_cf = np.percentile(
    boot_cf_f1,
    [2.5, 97.5],
)

print("\nCorrect - Mismatched / Macro-F1")
print(f"Bootstrap mean Delta = {boot_cf_f1.mean():+.4f}")
print(f"95% CI = [{lo_cf:+.4f}, {hi_cf:+.4f}]")
print(
    f"P(Delta > 0) = "
    f"{np.mean(boot_cf_f1 > 0):.4f}"
)


# Correct vs Mismatched Accuracy
boot_cf_acc = bootstrap_delta(
    correct_preds,
    mismatch_preds,
    accuracy,
)

lo_cf_acc, hi_cf_acc = np.percentile(
    boot_cf_acc,
    [2.5, 97.5],
)

print("\nCorrect - Mismatched / Accuracy")
print(f"Bootstrap mean Delta = {boot_cf_acc.mean():+.4f}")
print(
    f"95% CI = "
    f"[{lo_cf_acc:+.4f}, {hi_cf_acc:+.4f}]"
)


# ------------------------------------------------------------
# SAVE BOOTSTRAP ARRAYS
# ------------------------------------------------------------

np.savez_compressed(
    ROOT / "bootstrap_10000.npz",
    temporal_minus_text_f1=boot_tt_f1,
    temporal_minus_text_accuracy=boot_tt_acc,
    correct_minus_mismatch_f1=boot_cf_f1,
    correct_minus_mismatch_accuracy=boot_cf_acc,
)

print("\nSaved:")
print(ROOT / "bootstrap_10000.npz")