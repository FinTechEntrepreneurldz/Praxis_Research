"""Rebuild the two paper tables and the descriptive accuracy breakdown."""
import json
from pathlib import Path

import pandas as pd

root = Path(__file__).resolve().parent
results = root / "results"
release = root / "release"
release.mkdir(exist_ok=True)

decisions = pd.read_csv(results / "decisions.csv")
by_domain = pd.read_csv(results / "domain_by_seed.csv")
pooled = pd.read_csv(results / "pooled_by_seed.csv").assign(domain="pooled")
summary = pd.concat([pooled, by_domain], ignore_index=True)
cases = decisions.groupby("domain")["id"].nunique().to_dict()
cases["pooled"] = decisions["id"].nunique()
summary["cases"] = summary["domain"].map(cases)
summary["severe_rate"] = summary["severe"] / summary["cases"]
summary["completion_rate"] = summary["completed"] / summary["eligible"]
summary.to_csv(release / "Direct_Safety_Results.csv", index=False)

effects = pd.DataFrame(json.loads((results / "paired.json").read_text()))
for metric in ("severe", "completion"):
    effects[[metric + "_lo95", metric + "_hi95"]] = effects[metric + "_ci"].tolist()
effects.drop(columns=["severe_ci", "completion_ci"]).to_csv(
    release / "Direct_Safety_Paired_Effects.csv", index=False
)

# A correct final answer also identifies the reference disposition for that case.
labels = ["EXECUTE", "ASK", "DEFER", "DECLINE"]
correct = decisions.loc[decisions["correct"].eq(1)]
fits = decisions.groupby(["mode", "seed"]).size().index
counts = correct.groupby(["mode", "seed", "final"]).size().unstack(fill_value=0)
counts = counts.reindex(index=fits, columns=labels, fill_value=0)
assert counts.sum(axis=1).eq(decisions.groupby(["mode", "seed"])["correct"].sum()).all()
counts.to_csv(release / "Correct_by_disposition_and_seed.csv")
