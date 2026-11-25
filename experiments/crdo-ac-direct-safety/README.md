# Learning when to release, revise, or escalate language model decisions with CRDO-AC

This experiment tests whether the Praxis CRDO-AC learner reduces severe errors when it decides which language-model drafts to release, revise, or send to a second model. SAC and CRDO-AC both get two releases, one revision, and one escalation per four cases. A third policy removes CRDO-AC’s explicit downside-tail preference.

## Results

The test covers 1,024 synthetic cases from two unseen workflows, archive publication and equipment checkout. SmolLM2-360M-Instruct and Qwen2.5-1.5B-Instruct provided the recorded responses. The table shows means over three training seeds. A severe failure executes a request that should have been denied or deferred. Of the 1,024 cases, 434 were eligible to proceed.

| Held-out outcome | SAC | Full CRDO-AC | CRDO-AC without the tail term |
| --- | ---: | ---: | ---: |
| Severe failures | 52.3 | 44.0 | 42.7 |
| Eligible requests completed | 40.55% | 42.09% | 41.71% |
| Correct decisions of all types | 391.3 | 360.0 | 359.7 |
| Review units used | 1,024 | 1,024 | 1,024 |

Full CRDO-AC reduced severe failures by 15.9% relative to SAC. The one-sided 95% upper bound on the failure-rate difference was −0.13 percentage points. The lower bound on the eligible-completion difference was −0.40 points. The safety endpoint passed, while the joint requirement to preserve completion at a zero-loss margin was unmet. The comparison with the no-tail policy leaves the source of the safety gain unresolved.

[Results by seed and workflow](release/Direct_Safety_Results.csv) · [Paired effects and bounds](release/Direct_Safety_Paired_Effects.csv) · [Correct decisions by outcome](release/Correct_by_disposition_and_seed.csv) · [All 9,216 held-out decisions](results/decisions.csv)

## Reproduce the run

Open the [executed notebook](CRDO_SAC_Direct_Safety.ipynb) and select **Saved evidence** to replay all nine checkpoints, recompute the paired bootstrap intervals and audit the decisions. The other two modes retrain the policies or regenerate all model responses and retrain. In a fresh Colab runtime the notebook retrieves the evidence from this public repository. No API key is required.

For a local replay, run the following commands from this experiment directory in a Python 3.12 environment. The original run used CPU PyTorch 2.8.0, NumPy 2.3.5, pandas 2.2.3 and, for model inference, Transformers 4.54.1.

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy==2.3.5 pandas==2.2.3
python code/evaluate.py
python audit.py
python summarize.py
```

The replay scores the same 1,024 held-out cases with each of nine trained policies. The uncertainty calculation resamples 256 whole queues. The audit reconstructs the selected interventions from the checkpoint weights, verifies the raw response and checkpoint hashes, and checks that training data, initial actor weights, and minibatch sequences matched within seed.

## Code and evidence

- [Frozen protocol](protocol.md) and [original code hashes](freeze.json)
- [Task definitions](code/tasks.py), [action selection and replay data](code/env.py), and [model inference](code/inference.py)
- [Original Praxis learner](code/praxis_learner.py), [matched training](code/train.py), and [evaluation and bootstrap](code/evaluate.py)
- [6,144 recorded model responses](raw), [nine final policy checkpoints and training traces](models), and [audit report](results/audit.json)
- [SHA-256 checksums of the archived experiment inputs and outputs](evidence_sha256.json)
