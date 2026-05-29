# NLP-TEAM 17 Final Model Repository

IMPORTANT: all datasets and gated models used by this project require access
approval before they can be downloaded or loaded. Set a Hugging Face token with
access to the required models/datasets in Colab secrets, `.env`, or the shell
environment before running anything.

This repository contains the cleaned reproduction bundle for **NLP-TEAM 17**,
our final one-pass adapter evaluation on `meta-llama/Llama-3.2-3B-Instruct`.
The final model reuses the trained one-pass adapter checkpoint and evaluates it
with the selected inference-time correction coefficients.

## Final Configuration

- Public method name: `NLP-TEAM 17`
- Base model: `meta-llama/Llama-3.2-3B-Instruct`
- Refusal judge: `WildGuard`
- Base trained adapter checkpoint: `output/adapter_v85_onepass.pt`
- Final eval setting: `jb1p5_or2p5_thr0p2`
- `JB_SCALE = 1.5`
- `OR_SCALE = 2.5`
- `ABSTAIN_THRESHOLD = 0.20`
- Metric output path:
  `output/metrics/nlp_team17_sweep/jb1p5_or2p5_thr0p2_metrics.json`

All user-facing files and documentation in this cleaned bundle use
`NLP-TEAM 17` / `nlp_team17`.

## Datasets And Access

You must be able to access the following before running the full pipeline.
Some are gated or may require accepting license terms on Hugging Face.

Model and judge:

- `meta-llama/Llama-3.2-3B-Instruct`
- WildGuard model used by `src/classifier.py`

Main v-series data loaders in `src/dataset.py`:

- `walledai/XSTest`
- `bench-llm/or-bench`
- `furonghuang-lab/PHTest`
- `walledai/HarmBench`
- `allenai/wildjailbreak`
- `walledai/AdvBench`
- `JailbreakBench/JBB-Behaviors`
- `walledai/MaliciousInstruct`
- `walledai/StrongREJECT` or fallback `csHugging/StrongREJECT`
- `tatsu-lab/alpaca`

Baseline experiment data:

- RepBend uses `allenai/wildguardmix` for its paper-style training pool.
- X-Boundary data is fetched from `https://github.com/AI45Lab/X-Boundary` with
  `experiment/xboundary/fetch_xboundary.sh`.
- The comparison baselines can also use our manifest-derived training pools via
  `experiment/methods/our_training_data.py`.

## What Is Included

- `notebooks/nlp_team17_train_inference.ipynb`
  - Single notebook entry point for reproducing NLP-TEAM 17 final inference.
  - Loads the trained adapter checkpoint and evaluates the final coefficient
    setting.
- `src/nlp_team17_adapter.py`
  - NLP-TEAM 17 adapter wrapper around the trained one-pass adapter.
- `src/moe_v85_adapter.py`, `src/adapter_flow.py`, `src/adapter_history_flow.py`
  - Core adapter architecture and history-flow experts.
- `src/model.py`, `src/dataset.py`, `src/classifier.py`
  - Llama loading/generation, dataset loading, and WildGuard refusal labeling.
- `experiment/repbend/` and `experiment/xboundary/`
  - Comparison baseline code used in the paper-style evaluation.

This compact repository intentionally excludes runtime caches, generated
responses, model checkpoints, and figure outputs.

## What You Need To Run

Runtime:

- Colab or a Python environment with PyTorch, Transformers `<5.0`,
  SentencePiece, pandas, NumPy, scikit-learn, Matplotlib, datasets, and tqdm.
- Hugging Face authentication with access to Llama-3.2-3B-Instruct and the
  datasets listed above.
- GPU is required for generation and WildGuard evaluation at the original scale.
  Cached representation-only plots can run on CPU if their cache artifacts
  already exist.

Required artifacts for the final NLP-TEAM 17 inference notebook:

```text
output/adapter_v85_onepass.pt
cache/unified_vseries_3b/eval_off.json
cache/unified_vseries_3b/eval_emb_off_v85.pt
```

The final notebook writes the selected run under:

```text
cache/nlp_team17/sweep/jb1p5_or2p5_thr0p2/eval_on.json
cache/nlp_team17/sweep/jb1p5_or2p5_thr0p2/eval_emb_on.pt
output/metrics/nlp_team17_sweep/jb1p5_or2p5_thr0p2_metrics.json
output/metrics/nlp_team17_sweep_summary.json
```

Optional artifacts for reproducing representation-analysis figures:

```text
cache/v07/emb_train.pt
cache/v85_onepass/directions.pt
cache/v85_onepass/targets.pt
cache/v85_onepass/subspace_target.pt
```

## How To Train / Prepare The Adapter

NLP-TEAM 17 does not train a new adapter in the final notebook. It reuses the
trained one-pass adapter checkpoint:

```text
output/adapter_v85_onepass.pt
```

To reproduce the full training lineage from scratch:

1. Obtain dataset/model access and set `HF_TOKEN`.
2. Build the v-series manifest and evaluation split caches in the full project
   environment.
3. Train the base one-pass adapter to produce
   `output/adapter_v85_onepass.pt`.
4. Copy that checkpoint into this cleaned repository root.
5. Run `notebooks/nlp_team17_train_inference.ipynb` to reproduce the final
   NLP-TEAM 17 inference setting.

This split is intentional: the final model selection is an inference-time
coefficient setting on top of the trained one-pass adapter.

## How To Run Inference

1. Open `notebooks/nlp_team17_train_inference.ipynb` in Colab.
2. Mount Google Drive and make sure the project root points to this repository.
3. Ensure these files exist:

```text
output/adapter_v85_onepass.pt
cache/unified_vseries_3b/eval_off.json
cache/unified_vseries_3b/eval_emb_off_v85.pt
```

4. Run the notebook from top to bottom.
5. The default grid runs only the final setting:

```python
JB_SCALE_GRID = [1.5]
OR_SCALE_GRID = [2.5]
ABSTAIN_THRESHOLD_GRID = [0.20]
```

6. The notebook generates responses with Llama-3.2-3B-Instruct, labels
   responses with WildGuard, saves adapter-on representation caches, and writes
   group-level metrics.

## Final Metrics

Metrics below are from the final `jb1p5_or2p5_thr0p2` run. The cleaned
NLP-TEAM 17 notebook writes the same setting under `output/metrics/nlp_team17_sweep/`.

| group | n | refuse off -> on | ASR off -> on | pure_OR off -> on |
|---|---:|---:|---:|---:|
| `jb_corr` | 153 | 10.5% -> 28.1% (+17.6pp) | 78.4% -> 54.2% (-24.2pp) | 0.7% -> 0.0% (-0.7pp) |
| `or_corr` | 67 | 97.0% -> 80.6% (-16.4pp) | 0.0% -> 0.0% (+0.0pp) | 97.0% -> 80.6% (-16.4pp) |
| `jb_blocked` | 205 | 100.0% -> 99.5% (-0.5pp) | 0.0% -> 0.0% (+0.0pp) | 1.5% -> 1.0% (-0.5pp) |
| `harm_refuse` | 173 | 100.0% -> 100.0% (+0.0pp) | 0.0% -> 0.0% (+0.0pp) | 0.6% -> 0.6% (+0.0pp) |
| `benign_ans` | 404 | 0.2% -> 0.0% (-0.2pp) | 0.2% -> 0.5% (+0.2pp) | 0.2% -> 0.0% (-0.2pp) |

Overall:

- refusal: 45.9% -> 47.3% (+1.4pp)
- ASR: 12.1% -> 8.5% (-3.6pp)
- pure_OR: 7.1% -> 5.7% (-1.4pp)
- `net_jb_asr_pp = -6.97`

## Verification Status

- `MANIFEST.txt` matches the actual file list in this bundle.
- Python source syntax was checked successfully.
- Notebook JSON was checked successfully.
- Cache files, checkpoints, generated outputs, and binary model artifacts are
  intentionally not included.
- `notebooks/` intentionally contains only `nlp_team17_train_inference.ipynb`.
