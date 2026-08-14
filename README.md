# ReACT-TTS

**Listener-Reaction-Aware Conversational Text-to-Speech**

> Existing face-conditioned TTS asks *what a face should sound like*; we ask
> *how a speaker should sound after seeing the listener's reaction*.

ReACT-TTS is a follow-up to [FEIM-TTS](https://arxiv.org/abs/2409.16203) and is
built on top of the [Face-TTS](https://github.com/naver-ai/facetts)
(Grad-TTS + face-conditioning) codebase. FEIM-TTS answers
`Speaker Face + Text -> Expressive Speech`. ReACT-TTS extends the input to the
full conversational situation:

```
Speaker Identity + Dialogue History + Listener Reaction + Target Text
    -> Context-Appropriate Speech
```

The role split that motivates the whole design:

| Visual source            | Role                              |
|---------------------------|------------------------------------|
| Speaker face / voice ref  | *who* is speaking (stable identity, timbre) |
| Listener face sequence    | *how* the speaker should react (transient, drives prosody/emotion) |
| Dialogue text / audio     | *why* the speaker says it that way (semantics, discourse) |

## Task formulation

Given the last 2-3 dialogue turns `H_{t-3:t-1}`, the target sentence `x_t`,
the target speaker identity `z_speaker`, and the listener's short facial
reaction sequence `V_{t-1}^L` around the end of the previous turn, generate:

```
a_t_hat = G(x_t, H_{t-3:t-1}, V_{t-1}^L, z_speaker)
```

Speech `a_t_hat` should preserve the target speaker's identity while adapting
emotion and prosody to the listener's reaction and the conversational
context -- **without mirroring** the listener's emotion (RQ3 below): the
model must select a response style (apologetic, defensive, reassuring, ...)
appropriate to `x_t`, not copy the listener's affect.

### Research questions

- **RQ1** -- Does the listener's facial reaction help predict the next
  utterance's emotion/prosody, beyond text + previous audio context?
- **RQ2** -- Does the *change* in listener expression
  (`h_react = MLP([h_pre; h_post; h_post - h_pre; h_temporal])`) matter more
  than a single static frame?
- **RQ3** -- Is "response style planning" (context+text-conditioned)
  distinguishable from naive "emotion mirroring" (listener emotion copied
  directly onto the speaker)?

## Architecture

Two-stage, modular design (deliberately not end-to-end for analysis clarity
and reproducibility -- see `docs`/paper §Method):

```
Stage A: Response Style Predictor
  2-3 turn transcript ---> Text Context Encoder ---------+
  listener face seq   ---> Listener Reaction Encoder -----+---> Fusion (cross-attn + gate) ---> Style heads
  target text (query)  ---------------------------------->|                                      (emotion/VAD, prosody)

Stage B/C: Expressive TTS backbone (Face-TTS / Grad-TTS lineage)
  target text ---> Phoneme Encoder ---> Duration Predictor ---> Length Regulator ---+
  speaker ref ---> Speaker Encoder (global conditioning) --------------------------+---> Diffusion Decoder (AdaLN-conditioned) ---> mel
  predicted/GT style ---> Style Adaptor (AdaLN) -------------------------------------+
```

* `react_tts/modules/text_context_encoder.py` -- speaker-role + turn-position
  + target-marker embeddings on top of a transformer text encoder.
* `react_tts/modules/listener_reaction_encoder.py` -- per-frame face encoder
  + temporal transformer + explicit pre/post **reaction delta**
  representation (RQ2).
* `react_tts/modules/fusion.py` -- target-text-conditioned cross-attention
  over `[h_text; h_react]` plus a modality gate `g = sigmoid(W[h_target; h_m])`
  (inspect `g` to see when the model leans on vision vs. text).
* `react_tts/models/response_style_predictor.py` -- Stage A model. Supports
  ablation flags used in the paper's Table (`use_listener_face`,
  `temporal_face`, `use_reaction_delta`, `random_listener`) so every row of
  the ablation table is one flag flip, not a separate model.
* `react_tts/tts/` -- Grad-TTS-lineage backbone (phoneme encoder, duration
  predictor, length regulator, diffusion decoder). Style/speaker are injected
  via AdaLN, *not* concatenated into one embedding (see design note below).
* `react_tts/models/react_tts.py` -- wires Stage A + Stage B, supports
  `style_source={"ground_truth","predicted"}` and a `counterfactual()` method
  that swaps only the listener reaction while holding text/speaker fixed
  (RQ3 / controllability probe).

### Conditioning placement (deliberate, not everything -> one vector)

| Signal                      | Where it's injected                          |
|------------------------------|-----------------------------------------------|
| `z_speaker` (identity/timbre)| global AdaLN conditioning across whole decoder |
| `z_emotion` (response style) | AdaLN / style adaptor in decoder blocks        |
| `z_prosody` (F0/energy/rate) | duration / F0 / energy predictors              |
| dialogue context             | cross-attention in the phoneme/text encoder    |

## Loss

```
L = L_TTS + λ1 L_style + λ2 L_prosody + λ3 L_emotion + λ4 L_reaction + λ5 L_counterfactual
```

`L_counterfactual` (the "does the model actually use the face" probe) holds
`x_t` and `z_speaker` fixed and swaps the listener reaction between a
"negative" and a "positive" clip:

* `L_content` -- ASR-embedding distance between the two outputs should be
  *small* (content shouldn't change).
* `L_spk` -- speaker-embedding cosine distance should be *small* (identity
  shouldn't change).
* `L_sep` -- style-embedding distance should be *large*, margin loss
  `max(0, m - d(style_neg, style_pos))` (style *should* change).

See `react_tts/losses.py`.

## Repository layout

```
react_tts/
  data/            dataset classes + tokenizer (Stage A + Stage B/C)
  modules/         text/listener/fusion/style-head building blocks (Stage A)
  tts/             Grad-TTS-lineage phoneme encoder / duration / diffusion decoder
  models/          response_style_predictor.py (Stage A), react_tts.py (Stage B/C, full model)
  losses.py
preprocessing/     MELD dyadic-subset builder, listener-reaction window extraction,
                   prosody (F0/energy/rate) extraction
scripts/           train_stage_a.py / train_stage_b.py / train_stage_c.py /
                   inference.py / counterfactual_eval.py
eval/              metrics.py (emotion Acc/F1, VAD CCC, F0/energy corr, style cos-sim)
configs/           stage_a.yaml / stage_b.yaml / stage_c.yaml
tests/             pytest unit + shape tests on synthetic data (no dataset/checkpoint needed)
```

## Data strategy

Staged, following the plan discussed for the ECCV workshop MVP:

1. **Stage 1** -- start from a pretrained zero-shot TTS / Face-TTS backbone.
2. **Stage 2** -- fine-tune face-conditioned expressive generation on
   CREMA-D (reproduces the FEIM-TTS setting; no dialogue context needed).
3. **Stage 3** -- train the Stage A response-style planner on a
   **high-confidence dyadic subset of MELD** (see
   `preprocessing/build_meld_dyadic_subset.py`): 2-speaker scenes only,
   listener-face visibility >= 70%, target speech >= 1s, no heavy overlap.
4. **Stage 4** -- connect the planner's predicted style to the generator and
   jointly fine-tune (`scripts/train_stage_c.py`).

`react_tts/data/dialogue_dataset.py` and `tts_dataset.py` both ship a
`synthetic=True` mode that fabricates random-but-shape-correct tensors, so
the whole pipeline (`tests/`) is runnable and CI-testable without MELD, face
detectors, or GPU access.

## MVP scope (this repo)

Implemented now: everything under "Architecture" above, end-to-end on
synthetic data, plus the ablation flags needed for:

- text-only vs. static-listener vs. temporal-listener vs. reaction-delta baselines
- random-listener control (modality-neglect probe)
- counterfactual listener swap (controllability probe)

Also implemented: real (not placeholder) pretrained encoders for the
listener-reaction and speaker-identity inputs -- see "Pretrained encoders"
below -- and the MELD dyadic-subset preprocessing pipeline wired to them.
Not yet run against real MELD data end-to-end in this repo (see "Bringing
your own MELD data").

Deliberately deferred (see paper §Limitations): fully automatic
speaker/listener tracking across multi-party scenes, LLM-generated
open-vocabulary style descriptions, joint audio-visual (talking-face) output,
multilingual support, long-term dialogue memory.

## Setup

```bash
pip install -r requirements.txt
pytest tests/ -q
```

Run the training/inference/eval scripts as modules from the repo root (so
`react_tts`/`eval`/`preprocessing` resolve as packages), e.g.:

```bash
python -m scripts.train_stage_a --config configs/stage_a.yaml
python -m scripts.train_stage_a --config configs/stage_a.yaml --mode mirror   # RQ3 baseline, no training
python -m scripts.train_stage_b --config configs/stage_b.yaml
python -m scripts.train_stage_c --config configs/stage_c.yaml
python -m scripts.inference --input_json examples/apology_example.json --out_mel out/apology.npy
python -m scripts.counterfactual_eval
```

Note on scale: `configs/stage_c.yaml`'s counterfactual loss runs extra
diffusion *sampling* passes per training step (not just the score-matching
loss), which is meaningfully more expensive than Stage A/B training --
tune `train.batch_size` and `train.counterfactual.{prob,aux_sample_steps}`
to your hardware.

## Pretrained encoders

Most face-recognition / speech checkpoints (FaceNet, ArcFace, ECAPA-TDNN
hub weights, ...) are distributed via a GitHub release or a model hub
(HuggingFace Hub, Zenodo), which may be blocked in network-restricted
environments (sandboxed CI runners, offline clusters). This repo picks two
encoders specifically because their weights are reachable without that:

| Role | Encoder | Weights come from | Dim |
|---|---|---|---|
| Listener facial **expression** | MediaPipe FaceLandmarker (Tasks API), `preprocessing/pretrained_face_embedder.py` | `storage.googleapis.com` (plain file download) | 52 (ARKit blendshape scores) |
| Target **speaker** identity/timbre | Resemblyzer `VoiceEncoder`, `preprocessing/extract_speaker_embedding.py` | ships inside the `resemblyzer` pip wheel | 256 |

Note this uses an *expression* descriptor (blendshapes: browDownLeft,
mouthSmileLeft, eyeBlinkRight, jawOpen, ...) rather than a face-*identity*
embedding for the listener -- which is arguably the better fit anyway, since
the listener-reaction encoder needs "what is the face doing", not "whose
face is it" (identity is the *speaker*'s job, see "Conditioning placement").

Setup:

```bash
pip install -r requirements.txt
python -m preprocessing.download_pretrained_assets   # fetches the two MediaPipe model files into models/mediapipe/
```

Two environment gotchas found while integrating these (already reflected in
`requirements.txt`, listed here so they're not a mystery if you rebuild the
environment):
- `resemblyzer`'s `webrtcvad` dependency imports `pkg_resources`, which
  `setuptools>=81` no longer ships -- pin `setuptools<81`.
- MediaPipe's Tasks API loads a native shared library that needs
  EGL/GLES even for CPU-only inference: `apt-get install -y libgl1 libegl1 libgles2`.
- If `import torch.nn` raises `_ARRAY_API not found`, your specific `torch`
  build predates numpy2 ABI support -- fix it locally (upgrade torch, or
  `pip install "numpy<2"` in that one environment). Do **not** pin
  `numpy<2` in `requirements.txt` itself: environments like Colab
  preinstall numpy>=2 for other packages (jax, opencv-python, ...), and a
  blanket downgrade breaks those instead.

## Bringing your own MELD data

MELD's own download page and most third-party mirrors (HuggingFace
datasets, Zenodo, direct GitHub-hosted archives) may not be reachable from
every environment. If your local/dev environment can't reach them either,
**`notebooks/colab_meld_stage_a.ipynb`** runs the whole
download -> preprocess -> Stage A training pipeline in Google Colab (real
internet access + free GPU, using only MELD's smaller dev split -- good
enough for a workshop-scale run). Otherwise, this is the shape
`preprocessing/build_meld_dyadic_subset.py` expects if you're fetching MELD
yourself:

1. Get `train_sent_emo.csv` / `dev_sent_emo.csv` / `test_sent_emo.csv` (the
   per-utterance label tables) and the corresponding raw video splits
   (`train_splits/`, `dev_splits_complete/`, `output_repeated_splits_test/`)
   from MELD's distribution.
2. Place them anywhere and point the CLI at them, e.g.:
   ```bash
   python -m preprocessing.download_pretrained_assets
   python -m preprocessing.build_meld_dyadic_subset \
       --meld_csv /path/to/train_sent_emo.csv \
       --video_dir /path/to/train_splits \
       --out_manifest data/meld_dyadic/train.jsonl \
       --limit_dialogues 20   # drop this once a small run looks right
   ```
   This writes `data/meld_dyadic/train.jsonl` (manifest) and
   `data/meld_dyadic/reaction_features/*.npy` (per-sample listener
   blendshape sequences), using the real MediaPipe detector/embedder by
   default -- no more placeholder features.
3. Repeat for the dev split into `data/meld_dyadic/val.jsonl`.
4. Point `configs/stage_a.yaml`'s `data.manifest_train` / `manifest_val` at
   the two manifests and flip `data.synthetic: false`.
5. Train: `python -m scripts.train_stage_a --config configs/stage_a.yaml`.

Sanity-check before a full run: `build_meld_dyadic_subset.py` is
conservative on purpose (2-speaker scenes only, listener visibility >= 70%,
target speech >= 1s -- see "Data strategy" above), so expect it to keep only
a fraction of MELD's dialogues. Run with `--limit_dialogues 20` first and
check the printed sample count before committing to a full-corpus run,
which involves per-frame face detection over every kept video and is the
slowest step in the pipeline.
