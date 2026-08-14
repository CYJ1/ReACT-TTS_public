# ReACT-TTS

Official implementation of:

**Listen Before You Speak: Response Planning from Listener Facial Reactions for Conversational Speech Generation**

ReACT-TTS explores whether a speaker can better plan *how to respond* by observing the listener's facial reaction immediately before speaking.

Unlike conventional conversational TTS systems that rely primarily on textual dialogue context, ReACT-TTS incorporates **pre-response listener facial reactions** as an additional conversational signal for response-style planning.

---

## Overview

Given a dialogue context and the listener's facial reaction before the target response, ReACT-TTS predicts a response-style representation that can be used for conversational speech generation.

```text
Dialogue Context ───────────────┐
                               │
Listener Reaction Frames       │
        │                      │
        ▼                      │
Temporal Reaction Encoder      │
        │                      │
        └──────► Fusion ◄──────┘
                    │
                    ▼
             Response Planning
                    │
                    ▼
            Style Representation
                    │
                    ▼
              Speech Generator
```

The listener reaction is extracted from the **1-second interval immediately preceding the target response**.

---

## Main Research Question

> Can the listener's facial reaction provide complementary information for planning the speaker's upcoming conversational response?

Our experiments investigate:

- Text-only response planning
- Static listener-face conditioning
- Temporal listener-reaction conditioning
- Explicit reaction-difference features
- Correct vs. mismatched listener reactions
- Downstream speech generation using the predicted response representation

---

## Dataset

Experiments are conducted on the **MELD** dataset using a strictly filtered dyadic conversational subset.

### Response-planning subset

| Split | Samples |
|------:|--------:|
| Train | 1,117 |
| Dev | 116 |
| Test | 261 |

The preprocessing protocol includes:

- Exactly two dialogue participants
- Three preceding context turns
- Target utterance duration ≥ 1 s
- 1-second pre-response listener-reaction window
- 16 sampled reaction frames
- Listener visibility filtering
- Face identity consistency filtering
- No Test-set threshold tuning

For speech-generation experiments, only target speakers with training-reference audio are retained:

| Split | Generation samples |
|------:|-------------------:|
| Dev | 115 |
| Test | 252 |

MELD itself is **not distributed in this repository**.

---

## Response Planning

The main response planner combines textual conversational context with temporally encoded listener facial reactions.

### Test Results

Results below are averaged over 10 random seeds.

| Method | Accuracy ↑ | Macro-F1 ↑ | CCC ↑ |
|---|---:|---:|---:|
| Text-only | 0.4521 ± 0.0316 | 0.2489 ± 0.0152 | 0.2173 ± 0.0463 |
| Temporal Listener Reaction | 0.4483 ± 0.0243 | **0.2582 ± 0.0131** | **0.2282 ± 0.0291** |

For Macro-F1, Temporal Listener Reaction outperformed Text-only in **7/10 seeds**.

Bootstrap evaluation over Test samples gives:

```text
Δ Macro-F1 = +0.0094
95% CI = [-0.0057, +0.0247]
P(Δ > 0) = 0.884
```

Because the confidence interval includes zero, we interpret this result as a **positive tendency rather than a statistically significant improvement**.

---

## Ablation Study

A controlled 5-seed ablation compares different visual-reaction representations.

| Variant | Macro-F1 ↑ |
|---|---:|
| Text-only | 0.2367 ± 0.0021 |
| Static Listener Face | 0.2429 ± 0.0272 |
| Temporal + Explicit Δ | 0.2407 ± 0.0191 |
| **Temporal Listener Reaction** | **0.2558 ± 0.0135** |

These results suggest that temporal modeling is useful for capturing listener-reaction dynamics, while explicitly supplying frame-difference features does not provide an additional benefit in this setting.

---

## Listener-Mismatch Analysis

To examine whether the model uses listener-specific reaction information, we compare the corresponding listener reaction with a mismatched reaction.

| Listener Input | Macro-F1 ↑ |
|---|---:|
| Correct Listener | **0.2582 ± 0.0131** |
| Mismatched Listener | 0.2526 ± 0.0117 |

Bootstrap analysis:

```text
Δ Macro-F1 = +0.0055
95% CI = [-0.0027, +0.0135]
P(Δ > 0) = 0.911
```

The corresponding listener reaction tends to perform better than the mismatched condition, although the confidence interval again includes zero.

---

## Speech Generation

We additionally connect the response planner to a Grad-TTS-based acoustic model to test whether the predicted response representation can be consumed by an end-to-end speech-generation pipeline.

The generation model uses:

- Grad-TTS acoustic backbone
- Monotonic Alignment Search (MAS)
- 128-bin mel spectrogram
- 16 kHz audio
- HiFi-GAN vocoder
- 256-D speaker embedding
- 256-D response-style embedding

The acoustic condition is formed as:

```text
Speaker Embedding [256]
        +
Response Style [256]
        ↓
Global Condition [512]
        ↓
Grad-TTS
        ↓
Mel Spectrogram
        ↓
HiFi-GAN
        ↓
Waveform
```

The response planner is frozen during the final generation stage, and a trainable adapter maps its response-style representation into the acoustic style space.

The current speech-generation experiment is intended primarily as an **end-to-end feasibility study**, rather than as a claim of improved acoustic quality.

---

## Mel-Spectrogram Configuration

```text
Sample rate : 16000 Hz
n_fft       : 1024
hop_length  : 160
win_length  : 1024
n_mels      : 128
f_min       : 0
f_max       : 8000
```

---

## Repository Structure

```text
ReACT-TTS/
├── configs/
│   └── ...
├── react_tts/
│   ├── data/
│   ├── models/
│   └── tts/
│       └── grad_tts/
├── scripts/
│   ├── train_grad_stage_b.py
│   ├── train_grad_stage_c.py
│   ├── infer_grad_stage_c.py
│   ├── infer_grad_stage_c_all.py
│   ├── evaluate_generated_speech.py
│   └── ...
└── README.md
```

---

## Training Pipeline

The current implementation follows three stages.

### Stage A — Response Planning

Train the response planner using dialogue context and listener facial reactions.

The primary configuration uses temporal listener-reaction features without explicit difference features.

### Stage B — Acoustic Pretraining

Train the Grad-TTS acoustic model using ground-truth emotion/style supervision and speaker embeddings.

### Stage C — Planner-to-Speech Adaptation

Freeze the Stage-A response planner and connect its predicted style representation to the acoustic model through a trainable response-style adapter.

```text
Frozen Stage-A Planner
        ↓
Predicted Style
        ↓
ResponseStyleAdapter
        ↓
Acoustic Style
        ↓
Grad-TTS
```

---

## Inference

A single utterance can be generated using:

```bash
python3 -m scripts.infer_grad_stage_c \
    --config <CONFIG> \
    --checkpoint <CHECKPOINT>
```

Batch inference is available through:

```bash
python3 -m scripts.infer_grad_stage_c_all \
    --config <CONFIG> \
    --checkpoint <CHECKPOINT> \
    --output_dir <OUTPUT_DIR>
```

Exact arguments may vary depending on the experimental configuration.

---

## Evaluation

Generated speech can be evaluated using:

```bash
python3 -m scripts.evaluate_generated_speech \
    --temporal_dir <TEMPORAL_OUTPUT_DIR> \
    --text_dir <TEXT_ONLY_OUTPUT_DIR>
```

The current evaluation pipeline includes:

- ASR Word Error Rate (WER)
- Character Error Rate (CER)
- Speaker similarity

Response-planning evaluation additionally reports:

- Accuracy
- Macro-F1
- Concordance Correlation Coefficient (CCC)
- Multi-seed statistics
- Bootstrap confidence intervals

---

## Generated Samples

Generated audio and experiment artifacts are not included in this public repository.

They are stored separately to avoid distributing large binary artifacts and dataset-derived materials.

Audio demonstrations may be released separately.

---

## Requirements

The implementation has been tested with Python 3.10 and PyTorch-based training/inference.

Core dependencies include:

```text
torch
torchvision
librosa
soundfile
pyworld
einops
resemblyzer
faster-whisper
```

Please refer to the repository configuration and scripts for experiment-specific requirements.

---

## Citation

Citation information will be added after publication.

```bibtex
@inproceedings{chu2026listen,
  title     = {Listen Before You Speak: Response Planning from Listener Facial Reactions for Conversational Speech Generation},
  author    = {Yunji Chu},
  year      = {2026}
}
```

---

## Acknowledgements

This implementation builds upon ideas and components from prior work on conversational speech synthesis, Grad-TTS, Face-TTS, HiFi-GAN, and multimodal affect modeling.

Please refer to the paper for complete citations and discussion of related work.

---

## License

Please refer to `LICENSE` for repository licensing information.

Third-party datasets, pretrained models, and external components remain subject to their original licenses.
