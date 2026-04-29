# DA6401W Project Report Plan

**Author:** Rushikesh Chavan (DA25G508)
**Course:** DA6401W Deep Learning, IIT Madras
**Project:** Marcus — Real-Time Stoic Voice Agent

## 1. Honest project framing

The original aspirational title (*"Streaming Voice Agent with RNN-T &
Emotion Fusion"*) does **not** describe what we built. We built:

- Whisper-large-v3-turbo ASR (transformer encoder-decoder, **not** RNN-T)
- LoRA-fine-tuned Llama 3.2 3B (4-bit quantized, MLX) — Marcus Aurelius persona
- Kokoro 82M TTS (deep-baritone preset voice, **not** custom voice cloning)
- Half/full-duplex pipeline with auto-calibrated barge-in
- No acoustic emotion fusion — LLM sees only ASR text

**Reframe for the report**: focus on the DL substance we *did* deliver, which
is plenty for a 30-mark course project:

1. Parameter-efficient fine-tuning (LoRA) of a 3B quantized LLM for persona transfer
2. Composite reward function design and analysis
3. Empirical study: 104-example dataset → train/val loss dynamics, overfit characterisation, checkpoint comparison
4. Engineering: real-time audio pipeline with auto-calibrated VAD that handles speaker→mic echo without AEC

The most interesting *DL finding* in the project is honest and slightly
counter-intuitive: **base Llama with a strong system prompt outscored our
LoRA-adapter on a composite reward** despite the adapter clearly learning
a distinctive Stoic voice. This is publishable-quality observation about
the limits of small datasets vs. polished base models, and we should lead
with it.

**Suggested report title:**
*"LoRA Persona Transfer at the Limit: A 100-Example Study in Streaming
Stoic Voice Synthesis"*

---

## 2. Mark allocation strategy

| Section | Marks | Pages | Priority |
|---|---|---|---|
| Introduction | 1 | 0.5 | Low |
| Background / SOTA | 1 | 1 | Low |
| Problem Statement | 4 | 1.5 | **High** |
| Dataset Description | 2 | 1.5 | Medium |
| Methodology / Architecture | 4 | 4 | **High** |
| Theoretical Analysis | 2 | 1.5 | Medium |
| Data Preprocessing & Training | 3 | 2 | **High** |
| Experimental Results (plots) | 5 | 3 | **High** |
| Experimental Results (code) | 5 | 2 | **High** |
| Insights / Difficulties | 3 | 1.5 | **High** |
| **Total** | **30** | **~18-20** | |

The high-mark sections (Problem, Methodology, Training, Results, Insights
= 24/30) deserve careful writing and good figures. Intro/SOTA are
boilerplate.

---

## 3. Section-by-section content plan

### 3.1 Abstract (~150 words)
> We present *Marcus*, a real-time voice agent embodying the persona of
> Marcus Aurelius via parameter-efficient LoRA fine-tuning of a 4-bit
> quantized Llama 3.2 3B Instruct model on 104 hand-crafted instruction-
> response pairs derived from the Meditations, Discourses, and Seneca's
> Letters. The end-to-end pipeline (Whisper ASR → LoRA-Llama → Kokoro
> TTS) runs entirely locally on Apple Silicon via the MLX framework. We
> design a composite reward function combining Stoic-concept alignment,
> persona-marker consistency, length-suitability, and anachronism
> avoidance. We present a controlled training study showing val loss
> decreases from 3.43 to 2.59 within one epoch but overfits sharply by
> iter 50 with a 3B model on 100 examples. Surprisingly, the base
> instruction-tuned model outscores our adapter on the composite reward
> due to the metric rewarding the formulaic prose Llama already produces.
> We also contribute an auto-calibrated voice-activity detector that
> measures speaker-mic echo at startup and adapts the barge-in
> threshold per environment.

### 3.2 Introduction (1 mark, ~0.5 page)
- Motivation: voice-first conversational AI for stress management / journaling
- One-paragraph project sketch
- Contribution list (3-4 bullets)
- One short paragraph on scope: *"This project does not implement
  acoustic emotion fusion or a custom RNN-Transducer; we use Whisper for
  ASR and content-only LLM conditioning. See §9 for limitations."*

### 3.3 Background and State-of-the-Art (1 mark, ~1 page)
- Streaming ASR taxonomy: CTC, RNN-T, encoder-decoder transformers (Whisper)
- LLM persona adaptation: full fine-tuning vs. LoRA vs. prompting; cite Hu et al. 2021 (LoRA), Dettmers et al. 2023 (QLoRA)
- TTS: Tacotron, FastSpeech, neural codec models; Kokoro 82M
- On-device LLMs: Apple MLX, llama.cpp, GGUF; cite the 4-bit quantization paper (Frantar et al. GPTQ)
- Voice agents: Siri, Alexa, character.ai (cloud); on-device alternatives are an active research area
- *Why this combination matters*: real-time, private, offline-capable Stoic dialogue is a non-trivial engineering challenge even with off-the-shelf components

### 3.4 Problem Statement (4 marks, ~1.5 pages)

This section gets 4 marks — deserves rigour. Structure:

**Informal**: Build a real-time voice agent that responds as a Stoic philosopher.

**Formal**:

> Given an audio stream $X(t)$ from a microphone, we seek to produce an
> audio stream $Y(t)$ from a speaker such that:
> - $Y(t)$ is a coherent natural-language response delivered in a deep
>   male voice
> - The text content of $Y$, denoted $T_Y$, embodies Stoic philosophical
>   counsel in the persona of Marcus Aurelius, measured via a reward
>   function $R(T_Y) \in [0, 1]$ defined in §5.3
> - End-to-end latency $\Delta = t_{\text{end of }X} - t_{\text{start of }Y}$
>   satisfies $\Delta < \tau$ where $\tau$ is a usability threshold
>   (we target 4s cold, 1s warm)
> - Total memory consumption $M_{\text{total}} \leq 16$ GB (M2 Pro
>   unified memory budget)
> - The system supports user interruption (barge-in): if the user
>   begins speaking while $Y$ is playing, $Y$ is terminated and a new
>   response cycle begins

**Sub-problems**:
1. **Persona transfer with limited data**: Given $\sim$100 instruction-
   response pairs $\{(q_i, r_i)\}$, learn a low-rank update $\Delta W$
   to a base LLM such that responses to held-out prompts maximise $R$.
2. **Latency under streaming**: Decompose $\Delta$ into ASR, LLM, and
   TTS contributions; identify which stage dominates and how to overlap.
3. **Echo-resistant interruption detection**: Without acoustic echo
   cancellation, distinguish user voice from speaker bleed-back into
   the microphone.

**Constraints summary**: M2 Pro hardware, 16 GB RAM, no internet at
inference, no proprietary APIs, all weights local.

### 3.5 Dataset Description (2 marks, ~1.5 pages)

**Source corpora** (Table 1):

| Source | Pages | Words | Translation | Origin |
|---|---|---|---|---|
| Meditations | 256 | 36.5K | Long | Project Gutenberg #2680 |
| Discourses | 532 | 87.7K | Long | Project Gutenberg #10661 |
| Enchiridion | n/a | 12K | Carter | Project Gutenberg #45109 |
| Seneca: De Vita Beata, De Beneficiis, De Ira, De Clementia | 436 | 211K | Stewart | Project Gutenberg #56075 |
| Seneca: Letters to Lucilius | 436 | 30K (extracted) | Inwood (Oxford Clarendon) | User PDF |
| **Total** | | **~377K** | | |

**Processed corpus**: 1,761 passages, mean 160 words/passage, after
Gutenberg-boilerplate stripping, unicode normalisation, and paragraph-
boundary chunking with min/max word constraints (30, 300).

**Synthetic instruction-response pairs**: 104 pairs hand-authored by the
author, each consisting of:
- A modern user message (avg 17 words) describing a real-life concern
  (work stress, grief, anxiety, anger, parenting, etc.)
- A Marcus Aurelius response (avg 123 words) drawing on Stoic principles

The pairs span 50+ distinct life situations chosen for thematic diversity
(table of categories in appendix).

**Data quality assessment** — describe the composite reward distribution
on the training set:
- Mean reward score: 0.778
- 80% of pairs score above 0.7
- Min/max: 0.444 / 0.925

**Train/val split**: 90/10 via random shuffle (seed 42) → 93 train, 11 val.

### 3.6 Solution / Methodology / Architecture (4 marks, ~4 pages)

This is the largest section. Structure as several subsections:

#### 3.6.1 System overview
**Figure 1**: Pipeline block diagram
```
[Mic] → [VAD/AudioCapture] → [Whisper ASR] → [Conversation history] →
[Llama-3.2-3B-4bit + LoRA adapter] → [Sentence-streaming TTS (Kokoro)] →
[AudioPlayer] → [Speaker]
                 ↑                                           ↓
                 └────────[Barge-in detection] ──────────────┘
```

#### 3.6.2 ASR: Whisper-large-v3-turbo (MLX port)
- Encoder-decoder transformer, 798M params, 4-bit weights
- Input: 16 kHz mono float32 PCM
- Output: text string + word-level timestamps (we use only text)
- Why we did not use RNN-T: streaming RNN-T MLX implementations are
  immature; Whisper-turbo offers comparable accuracy and runs at 8.5x
  real-time on M2 Pro with utterance-level chunking which is sufficient
  for our human-in-the-loop turn-taking pattern

#### 3.6.3 LLM: Llama-3.2-3B-Instruct + LoRA
- Base: 3.2B parameters, 4-bit weight-only quantization (mlx-lm)
- LoRA adapter: rank 8, alpha 16, applied to last 16 transformer layers
- Trainable parameters: 6.95M (0.216% of total)

**LoRA math** (formal):
> For a frozen weight matrix $W_0 \in \mathbb{R}^{d \times k}$, the
> LoRA-augmented forward pass is
> $$h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r} B A x$$
> where $A \in \mathbb{R}^{r \times k}$ is initialised with Kaiming
> uniform and $B \in \mathbb{R}^{d \times r}$ is initialised to zero, so
> at start of training $\Delta W = 0$. Only $A$ and $B$ are updated.
> With $r = 8$, $d = 3072$, the per-layer trainable parameters are
> $r \cdot (d + k)$ instead of $d \cdot k$, a $\sim 200\times$ reduction.

#### 3.6.4 TTS: Kokoro-82M (deep baritone)
- Light 82M-parameter MLX-native TTS model
- Voice: `bm_george` (British male preset, deep)
- Sentence-level streaming: text from LLM is buffered until a sentence
  boundary, then synthesised independently to enable token-by-token UX

#### 3.6.5 Composite reward function
Provide the full formula:
$$R(t) = 0.375 \cdot R_{\text{stoic}}(t) + 0.250 \cdot R_{\text{persona}}(t) + 0.1875 \cdot R_{\text{noanachronism}}(t) + 0.1875 \cdot R_{\text{length}}(t)$$
or with feedback:
$$R'(t, f) = 0.80 \cdot R(t) + 0.20 \cdot \frac{f + 1}{2}, \quad f \in \{-1, +1\}$$

Define each component:
- $R_{\text{stoic}}$: presence of $k$ distinct Stoic concepts (dichotomy of control, amor fati, memento mori, virtue, logos, present moment, equanimity, sympatheia) divided by 3, capped at 1.
- $R_{\text{persona}}$: starts at 0.5, +0.1 per persona marker ("my friend", "consider"), -0.2 per anti-marker ("lol", "as an AI")
- $R_{\text{noanachronism}}$: 1.0 minus 0.3 per modern term ("smartphone", "internet", etc.)
- $R_{\text{length}}$: peaked function maximised at 50-150 words.

#### 3.6.6 Audio I/O and barge-in detection
**Figure 2**: VAD state machine (LISTENING → PROCESSING → SPEAKING → barge-in)

The VAD callback (running in the audio thread, 200ms chunks) decides
"is voiced" via:
$$\text{voiced} = \text{RMS}(c) > \theta(s)$$
where the threshold depends on agent state $s$:
$$\theta(s) = \begin{cases} \theta_0 & \text{if listening} \\ \max(\theta_0 m, 1.5 \cdot b_{\text{measured}}) & \text{if speaking} \end{cases}$$
$\theta_0$ is the static silence threshold, $m$ is the multiplier, and
$b_{\text{measured}}$ is the 95th-percentile RMS observed during a
3-second TTS sample played at startup (auto-calibration).

**Sustained-voice gate**: barge-in fires only after $\geq 3$ consecutive
voiced chunks (600 ms) — single-chunk noise spikes are rejected.

### 3.7 Theoretical Analysis (2 marks, ~1.5 pages)

Pick 2-3 short analyses:

**3.7.1 Why LoRA succeeds with $\sim 100$ examples**
- The full-fine-tune update space is ${O(d^2)}$ per layer; the LoRA
  update space is $O(r \cdot d)$. For $d = 3072$, $r = 8$, this is a
  $384\times$ reduction in degrees of freedom.
- With $\sim 100$ training examples, the full update would catastrophically
  overfit (param count is 4 orders of magnitude larger than data points).
- Rank-8 update is consistent with the "intrinsic dimension" hypothesis
  (Aghajanyan et al. 2020): persona adaptation lives in a low-rank subspace.

**3.7.2 Reward function bias toward base model**
- Show that $R_{\text{stoic}}$ saturates at 1.0 once 3 Stoic terms appear;
  base Llama 3.2 (instruction-tuned) easily generates 3+ Stoic terms in
  any "advise me" response, so the saturation is hit by both base and
  adapter.
- $R_{\text{persona}}$ rewards the formulaic *"my friend, consider..."*
  opening — base Llama happens to use this phrasing by default, so the
  base scores high without any fine-tuning.
- Implication: a reward function rewarding *generic markers* of a persona
  cannot discriminate well between models when the base already exhibits
  surface markers. Better metric: novelty or specificity of advice (not
  measured here).

**3.7.3 Latency lower bound**
- Time-budget decomposition: $\Delta_{\text{cold}} = T_{\text{ASR}} +
  T_{\text{LLM,first sentence}} + T_{\text{TTS,first sentence}}$.
- Measured: $\Delta_{\text{cold}} \approx 2.5 + 5.2 + 3.6 = 11.3$ s for
  the first response on a cold cache; subsequent sentences within the
  same response add only TTS latency ($\sim 0.2$ s each warm).
- With sentence streaming: perceived latency drops to $\Delta_{\text{
  perceived}} \approx 2.5 + 1.5 + 3.6 = 7.6$ s (LLM streams; we don't wait
  for full response).

### 3.8 Data Preprocessing and Training Methodology (3 marks, ~2 pages)

#### 3.8.1 Text preprocessing
1. Download .txt from Project Gutenberg / extract from PDF (Inwood)
2. Strip Gutenberg `*** START / END OF` markers
3. Unicode normalise (NFC), replace fancy quotes/dashes/ellipsis
4. Collapse multi-blank-lines, strip line trailing whitespace
5. Chunk into passages: split on double-newlines, merge short paragraphs,
   split long ones at sentence boundaries, final word-count window [30, 300]

#### 3.8.2 Synthetic pair generation
- Hand-authored by author across 12 thematic categories (work, grief, etc.)
- 104 pairs, JSONL format:
  ```json
  {"user": "I bombed an interview...", "marcus": "My friend, the interview is past..."}
  ```

#### 3.8.3 Training data formatting
- Convert to mlx-lm chat JSONL:
  ```json
  {"messages": [{"role": "system", "content": "..."},
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."}]}
  ```
- 90/10 train/val split (random shuffle, seed 42)

#### 3.8.4 Training loop
- Framework: `mlx_lm.lora` (MLX-native LoRA trainer)
- Optimiser: Adam (default in mlx-lm)
- Learning rate: $1 \times 10^{-4}$
- Batch size: 4
- Sequence length cap: 2048 tokens
- Mask prompt: True (loss computed only on assistant tokens)
- Max iterations: 23 (= 1 epoch with batch 4 on 93 examples)
- Save every 5 iters: produces checkpoints at iter 5, 10, 15, 20, 25
- Val batches per evaluation: 5 (covers all 11 val examples)

#### 3.8.5 Hyperparameter exploration
We ran two configurations to characterise overfit behaviour:

| Config | Epochs | LoRA rank | Iters | Best val loss | Where? |
|---|---|---|---|---|---|
| Run 1 | 3 | 8 (mlx default) | 69 | 2.594 | iter 20 |
| Run 2 (this report) | 1 | 8 | 23 | 2.594 | iter 20 |

The 3-epoch run *also* hits its optimum at iter 20 then climbs to val
3.72 by iter 69 — confirming that 1 epoch is the right schedule for this
dataset size with this model.

### 3.9 Experimental Results (5 marks plots + 5 marks code, ~5 pages)

**Required plots** (8 figures):

| # | Figure | Data source | Generation status |
|---|---|---|---|
| F1 | Train + Val loss curve across iterations | log-extracted | Need to plot |
| F2 | Val loss vs. # of epochs (overfit characterisation) | log-extracted | Need to plot |
| F3 | Composite reward per checkpoint | run `compare_checkpoints.py` | Need to run |
| F4 | Component-wise reward decomposition (stoic / persona / length / anachronism) for adapter vs base on 5 prompts | from earlier eval | Have data, need to plot |
| F5 | End-to-end latency breakdown (stacked bar: ASR / LLM / TTS, cold vs warm) | measured earlier | Need to plot |
| F6 | Memory usage over time (model loading sequence) | psutil measurement | Have data, need to plot |
| F7 | VAD calibration: bleed RMS distribution during 3s test | run `marcus calibrate` and capture | Need to run |
| F8 | Word count distribution of training pairs (histogram) | trivial | Need to plot |

**Tables**:
- T1: Source corpora summary (already drafted in §3.5)
- T2: Hyperparameter configuration (in §3.8)
- T3: Sample qualitative outputs (5 prompts × {base, adapter}) — already
  have this data from earlier eval
- T4: Reward score table per checkpoint

**Code listings** (use `\begin{lstlisting}` for these — 5 marks for code):

| # | Listing | File | Lines |
|---|---|---|---|
| L1 | Composite reward function | `src/marcus/rewards/composite.py` | full |
| L2 | LoRA training command builder | `src/marcus/training/sft.py:train_sft_local` | function |
| L3 | VAD callback with auto-calibrated barge-in | `src/marcus/pipeline/audio_io.py:_audio_callback` | function |
| L4 | Agent main loop with sentence-streaming TTS | `src/marcus/pipeline/agent.py:run_streaming` | function |
| L5 | Auto-calibration of barge-in threshold | `src/marcus/pipeline/agent.py:_calibrate_barge_in_threshold` | function |

### 3.10 Inferences / Insights / Difficulties (3 marks, ~1.5 pages)

Honest section — this is where we *should* admit the things that didn't
work, since that's where the deep learning is.

**Insight 1: Base instruction-tuned LLMs are surprisingly hard to beat with $<\!200$ examples on generic metrics.**
> Our LoRA adapter learned a distinctive direct Stoic voice ("Weep,
> then.") but scored 0.115 lower on composite reward than the base model
> because the base already produces formulaic Stoic-sounding prose ("My
> dear friend, I sense the weight..."). The adapter's distinctive style
> hurts it on a metric rewarding generic persona markers.

**Insight 2: 1 epoch is the right schedule for $\sim 100$ examples on a 3B model.**
> Past iter 25 the model memorises training data verbatim. Train loss
> drops to 0.21, val loss climbs to 3.72 (worse than the 3.43 baseline).

**Insight 3: Acoustic echo is the fundamental barrier to barge-in without AEC.**
> No static threshold works across hardware. Per-environment auto-
> calibration during preload (3-second TTS bleed measurement) is
> essential. Even with calibration, severe-bleed environments (laptop
> speakers at full volume) require headphones for snappy interrupt.

**Insight 4: Whisper hallucinates on near-silence in characteristic ways.**
> Single-word repetition ("freaking freaking freaking..."), stock phrases
> ("Thanks for watching"). Mitigations: `no_speech_threshold=0.8`,
> `condition_on_previous_text=False`, plus a post-hoc filter rejecting
> 4+ consecutive identical tokens.

**Difficulties**:
- macOS UF_HIDDEN flag silently breaking editable installs (Python
  3.11+ skips UF_HIDDEN .pth files). Fix: ship a `sitecustomize.py`
  that adds `src/` to `sys.path` directly, bypassing the .pth mechanism.
- mlx-lm API changes between releases (`temp` removed in favour of
  `sampler` callable).
- Limited training data from a single author limits diversity; future
  work would generate $\sim 1000$ pairs via Claude API for a fair
  comparison against base.

**Future work**:
- GRPO reinforcement-learning fine-tune from real user thumbs up/down
  feedback (collect via the existing `/feedback` log)
- Acoustic emotion fusion: an emotion encoder + projection MLP feeding
  hidden-state hooks at a middle Llama layer (the original ambitious
  architecture)
- Replace Whisper with an MLX-native streaming RNN-T (Parakeet TDT
  has an MLX port but is less stable)

---

## 4. Plots to generate (concrete commands)

### 4.1 Training loss curves (F1, F2)
Source: training log values already captured in chat history.
```python
# scripts/plot_training.py
import matplotlib.pyplot as plt

iters = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 69]
val_loss = [3.434, 2.675, 2.596, 2.620, 2.594, 2.651, 3.119, 3.099,
            3.171, 3.245, 3.351, 3.726, 3.705, 3.638, 3.608]
train_loss = [None, 2.899, 2.682, 2.700, 2.639, 2.074, 0.899, 0.945,
              0.963, 1.009, 0.407, 0.228, 0.226, 0.252, 0.210]

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(iters, val_loss, marker="o", label="Val loss")
ax.plot(iters, train_loss, marker="s", label="Train loss")
ax.axvline(x=20, color="gray", linestyle="--", alpha=0.5,
           label="Best val (iter 20, val 2.59)")
ax.axvline(x=23, color="green", linestyle=":", alpha=0.5,
           label="1 epoch boundary")
ax.set(xlabel="Iteration", ylabel="Loss",
       title="LoRA Training: Val Loss Bottoms at Iter 20, Then Overfits")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("report/figures/F1_training_loss.pdf")
```

### 4.2 Component-wise reward (F4)
Already have the data from the eval run. Plot as a grouped bar chart
(5 prompts × 2 models × 4 reward components).

### 4.3 Latency stacked bar (F5)
Cold start: ASR 2.56s + LLM 7.91s + TTS 6.76s = 17.23s.
Warm: ASR ~0.5 + LLM-first-sentence ~1.0 + TTS-first-sentence ~0.2 = 1.7s.

### 4.4 Memory bar chart (F6)
Already have the values: baseline 23 MB, +ASR 1725 MB, +LLM 3969 MB,
+TTS 3854 MB, after inference 804 MB.

### 4.5 Reward per checkpoint (F3)
Run:
```bash
./marcus train sft  # not needed if checkpoints exist
uv run python scripts/compare_checkpoints.py --adapter-dir adapters/marcus-sft-v1
```
Capture the table output, plot composite-reward column vs iter number.

### 4.6 Word count histogram (F8)
```python
import json, matplotlib.pyplot as plt
pairs = [json.loads(l) for l in open("data/synthetic/instruction_pairs.jsonl")]
marcus_words = [len(p["marcus"].split()) for p in pairs]
plt.hist(marcus_words, bins=20)
plt.axvline(50, color="red", linestyle="--"); plt.axvline(150, color="red", linestyle="--")
plt.xlabel("Words in Marcus response"); plt.ylabel("Count")
plt.title("Distribution of training-pair response lengths")
plt.savefig("report/figures/F8_word_dist.pdf")
```

---

## 5. References to cite

| Work | Cite as | Used for |
|---|---|---|
| Hu et al. 2021 — LoRA | `\cite{hu2021lora}` | LoRA method |
| Dettmers et al. 2023 — QLoRA | `\cite{dettmers2023qlora}` | 4-bit + LoRA |
| Frantar et al. 2022 — GPTQ | `\cite{frantar2022gptq}` | weight-only 4-bit quantization |
| Radford et al. 2022 — Whisper | `\cite{radford2022whisper}` | ASR |
| Aurelius — Meditations | `\cite{aurelius}` | dataset source |
| Aghajanyan et al. 2020 — Intrinsic Dim | `\cite{aghajanyan2020intrinsic}` | LoRA theoretical basis |
| Vaswani et al. 2017 — Transformer | `\cite{vaswani2017attention}` | architecture preliminaries |
| Ouyang et al. 2022 — InstructGPT / RLHF | `\cite{ouyang2022instruct}` | future-work GRPO context |
| Touvron et al. 2023 — Llama 2 / 3 | `\cite{touvron2023llama}` | base model |
| Apple MLX | website cite | framework |

---

## 6. Repo layout for the report

```
report/
├── REPORT_PLAN.md           ← this file
├── Project.tex              ← main report (use the supplied template)
├── ecsproject.cls           ← class file (already supplied)
├── Definitions.tex          ← macros (already supplied)
├── References.bib           ← BibTeX file
└── figures/                 ← all PDFs/PNGs referenced from Project.tex
    ├── F1_training_loss.pdf
    ├── F2_overfit.pdf
    ├── F3_reward_per_ckpt.pdf
    ├── F4_reward_components.pdf
    ├── F5_latency.pdf
    ├── F6_memory.pdf
    ├── F7_vad_calibration.pdf
    └── F8_word_dist.pdf
```

---

## 7. Writing schedule (suggested, 4-day plan)

| Day | Work |
|---|---|
| 1 | Generate all 8 plots; finalise data tables; write Methodology + Theoretical Analysis (largest sections) |
| 2 | Write Problem Statement + Dataset + Training Methodology |
| 3 | Write Experimental Results + Insights/Difficulties (use plots heavily) |
| 4 | Write Intro/Background/Abstract; format references; proofread; final compile |

---

## 8. Title-page values (for `\title`, `\authors`)

```latex
\title  {LoRA Persona Transfer at the Limit:
         A Streaming Stoic Voice Agent on Apple Silicon}
\authors{\texorpdfstring
         {\href{mailto:da25g508@smail.iitm.ac.in}{Rushikesh Chavan (DA25G508)}}
         {Rushikesh Chavan (DA25G508)}}
\course {DA6401W Deep Learning}
```
