# A3: Self-Supervised Learning — Teaching Guide

---

## Opening: The Labeling Problem

**What to say:**

> "ImageNet has 1.2 million labeled images. It took years and thousands of human annotators. Your model trained on it — it has seen more labeled data than any human ever will. Now imagine you're a doctor trying to detect a rare disease. You have 200 MRI scans. That's it. Supervised learning collapses."

**The core question of this lab:**

> "What if the signal for learning came from the data itself — not from humans?"

---

## The SSL Intuition (before any code)

**Pause and ask students:**

> "If I show you two photos of the same dog — one bright, one dark, one cropped differently — do you still know it's the same dog?"

Of course. Because you understand the *concept* of a dog, not just pixel patterns. SSL tries to teach models this same invariance.

**The key idea:**

> "If two views come from the same image, their representations should be similar. If they come from different images, they should be different."

This is the entire foundation of contrastive SSL.

---

## Part 1: SimCLR (Cells 3–8)

### The story

**What to say:**

> "SimCLR (Chen et al., 2020) is the 'hello world' of modern SSL. It's simple enough to understand completely, powerful enough to match supervised learning on many benchmarks."

### Architecture walkthrough

```
Image x
 ├── Augment (random crop, flip, color jitter...) → x_i
 └── Augment (independently) → x_j

x_i → Encoder (ResNet) → h_i → Projector (MLP) → z_i ─┐
x_j → Encoder (ResNet) → h_j → Projector (MLP) → z_j ─┴─→ NT-Xent Loss
```

**Three things to explain:**

**1. Why two augmentations of the same image?**

> "That's the self-supervision signal. We're not saying 'this is a cat.' We're saying 'this crop and this crop came from the same image — they should look similar in feature space.' The label is the image identity, and we get it for free."

**2. Why a projector head?**

> "Counterintuitive result from the paper: the projector MLP actually *hurts* the representations for downstream tasks. They found that `h` (before the projector) works better for linear evaluation. The projector absorbs augmentation-specific information so the encoder doesn't have to."

**3. NT-Xent Loss (the loss function)**

> "For a batch of N images, we get 2N views. For each view, we want its pair to be closer than ALL other 2N-2 views. Think of it as a softmax over a similarity matrix."

$$\ell_{i,j} = -\log \frac{\exp(\text{sim}(z_i, z_j) / \tau)}{\sum_{k \neq i} \exp(\text{sim}(z_i, z_k) / \tau)}$$

**Temperature τ (tau):**

> "τ controls how sharp the distribution is. Low τ → sharper → model is penalized more for similar negatives. High τ → softer → easier task. SimCLR uses τ=0.5 — found by ablation."

**SimCLR's dirty secret — batch size:**

SimCLR needs **large batches** (4096+) in the original paper because more negatives = better contrastive signal. With small batches, there aren't enough negatives to make the task hard.

**Pause and ask students:**

> "If your batch size is 2 (1 image → 2 views), how many negatives does each view have? Is that a useful learning signal?"
>
> Answer: Zero other images in the batch, no negatives at all. The loss would always be trivially zero.

---

### Linear Evaluation (Cell 6)

**What to say:**

> "The gold standard test for SSL: freeze the encoder completely. Train only a linear layer on top using labels. If accuracy is high, the SSL encoder learned meaningful features — not just memorized augmentation artifacts."

This is important: we're testing whether SSL features are **linearly separable** — as good as or better than supervised features.

---

## Bridge: SimCLR → MoCo → BYOL → DINO

### SimCLR's Problem: Batch Size

SimCLR's contrastive signal comes entirely from the current batch. More negatives = stronger signal, so the original paper uses **batch size 4096** (32 TPUs). With batch 256 you get ~510 negatives — the task becomes too easy and the model stops learning meaningful structure.

**What to say:**

> "If your batch size is 2 — one image, two views — how many negatives do you have? Zero. The loss is trivially zero. SimCLR needs a crowd of negatives to work."

---

### MoCo: Decouple Batch Size from Negatives

MoCo (He et al., 2020) fixes the batch size problem with a **memory queue**:

```
Current batch (query) ──▶ Encoder q (backprop) ──▶ compare with ──▶ NT-Xent loss
                                                          ↑
                    Memory queue (65,536 keys) ◀── Momentum Encoder k (EMA, no backprop)
```

- **Memory queue**: stores encoded keys from *past* batches — 65k negatives regardless of current batch size
- **Momentum encoder**: the key encoder must be consistent with old keys still in the queue. If it changes too fast, old keys become stale and mislead training. Solution: EMA update with momentum = 0.999

$$\theta_k \leftarrow 0.999 \cdot \theta_k + 0.001 \cdot \theta_q$$

**What to say:**

> "MoCo separates 'how many negatives' from 'how big is my GPU batch.' The queue gives you 65,000 negatives. The momentum encoder keeps those keys fresh. This is the same EMA idea DINO will use later — but here it's for consistency of negatives, not for self-distillation."

**MoCo still has negatives** — it treats every other image as "different", even two photos of the same dog. That's a noisy signal. The next step removes negatives entirely.

---

### BYOL: No Negatives — and the Collapse Problem

BYOL (Grill et al., 2020) says: forget negatives. Just make two augmented views of the same image match each other.

```
View 1 ──▶ Online Encoder ──▶ Online Projector ──▶ Predictor ──▶ ─┐
                                                                    ↓ MSE loss
View 2 ──▶ Target Encoder ──▶ Target Projector ──────────────────▶ stop_grad
           (EMA of online, no gradient)
```

**The asymmetry is critical:** only the online branch has a `Predictor` MLP. The target has no predictor, no gradient. Remove either and it collapses.

#### Mode Collapse — The Core Problem of Non-Contrastive SSL

Without negatives, the easiest solution is to output the same vector for every input:

```
f(dog) = f(cat) = f(car) = [0, 0, 1, 0, 0, ...]
```

Loss = 0 instantly. But the encoder learned nothing. This is **mode collapse**.

**What collapse looks like in practice:**
- Loss drops to near-zero in the first few steps (suspiciously fast)
- Linear eval accuracy stays at ~10% (random for 10 classes)
- t-SNE: one dense blob, no class separation
- Embedding variance across the batch ≈ 0

**Why doesn't BYOL collapse?** The EMA target + predictor asymmetry creates an implicit regularization — the online network chases a moving target it can never fully reach. BatchNorm in the projector also helps by preventing constant-output solutions. The community debated this for a year after the paper came out.

**Pause and ask students:**

> "If you remove the predictor from BYOL — so both branches are symmetric — what happens? Try it. It collapses immediately."

---

## Part 2: DINO (Cells 10–18)

### The story

**What to say:**

> "DINO asks: what if we train a student network to match the output of a teacher network — where the teacher is a slowly-updated copy of the student? No labels. No negatives. Just self-distillation."

### Architecture — Teacher/Student

```
global crop 1 → [Student ViT] → softmax(z / τ_s) ─┐
global crop 2 → [Student ViT] → softmax(z / τ_s) ─┤
local crop 1  → [Student ViT] → softmax(z / τ_s) ─┤→ cross-entropy loss
local crop 2  → [Student ViT] → ...               ─┘
                                                    ↑
global crop 1 → [Teacher ViT] → softmax((z - c) / τ_t)  (no grad)
global crop 2 → [Teacher ViT] → softmax((z - c) / τ_t)

Teacher weights = EMA of Student weights  (no gradient, no backprop)
```

**Three things to explain:**

**1. EMA teacher (Exponential Moving Average)**

> "The teacher is not trained by backprop. It's a running average of the student weights. Think of it as a 'smoothed-out, slower-updating' version of the student. This gives stable targets — otherwise the student would be chasing a target that's changing just as fast."

```python
teacher = momentum * teacher + (1 - momentum) * student
# momentum = 0.996 → teacher updates very slowly
```

**2. Multi-crop strategy**

> "Teacher sees only global crops (large, full context). Student sees global AND local crops (small patches). The student has to predict what the teacher sees from a tiny local view — this forces it to understand global context from local information. Harder task → better features."

**3. Centering — the key trick preventing collapse**

**This is the most important concept to explain carefully:**

> "BYOL needed a predictor MLP + BatchNorm to prevent collapse — and nobody fully understood why for a year. DINO replaces all of that with one explicit, interpretable operation: centering."

**The collapse scenario without centering:**

```
teacher(dog) softmax  → [0.98, 0.01, 0.01, ...]   ← dim 0 always dominates
teacher(cat) softmax  → [0.97, 0.02, 0.01, ...]
teacher(car) softmax  → [0.99, 0.01, 0.00, ...]
→ student learns: always output [1, 0, 0, ...] → loss → 0, nothing learned
```

**With centering:**

$$z_{\text{teacher,corrected}} = z_{\text{teacher}} - c$$

$$c \leftarrow m \cdot c + (1 - m) \cdot \text{mean\_batch}(z_{\text{teacher}})$$

If dimension 0 dominates, `c[0]` grows to cancel it out. The teacher is forced to spread probability mass across all dimensions — the student must learn real structure to match.

```
(teacher(dog) - c) softmax  → [0.15, 0.12, 0.08, ...]   ← spread out
(teacher(cat) - c) softmax  → [0.08, 0.18, 0.11, ...]
→ student must model actual image differences
```

**Temperature asymmetry:**

| | Temperature | Effect |
|---|---|---|
| Teacher | τ = 0.04 | Sharp, confident target — forces strong signal |
| Student | τ = 0.1 | Softer — avoids gradient explosion |

> "The student must match a distribution that is *sharper than itself*. This creates a difficulty gradient that drives learning — the student can never fully catch the teacher."

**Pause and ask students:**

> "What happens if we set teacher temperature = student temperature = 0.1? The task becomes easier. What if τ_teacher → 0? The teacher becomes a one-hot — near-infinite gradients. DINO's asymmetric temperatures are carefully chosen."

---

### The "Wow" Moment: Attention Maps (Cell 17)

**Build up to this — don't just show the output.**

**What to say:**

> "DINO was trained with zero segmentation labels. Zero bounding box labels. We just told it: 'these crops came from the same image.' Now watch what its attention does."

> "In a Vision Transformer, the [CLS] token attends to every patch. After DINO training, the [CLS] token has learned to attend to the *semantically relevant* patches — the foreground object — and ignore the background."

> "Nobody told it what the foreground was. Nobody drew a single mask. It emerged purely from the SSL objective."

**This is why DINO is famous.** Visualization of this result in the original paper shocked the community — people didn't expect unsupervised segmentation to emerge from a classification-style pretext task.

**Pause and ask students:**

> "What does this tell us about what the model is actually learning? Is it memorizing pixel patterns, or learning something more abstract?"

---

## SimCLR vs DINO — Side by Side

| | SimCLR | DINO |
|---|---|---|
| **Negatives** | Yes (all other images in batch) | No |
| **Backbone** | ResNet (works poorly with ViT) | ViT-native |
| **Batch size** | Needs large (4096+) | Works with small batches |
| **Collapse prevention** | Negatives push representations apart | Centering + EMA teacher |
| **Key result** | Good linear eval | Unsupervised segmentation emerges |
| **Temperature** | Single τ on loss | Separate τ for teacher/student |

---

## t-SNE Comparison (Cells 20–21)

**What to say:**

> "t-SNE projects high-dimensional features into 2D. If SSL worked, images of the same class should cluster together — even though the model never saw class labels during training."

**What to look for:**
- Well-separated clusters → good representations
- Mixed clusters → model learned augmentation invariance but not semantic meaning
- SimCLR with ResNet vs DINO with ViT — expect DINO to show cleaner class separation

---

## Exercise Walkthrough Tips

**Ex1a (DINO centering ablation):** The most valuable exercise in the lab. Without centering, the teacher distribution collapses to one or two dominant dimensions. Students should see:
- Loss stops decreasing quickly (trivial solution found)
- `center.norm()` stays near zero (nothing to subtract)
- Attention maps go flat — uniform attention over all patches, no foreground bias
- Linear eval accuracy stays near random (~10%)

The center norm in normal training should **stabilize** — not grow unboundedly. Growing center norm means the centering update is chasing a collapsing distribution.

**Ex1b (Multi-crop ablation):** Without local crops, the student only sees the same views as the teacher — the task is easier and features are less transferable. Expect ~2-5% lower linear eval accuracy. Attention maps may look slightly less focused since the model never had to infer global context from local patches.

**Ex2 (MAE masking ablation):** Low masking (0.25) → reconstruction is easy (neighbors reveal content) → model learns texture interpolation, not semantics → worse linear eval despite lower reconstruction loss. High masking (0.75) → harder task → model must understand global structure → better features. The key insight: **reconstruction loss and representation quality are inversely correlated at low masking ratios**.

**Ex3 (Three-way comparison):** Students often expect MAE to win on linear eval since it's newer. On CIFAR-10 with 10 epochs, DINO often beats MAE — MAE needs much longer training (1600 epochs on ImageNet) to shine. The discussion question about medical segmentation should lead students to DINO: interpretable attention maps + strong spatial features + no need for pixel-level labels.

---

## The Big Picture — SSL Evolution

```
2020  SimCLR     "Contrastive + augmentations. Needs big batches + negatives."
        ↓
2020  BYOL       "No negatives — EMA teacher + predictor MLP. Mysterious why it works."
        ↓
2021  DINO       "No negatives + no predictor. Centering explains the mystery.
                  ViT attention maps → emergent segmentation. Zero labels."
        ↓
2022  MAE        "Mask 75% of patches, reconstruct them. Even simpler pretext task."
        ↓
2023  DINOv2     "DINO at scale (142M params, curated 142M images).
                  Best off-the-shelf vision features ever measured."
```

**Closing question for students:**

> "SSL removes the need for labels during *pretraining*. But linear evaluation still uses labels. Is there a world where we never use labels at all? What would that look like?"
>
> Follow-up: DINOv2 features are so good that k-NN classification (no training at all, just nearest neighbors in feature space) matches fine-tuned supervised models on many benchmarks. Labels may matter less than we thought.
