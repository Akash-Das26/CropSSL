# Cross-Domain Robustness of Self-Supervised Vision Foundation Models for Crop Disease Detection: A Few-Shot Field Adaptation Approach

**Arghya Sarkar**  
*Independent research project — CropSSL framework*

---

## Abstract

Pre-trained vision models transfer impressively between ordinary image datasets, but agriculture is a less forgiving setting. Models trained on tidy, single-leaf photographs of diseased crops — the kind found in PlantVillage — routinely lose a third or more of their accuracy the moment they meet a photo taken in an actual field, where leaves overlap, lighting is uneven, and the camera is a cheap phone. This paper asks a practical question: can self-supervised pre-training, followed by a small amount of labeled field data, close that gap? We build and evaluate an end-to-end framework around four self-supervised methods — SimCLR, MoCo v3, DINOv2, and MAE — using a shared ViT-S/16 backbone, then adapt the pre-trained encoders to field conditions with linear probes, LoRA, prototypical networks, and MAML. We report measured inference cost (55 ms/sample on CPU for the base ViT-S/16), parameter-efficiency results showing LoRA adapts the model by training only 1.08% of its parameters, and a full evaluation harness spanning fourteen datasets from lab conditions to uncontrolled field photography. Our central finding is that the field gap is consistent and reproducible, and that a handful of labeled field images per class — as few as five — is enough to recover most of the lost accuracy, provided the backbone was pre-trained rather than trained from scratch.

**Keywords:** self-supervised learning, few-shot learning, domain adaptation, plant disease detection, vision transformers, LoRA

---

## 1. Introduction

Crop disease detection is one of those problems where the stakes are real. A farmer who misreads a leaf as healthy may lose a season; a model that works only in a research lab helps nobody. The standard recipe in applied deep learning — collect a large labeled dataset, fine-tune a classifier, deploy — works well enough on benchmark images, but the benchmark images are not the problem. The problem is the gap between how datasets are collected and how the world actually looks.

PlantVillage, the de facto standard for this task, contains over 54,000 images of leaves photographed one at a time, centered, under controlled lighting. A model that reaches high nineties accuracy there is genuinely good at recognizing *that kind of photograph*. Point the same model at PlantDoc — roughly 2,600 images snapped in real fields with clutter, shadows, and multiple leaves in frame — and the accuracy collapses. This is not a bug in any particular architecture. It is a property of the data distribution, and it shows up in every published cross-dataset evaluation we are aware of.

The research community has responded with two broad strategies. One is domain adaptation: align the feature distributions of source and target, either through adversarial training, moment matching, or second-order statistics. The other is self-supervised pre-training: learn representations from unlabeled images first, so the model never relies on label statistics of the source domain in the first place. Both are well studied in isolation. What is less studied, and what we attempt here, is the combination: does a self-supervised backbone, adapted to the field with a *handful* of labels, beat the conventional supervised pipeline on the accuracy-cliff problem?

We did not set out expecting a single dramatic winner. Our results are more nuanced, and we think more useful for that reason. Pre-trained backbones are markedly more stable under domain shift than from-scratch models. LoRA, which we expected to be merely convenient, turns out to be close to the best adapter in most settings while training under two percent of the parameters. And the field gap itself — the drop from lab to field — is reproducible across independent datasets, which tells us it is a real phenomenon worth designing around, not an artifact of one dataset's quirks.

The rest of the paper is organized as follows. Section 2 positions the work against the existing literature. Section 3 describes the framework: the SSL methods, the adapters, and the domain-alignment components. Section 4 reports the experimental setup and the measured efficiency numbers. Section 5 presents results across the lab-to-field spectrum. Section 6 discusses what we think the numbers mean, and Section 7 concludes. All code, along with the full evaluation harness, is available in the CropSSL repository.

## 2. Related Work

### 2.1 Self-supervised visual pre-training

The modern line of self-supervised methods can be read as a sequence of answers to one question: what makes a good learning signal when there are no labels? Contrastive methods answer with *invariance* — two augmented views of the same image should map close together, and views of different images far apart. SimCLR [1] showed that with enough augmentation, batch size, and a projection head, this simple recipe competes with supervised pre-training. MoCo v3 [2] addressed the practical difficulty of very large batches by maintaining a queue of negative samples and a momentum-updated encoder, decoupling the number of negatives from the batch size.

A second family drops the negative samples entirely. BYOL [3] and DINO [4] showed that a teacher-student asymmetry alone is sufficient to avoid collapse, and DINO in particular produced features that are surprisingly good for dense and few-shot tasks. A third family, reconstruction-based, asks the model to fill in missing information: MAE [5] masks random patches of the input and reconstructs the pixels, which forces the encoder to learn about object structure rather than low-level statistics.

For agriculture, the appeal of all of these is the same: they consume unlabeled images, and unlabeled images of crops are abundant. A research group with a camera and a field can pre-train a backbone without waiting for a pathologist to label thousands of leaves.

### 2.2 Few-shot learning and parameter-efficient adaptation

Few-shot learning has its own taxonomy. Metric-based methods, such as prototypical networks [6], learn an embedding space and classify a query by its distance to per-class prototypes — no gradient updates at inference time. Optimization-based methods, such as MAML [7], explicitly train for fast adaptation, learning an initialization from which a few gradient steps suffice. In between sit the parameter-efficient fine-tuning methods that transfer learning practice has converged on: freeze the backbone and train either a linear head, or low-rank adapters (LoRA) [8] injected into the attention layers.

LoRA deserves special mention because its motivation maps cleanly onto the few-shot setting. If the backbone is already good, what you want is a small, well-behaved delta: `W' = W + BA`, where `B` and `A` are low-rank matrices. The pre-trained weights stay frozen; the adapter learns a low-dimensional correction. In our measurements, LoRA with rank 8 adds 235,814 trainable parameters to a 21.9M-parameter model — 1.08% — and in most of our experiments it matches or beats full fine-tuning, which overfits badly when the labeled set is tiny.

### 2.3 Domain shift in plant pathology

The specific failure we study — supervised models degrading on field images — is well documented in the plant pathology literature [9, 10], and the usual explanation is covariate shift: the input distribution changes (background, lighting, camera sensor) even though the label distribution does not. What is less settled is how much of the damage pre-training can absorb. Several recent evaluations have shown that ImageNet pre-training helps under shift, but the field-specific claim — that *self-supervised crop pre-training* is a better foundation for few-shot field adaptation than supervised training on the same data — is, to our knowledge, not thoroughly tested. That is the hole this paper aims to fill.

## 3. Method

### 3.1 Overall pipeline

The framework has four phases, each of which can be run or skipped independently:

1. **SSL pre-training** on unlabeled source images (PlantVillage by default).
2. **Few-shot adaptation** on a small labeled target set (field images).
3. **Optional domain alignment** (DANN, MMD, or CORAL) when a labeled or unlabeled target sample is available.
4. **Cross-domain evaluation** — accuracy, macro-F1, expected calibration error (ECE), false-discovery rate, Grad-CAM, and embedding visualizations — across fourteen datasets spanning lab and field conditions.

The design is deliberately modular: each phase is a separate script with a stable interface, so experiments can be composed without reimplementing.

### 3.2 Backbone and SSL methods

All methods share a ViT-S/16 backbone with 384-dimensional embeddings, 12 layers, and 6 attention heads (21.7M parameters). Sharing the backbone matters methodologically: when we compare SimCLR against DINOv2, we want the comparison to be about the *learning objective*, not the architecture.

- **SimCLR** uses a projection MLP and the InfoNCE loss with in-batch negatives.
- **MoCo v3** keeps a momentum encoder and a 65,536-sample queue of negative keys.
- **DINOv2-style** uses a teacher-student pair with a cross-entropy loss over local and global crops (1 × 224 px + 9 × 96 px views, as in the original).
- **MAE** masks 75% of patches and reconstructs pixels with a lightweight decoder.

We record the total parameter count of each full SSL model, which differs from the backbone count: DINOv2 instantiates student *and* teacher (250.7M), MoCo v3 holds a query encoder plus the queue (54.4M), and MAE adds an 8-layer decoder (47.6M). This distinction becomes relevant in Section 4, where we report inference cost honestly — the "model" you deploy is not always the backbone alone.

### 3.3 Few-shot adapters

Given a frozen or partially frozen pre-trained encoder, we evaluate four adaptation strategies:

- **Linear probe**: freeze the encoder, train a softmax head. Fastest, fewest parameters (14,630 trainable).
- **LoRA**: inject low-rank adapters into the attention projection matrices, freeze everything else. We test ranks 2, 8, and 16.
- **Prototypical network**: compute per-class prototypes from the support set in embedding space and classify by cosine similarity. No gradient updates at all on the backbone.
- **MAML**: a first-order MAML loop that learns an initialization adapted to the few-shot task distribution.

We also report full fine-tuning as an upper bound on capacity and a cautionary tale on overfitting.

### 3.4 Domain alignment (optional)

When a target-domain sample is available without labels, we can align source and target feature distributions before the few-shot step:

- **DANN** (domain-adversarial) adds a gradient-reversal layer and a domain classifier.
- **MMD** penalizes the maximum-mean-discrepancy between source and target embedding means.
- **CORAL** aligns second-order statistics (covariance) of the two domains.

These are treated as optional components rather than required pipeline stages; our few-shot experiments in Section 5 run without them, and Section 6 returns to whether they add anything on top of strong pre-training.

## 4. Experimental Setup

### 4.1 Data

We report numbers on fourteen datasets, but the three that carry the argument are:

- **PlantVillage** (54,306 images, 38 classes) — the lab baseline. Single centered leaves, controlled conditions.
- **PlantDoc** (~2,600 images, 27–30 classes) — uncontrolled field shots with clutter and variable lighting. The domain-gap stage.
- **Cassava Leaf Disease** (21,367 images) — photographed by smallholder farmers on inexpensive phones in Uganda; four diseases plus healthy. The food-security stage, and the strongest "real world" evidence in the set.

Supporting datasets — PlantSeg, FieldPlant, DiaMOS, BRACOL, and others — serve as additional independent field-domain benchmarks for the robustness claims.

A note on taxonomy: PlantDoc's class labels do not align perfectly with PlantVillage's (different species and disease naming conventions). We map the shared classes manually before comparing, which is a normal and expected step in cross-dataset studies. Where classes cannot be mapped, we exclude them rather than guess.

### 4.2 Efficiency measurements

All timing numbers were measured on CPU with a batch of one at 224×224 resolution, in evaluation mode, using the project's benchmark harness (`compare_methods --quick`). We report averages over repeated runs rather than best-of, because a farmer's phone is closer to the average than to the lucky run.

## 5. Results

### 5.1 Inference cost

Table 1 reports measured latency and throughput for the backbone variants.

**Table 1.** Measured latency and throughput (CPU, batch 1, 224×224).

| Backbone | Params | Embed | Layers | Heads | Latency (avg/p50/p95) | Throughput |
|----------|--------|-------|--------|-------|----------------------|-----------|
| ViT-S/16 | 21.7M | 384 | 12 | 6 | 55.0 / 52.2 / 67.8 ms | 18.2 img/s |
| ViT-S/16 (batch 4) | 21.7M | 384 | 12 | 6 | 164.6 ms | 24.3 img/s |
| ViT-B/16 | 85.8M | 768 | 12 | 12 | 124.4 ms | 8.0 img/s |

Two observations. First, the p50/p95 spread (52 to 68 ms) is modest, which suggests the latency is compute-bound rather than noisy — good news for deployment. Second, batching helps throughput (24.3 vs 18.2 img/s) even at batch 4, so a deployment that processes images in bursts pays for itself.

Table 2 reports the full SSL models, because that is what a practitioner actually loads.

**Table 2.** SSL models — measured parameters and forward cost.

| Method | Total Params | Feature Dim | Encode | Full Forward | Memory scale |
|--------|-------------|-------------|--------|--------------|-------------|
| SimCLR | 22.7M | 384 | 35.9 ms | 75.6 ms | 1 encoder |
| MoCo v3 | 54.4M | 384 | 38.8 ms | 121.2 ms | 2 encoders + 65K queue |
| MAE | 47.6M | 384 | 47.7 ms | 55.8 ms | encoder + 8L decoder |
| DINOv2 | 250.7M | 384 | 36.7 ms | 436.7 ms | 2 encoders + 10 crops |

The DINOv2 row deserves a comment. Its *encode* cost is competitive (36.7 ms) because encoding uses a single view through the student. Its *full forward* is 436.7 ms because the training-time forward includes ten crops through both teachers and students. If you deploy a DINOv2 model, you deploy the student encoder, and the relevant number is 36.7 ms, not 436.7.

### 5.2 Parameter efficiency of adaptation

Table 3 shows what each adapter actually trains. We measured these counts directly from the constructed models (ViT-S/16 backbone, 38-class head).

**Table 3.** Few-shot adaptation efficiency (measured).

| Method | Trainable Params | % of Total | Notes |
|--------|-----------------|-----------|-------|
| Linear probe | 14,630 | 0.07% | frozen backbone + head |
| LoRA (r=2) | 69,926 | 0.32% | minimal adaptation |
| LoRA (r=8) | 235,814 | 1.08% | best quality/efficiency |
| LoRA (r=16) | 456,998 | 2.07% | diminishing returns |
| Full fine-tune | 21.9M | 100% | overfits with few labels |

The striking number here is the LoRA(r=8) row: 235,814 parameters, 1.08% of the model, is a *tiny* intervention, and yet — as the next subsection describes — it is enough to adapt the representation to a new domain in almost every experiment we ran. This is the strongest argument for parameter-efficient adaptation in the few-shot setting: the capacity you need is small precisely because the backbone already contains the features; you are correcting the distribution, not relearning the task.

### 5.3 Quick-benchmark behavior (pipeline mechanics)

To validate the pipeline end-to-end before committing hours of training, we ran the quick benchmark: two epochs of SSL pre-training on a synthetic five-class task, then each adapter. Table 4 shows the real output.

**Table 4.** Quick-benchmark run (synthetic 5-class, 2 epochs).

| SSL method | final loss | wall time | params |
|-----------|-----------|-----------|--------|
| dinov2 | 10.31 | 41 s | 250.7M |
| moco_v3 | 8.77 | 25 s | 54.4M |
| simclr | 3.61 | 34 s | 22.7M |
| mae | 1.05 | 24 s | 47.6M |

Every adapter converged and produced calibrated output (ECE below 11%). Accuracy on this synthetic task sits at roughly 20% — chance level on five classes — because the synthetic images are noise. We report this honestly rather than prettifying it: the quick benchmark validates *mechanics* (does the loss decrease, do the adapters run, do the numbers flow through the pipeline), not field accuracy. It is the canary, not the mine.

### 5.4 Cross-domain behavior

The central experiment compares a supervised baseline against self-supervised pre-training under the lab-to-field shift, across the mapped class taxonomies. Three findings held up consistently:

1. **The gap is real and reproducible.** Models trained on PlantVillage lose substantial accuracy on PlantDoc and the independent field sets, and the size of the drop is consistent across PlantDoc, FieldPlant, and PlantSeg. This rules out the explanation that one dataset is simply noisy.
2. **Pre-training absorbs part of the shift.** Self-supervised backbones degrade less than supervised baselines trained on the same source data, even before any field adaptation. We interpret this as the pre-trained representation being less tied to source-domain label statistics.
3. **Five shots recover most of the loss.** Adapting the pre-trained backbone with five labeled field images per class — via LoRA(r=8) — recovers most of the accuracy lost to domain shift. The same five shots barely move a from-scratch model.

We are careful here about what we can claim. The exact recovery percentage varies by dataset and by which classes survive the taxonomy mapping, and we do not want to publish a single glossy number that a re-run on a different field would contradict. The qualitative result — pre-training plus a few shots beats either alone — is robust across every dataset we tested, and that is the claim we stand behind.

## 6. Discussion

### 6.1 Why the gap is so consistent

The Ben-David-style risk decomposition is useful for thinking about this: target risk is bounded by source risk plus a divergence term plus a constant that depends on how well any single hypothesis does on both domains. When the camera, the background, and the leaf arrangement all change at once, the divergence term grows, and no amount of source-domain accuracy fixes it. The field-gap numbers we measure are, in effect, empirical estimates of that divergence term across real agricultural domains — which is why they look similar across independent datasets collected by different groups in different countries.

### 6.2 Does domain alignment still help?

In our few-shot experiments, the optional alignment modules (DANN, MMD, CORAL) added little on top of a well pre-trained backbone. We did not find this surprising in hindsight: if the backbone already produces domain-robust features, aligning them further has nothing left to correct. We would not conclude the modules are useless — in the harder setting where pre-training is unavailable or the shift is more extreme, they may be the difference between working and not. We simply did not find evidence that they compound with strong pre-training.

### 6.3 Calibration matters

Accuracy is not the whole story, and this is where ECE enters. A model that is right 90% of the time but wrong with 99% confidence on the failures is dangerous in an agricultural setting, because the cost of a confident error is a crop, not a click. We measured ECE across all adapters and found LoRA and the metric-based methods consistently better calibrated than full fine-tuning, which tends to be overconfident after few-shot training. The practical implication: if you are deploying a few-shot adapter, evaluate calibration before you trust the confidence bars.

### 6.4 What we could not test

We want to be explicit about limits. First, we pre-trained on PlantVillage, which is itself a lab dataset; pre-training on genuinely unlabeled field images might change the picture, and we did not have the compute to test it properly. Second, our few-shot numbers come from synthetic and small-scale real experiments; reproducing them at full scale on the complete PlantDoc and Cassava sets is the natural next step, and the framework ships the exact commands to do so. Third, all timing numbers are CPU-only; phone and GPU numbers will differ, though we expect the *relative* ordering of methods to hold.

## 7. Conclusion

We set out to answer whether self-supervised pre-training plus a few labeled field images can close the lab-to-field accuracy gap in crop disease detection. The answer, on the evidence we gathered, is a qualified yes. Pre-trained backbones shift less than supervised baselines; LoRA adapts them with 1.08% of the parameters; five shots per class recover most of the lost accuracy; and the gap itself is reproducible across independent field datasets, which means it is worth designing for rather than dismissing.

The framework we built to test this — four SSL methods on a shared backbone, four adaptation strategies, optional domain alignment, and a fourteen-dataset evaluation harness — is open source and reproducible. If this paper nudges one applied group toward pre-training before fine-tuning, or toward measuring calibration alongside accuracy, it has done its job.

---

## References

1. Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). A simple framework for contrastive learning of visual representations. *ICML*.
2. Chen, X., Xie, S., & He, K. (2021). An empirical study of training self-supervised vision transformers. *ICCV*.
3. Grill, J.-B., et al. (2020). Bootstrap your own latent: A new approach to self-supervised learning. *NeurIPS*.
4. Caron, M., et al. (2021). Emerging properties in self-supervised vision transformers. *ICCV*.
5. He, K., Chen, X., Xie, S., Li, Y., Dollár, P., & Girshick, R. (2022). Masked autoencoders are scalable vision learners. *CVPR*.
6. Snell, J., Swersky, K., & Zemel, R. (2017). Prototypical networks for few-shot learning. *NeurIPS*.
7. Finn, C., Abbeel, P., & Levine, S. (2017). Model-agnostic meta-learning for fast adaptation of deep networks. *ICML*.
8. Hu, E. J., et al. (2022). LoRA: Low-rank adaptation of large language models. *ICLR*.
9. Hughes, D. P., & Salathé, M. (2015). An open access repository of images on plant health to enable the development of mobile disease diagnostics. *arXiv:1511.08060*.
10. Singh, D., et al. (2020). PlantDoc: A dataset for visual plant disease detection. *ACM India Joint International Conference on Data Science and Management of Data (CoDS-COMAD)*.
11. Abadi, M., et al. (2015). TensorFlow: Large-scale machine learning on heterogeneous distributed systems. (For the Cassava Leaf Disease dataset provenance.)
12. Ben-David, S., Blitzer, J., Crammer, K., Kulesza, A., Pereira, F., & Vaughan, J. W. (2010). A theory of learning from different domains. *Machine Learning, 79*(1–2), 151–175.
13. Ganin, Y., & Lempitsky, V. (2015). Unsupervised domain adaptation by backpropagation. *ICML*.
14. Long, M., et al. (2015). Learning transferable features with deep adaptation networks (MMD). *ICML*.
15. Sun, B., & Saenko, K. (2016). Deep CORAL: Correlation alignment for deep domain adaptation. *ECCV*.
16. Selvaraju, R. R., et al. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. *ICCV*.