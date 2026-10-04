# LoRA Training Research Candidates

Research summary dated October 2, 2026.

This document summarizes papers and methods that could extend the local AI Toolkit fork at shodan:~/ai-toolkit-qwen. It covers new training objectives, special-purpose adapters, alternative architectures, and improvements to training efficiency.

The strongest candidates for new capabilities are T-LoRA, SliderSpace, Direct Consistency Optimization, ZipLoRA, and Diffusion-KTO. Style-Friendly SNR sampling, LoRA+, and rsLoRA are inexpensive experiments to consider first. Larger projects include reward-based training, adaptation of fast models, step distillation, and fully 4-bit training.

Paper descriptions below are sourced research findings. Priorities, estimated integration effort, and proposed applications are engineering judgments based on the inspected fork, not measured results. Findings on SDXL, FLUX, or language models do not establish the same benefits on Qwen. No candidate was implemented or benchmarked as part of this research.

## Existing Toolkit Capabilities

The inspected checkout was at commit 9c927dd7, with additional uncommitted changes. Its six UI training workflows are:

- Ordinary LoRA Trainer, including image generation and supported editing tasks.
- Concept Slider.
- Fizgig Image Slider for Qwen Image 2.1.
- Fizgig Prompt Slider for Qwen Image 2.1.
- Flow-DPO LoRA for Qwen Image 2.1.
- Guidance Distillation LoRA for Qwen Image 2.1, currently present as an uncommitted local addition.

Ordinary training exposes LoRA, DoRA, and LoKr. Fizgig sliders support LoRA or signed DoRA. Flow-DPO and guidance distillation currently require LoRA.

Existing features include text and latent caching, quantized training, content/style timestep selection, layer learning-rate multipliers, output-preservation losses, wavelet loss, and pixel-frequency losses. The fork also contains LyCORIS integration and older YAML-only slider and concept-replacement trainers.

Guidance distillation targets CFG-1 inference and includes full-guidance and negative-contribution-only objectives. It does not distill the number of sampling steps.

## Methods That Add Training Capabilities

### T-LoRA for Training from Very Few Images

[T-LoRA: Single Image Diffusion Model Customization Without Overfitting](https://arxiv.org/abs/2507.05964) introduces timestep-dependent adapter rank and orthogonal initialization. The aim is to improve subject fidelity and prompt alignment when personalization data is extremely limited, including a single image. The paper demonstrates results on SDXL and FLUX.1-dev.

Potential use: learn a character or object with less copying of the reference pose, background, and composition.

Engineering assessment: high priority for small datasets, with moderate-to-large integration effort. The [official implementation](https://github.com/ControlGenAI/T-LoRA) requires timestep hooks during inference. Training support should therefore include a plan for ComfyUI sampling and loading, rather than assuming an ordinary LoRA loader reproduces the complete method.

### SliderSpace for Automatic Discovery of Controls

[SliderSpace: Decomposing the Visual Capabilities of Diffusion Models](https://arxiv.org/abs/2502.01639) discovers multiple interpretable visual directions from a concept prompt and trains an adapter for each direction. Its applications include concept decomposition, style exploration, and diversity enhancement. [Official code](https://github.com/rohitgandikota/sliderspace) includes SDXL and FLUX training.

Potential use: ask the model to explore “spaceships” or “illustrations,” then browse and combine discovered sliders without specifying every attribute in advance.

Engineering assessment: the most interesting creative addition. Existing sliders and sampling infrastructure provide a foundation, but discovery adds generated samples, semantic feature extraction, decomposition, and a distinct training objective. Moderate-to-large effort.

### Direct Consistency Optimization for Better Personalization

[Direct Consistency Optimization for Robust Customization of Text-to-Image Diffusion Models](https://arxiv.org/abs/2402.12004), or DCO, compares the adapted and frozen base models on training examples. It aims to retain pretrained knowledge while learning a subject or style, improving prompt alignment and the ability to compose independently customized concepts.

Potential use: improve a character or style LoRA's behavior under unfamiliar prompts and when combined with other adapters.

Engineering assessment: high practical priority and moderate effort. The fork already supports adapter-disabled reference passes. DCO could be a new objective that retains ordinary adapter inference, subject to a correct adaptation for Qwen's flow-matching formulation. Existing preservation losses should be comparison baselines.

### ZipLoRA for Learned Subject and Style Merging

[ZipLoRA: Any Subject in Any Style by Effectively Merging LoRAs](https://arxiv.org/abs/2311.13600) optimizes mixing coefficients for independently trained subject and style adapters. Its objective preserves their individual behavior while reducing interference. The paper's experiments primarily use SDXL.

Potential use: combine a character LoRA with a style LoRA when manual strength adjustments consistently sacrifice one or the other.

Engineering assessment: high priority if adapter composition is frequent. Implement as a separate optimization-based merging workflow. Moderate effort; large Qwen matrices, memory use, and compact export need investigation.

### B-LoRA for Separating Content and Style

[Implicit Style-Content Separation using B-LoRA](https://arxiv.org/abs/2403.14572) jointly trains two selected SDXL blocks from a single image. The resulting components can be used separately for content and style, supporting stylization and mixing. [Official code](https://github.com/yardenfren1996/B-LoRA) is available.

Potential use: extract the appearance of a reference illustration while avoiding unnecessary transfer of its subject or composition.

Engineering assessment: attractive and relatively straightforward to reproduce on SDXL. A Qwen version requires experiments identifying blocks with comparable roles. SDXL's block selection does not provide a validated Qwen mapping. T-LoRA addresses limited-data personalization; B-LoRA addresses content/style separation.

### Diffusion-KTO for Learning from Likes and Dislikes

[Aligning Diffusion Models by Optimizing Human Utility](https://arxiv.org/abs/2404.04465) introduces Diffusion-KTO. It uses binary feedback on individual images, without matched preference pairs or a separate reward model.

Potential use: train from a collection of individually liked and disliked generations with their prompts. This could reduce the data-collection burden of preference training.

Engineering assessment: a useful extension beyond the current paired Flow-DPO workflow. Moderate effort, including validation of the objective's adaptation to flow matching. Labels alone do not remove the need for correctly associated prompts and training data.

Implementation status: added to the cross-model implementation plan as a new
LoRA-only workflow for Qwen Image 2.1, Krea 2, Anima and Ideogram 4. Experimental
objective-math and pooled-reference fixtures are in progress; a usable trainer/form
is not implemented or selectable yet. The flow adaptation requires per-model
validation before enabling it.

### Style-Friendly SNR Sampling

[Style-Friendly SNR Sampler for Style-Driven Generation](https://arxiv.org/abs/2411.14793) concentrates training on noise levels where stylistic features emerge, using an explicit signal-to-noise-ratio distribution. The work includes modern backbones such as FLUX and SD3.5.

Potential use: improve capture of distinctive illustration, painting, or cartoon styles.

Engineering assessment: one of the cheapest worthwhile experiments. The fork's current Style option uses cubic timestep sampling; compare that behavior with the paper's log-SNR distribution. This should retain ordinary adapter files and inference.

## Reward Training and Targeted Suppression

### AlignProp and DRaFT

[Aligning Text-to-Image Diffusion Models with Reward Backpropagation](https://arxiv.org/abs/2310.03739) introduces AlignProp. [Directly Fine-Tuning Diffusion Models on Differentiable Rewards](https://arxiv.org/abs/2309.17400) introduces DRaFT. These methods optimize generated images against differentiable scores by backpropagating through sampling. Demonstrated objectives include aesthetics, image/text alignment, compressibility, and object-count control.

DRaFT-K truncates backpropagation to the last K sampling steps; DRaFT-LV targets lower-variance gradients when K equals one.

Engineering assessment: substantial training-loop work. Truncated variants are particularly worth evaluating under limited VRAM.

Proposed applications, not established results for this fork: palette restrictions, clean backgrounds, and consistent framing. Success depends on constructing a differentiable score that captures the desired behavior without rewarding undesirable shortcuts.

### Flow-GRPO

[Flow-GRPO: Training Flow Matching Models via Online RL](https://arxiv.org/abs/2505.05470) generates groups of candidates, scores them, and uses reinforcement learning to improve flow models. The paper demonstrates gains in composition, visual text rendering, and preference alignment.

Potential use: optimize against OCR correctness, object counts, or other automated checks that do not provide differentiable gradients.

Engineering assessment: high capability and high compute cost. Candidate generation becomes part of training, requiring rollout, scoring, and policy-update infrastructure. Existing offline Flow-DPO does not supply a complete online RL loop.

### SPM for Concept Erasure

[One-Dimensional Adapter to Rule Them All: Concepts, Diffusion Models and Erasing Applications](https://arxiv.org/abs/2312.16145) introduces SemiPermeable Membrane adapters, or SPMs. It combines targeted suppression with latent anchoring to protect unrelated behavior and an inference mechanism that regulates adapter effects according to the prompt.

Background: [Erasing Concepts from Diffusion Models](https://arxiv.org/abs/2303.07345) established a negative-guidance teacher approach to targeted concept removal.

Engineering assessment: lower priority unless suppression is a recurring requirement. Existing sliders already cover some related uses; the additional value would be explicit preservation and selective activation. Suppression should be evaluated on target and unrelated concepts, rather than assumed complete.

## Alternative Architectures and Training Improvements

| Method and source | Change | Engineering assessment |
| --- | --- | --- |
| **LoHa**, from [the LyCORIS paper](https://arxiv.org/abs/2309.14859) | Forms updates through an elementwise product of two low-rank matrix products, permitting a richer effective update rank. | First additional architecture to consider. It has diffusion-specific evidence and the fork already contains LyCORIS plumbing. Quantization, memory handling, checkpoint format, and sampling support still need validation. |
| **[OFT](https://arxiv.org/abs/2306.07280)** and **[BOFT](https://arxiv.org/abs/2311.06243)** | Adapt weights through orthogonal transformations; BOFT uses butterfly factorization for parameter efficiency. | A substantially different mechanism with diffusion experiments. Worth comparing for fidelity and retention of base capabilities. Larger integration, including loader and export handling. |
| **[LoRA+](https://arxiv.org/abs/2402.12354)** | Uses different learning rates for the two LoRA factors. | Cheap experiment through optimizer parameter groups; retains ordinary LoRA inference. Confirm gains with the fork's models and optimizers. |
| **[rsLoRA](https://arxiv.org/abs/2312.03732)** | Uses square-root-of-rank scaling to address weakened learning at higher ranks. | Cheap experiment. Existing alpha controls may express much of the behavior as a preset; preserve the intended strength during export and loading. |
| **[AdaLoRA](https://arxiv.org/abs/2303.10512)** | Allocates parameter budget among weight updates according to importance. | Useful for spending a fixed rank budget where it helps. Existing layer analysis can aid evaluation, but adaptive allocation needs separate training machinery. Strongest original evidence is from language tasks. |
| **[DyLoRA](https://arxiv.org/abs/2210.07558)** | Trains adapters to operate across a range of ranks. | Useful for producing multiple size/quality options from one run. Lower priority than methods adding image-training capabilities; original evidence is largely from language tasks. |
| **[PiSSA](https://arxiv.org/abs/2404.02948)** and **[LoRA-GA](https://arxiv.org/abs/2407.05000)** | Initialize from important base-weight components or estimated full-training gradients. | Potential convergence improvements. Require careful base/adapter accounting and export conversion; quantized weight access adds complexity. Treat image-model benefits as experiments rather than established gains. |

A new architecture should be evaluated together with its inference path. The ability to save a checkpoint is insufficient evidence that ComfyUI applies its intended behavior.

## Larger Projects

### D-OPSD for Personalizing Fast Models

[D-OPSD: On-Policy Self-Distillation for Continuously Tuning Step-Distilled Diffusion Models](https://arxiv.org/abs/2605.05204), published in 2026, addresses the loss of few-step capability during ordinary fine-tuning. The student trains on its own generated trajectories, using a teacher conditioned on the prompt and target image. Target models include Z-Image-Turbo and FLUX.2-klein.

Engineering assessment: investigate when the goal is to personalize a fast model while preserving its speed. Substantial effort. Applicability depends on the encoder and multimodal conditioning path; Qwen support should not be assumed from the title alone.

### LCM-LoRA for Reducing Sampling Steps

[LCM-LoRA: A Universal Stable-Diffusion Acceleration Module](https://arxiv.org/abs/2311.05556) demonstrates low-rank consistency distillation for accelerating Stable Diffusion generation.

Engineering assessment: a step-distillation trainer would add a capability beyond the fork's current CFG distillation. Selecting and validating the formulation and sampler for Qwen is substantial work. Existing Lightning or acceleration adapters provide useful comparison baselines.

### FourTune for Fully 4-Bit Training

[FourTune: Towards Fully 4-Bit Efficient Post-Training for Diffusion Models](https://hanlab.mit.edu/projects/fourtune), published in 2026, targets 4-bit weights, activations, and gradients. It uses a stabilizing branch and custom kernels.

The authors report 2.25 times lower memory use and 2.27 times higher training throughput than BF16 LoRA on FLUX.1-dev. These are their benchmark results, not predictions for Shodan or comparisons with this fork's already optimized quantized training.

Engineering assessment: highly relevant to the fork's memory-efficiency goals, but a backend project. Hardware support, kernels, model compatibility, and reproducibility need investigation.

### LayerDiffuse for Transparent Images and Layers

[Transparent Image Layer Diffusion using Latent Transparency](https://arxiv.org/abs/2402.17113) enables native transparent-image and layer generation through transparency-aware latent representations and diffusion-model adaptation.

Potential use: generate assets with real alpha channels or separately usable image layers.

Engineering assessment: substantial scope beyond an ordinary LoRA objective. Requires transparency-aware encoding/decoding, appropriate data, and compatible inference. A promising asset-generation project if transparency is the main goal.

## Recommended Implementation Order

1. **Run inexpensive comparisons:** Style-Friendly SNR sampling, LoRA+, and rsLoRA.
2. **Add a practical objective:** DCO, compared with ordinary training and existing preservation losses.
3. **Choose a capability based on need:** T-LoRA for tiny datasets; SliderSpace for creative exploration.
4. **Improve adapter workflows:** ZipLoRA when subject/style merging is difficult; Diffusion-KTO when preference-pair collection is difficult.
5. **Compare an architecture:** LoHa first, then OFT or BOFT if ordinary adaptation remains limiting.
6. **Reserve focused projects:** reward training, D-OPSD, step distillation, FourTune, or LayerDiffuse, each with its own evaluation budget.

For each experiment, compare against the current fork using fixed prompts and seeds, comparable data and training budgets, and the same inference settings where the method permits. Measure the intended capability alongside prompt adherence, subject/style fidelity, diversity, peak VRAM, wall-clock time, and checkpoint compatibility.

The next implementation should follow a concrete need. The presence of a paper establishes a method to investigate; it does not establish that the method improves this fork's models or hardware.
