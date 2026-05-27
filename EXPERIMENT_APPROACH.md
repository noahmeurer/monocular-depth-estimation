# Experiment Approach

This project studies how large monocular depth models can be used to improve a smaller single-view depth model. The broader motivation is that existing large models may perform well out-of-the-box, but smaller models are often the ones that matter in practice when inference has to run on limited hardware, edge devices, or latency-constrained systems. Our experiments explore how large models can be used to improve less-capable smaller models through teacher-student distillation.

The experiments are structured around four baselines:

1. **Zero-shot student evaluation**  
   Evaluate a monocular student model directly on the test set without any task-specific training. This establishes the starting point.

2. **Fine-tuning on ground truth labels**  
   Fine-tune the student model on the provided training labels, preferably with a conservative setup such as updating only the prediction head. The goal is to adapt to the dataset without destroying the model's pretrained depth priors.

3. **Teacher-generated pseudo-labels**  
   Use a stronger teacher model to generate cleaner pseudo-labels for the training images, then fine-tune the student on those labels instead of the noisy ground truth. This tests whether teacher-student distillation can transfer useful structure into a smaller model.

4. **Novel view synthesis as training augmentation**  
   Use the teacher model to create pseudo-labeled augmentations through small synthetic viewpoint shifts, then fine-tune the student on the expanded dataset. The goal is to see whether this can distill some scene-level invariance into a single-view model, similar in spirit to classical data augmentation (e.g. crops, rotations, color jittering, etc.) for discriminative vision models.

The intended story is not to beat the zero-shot performance of large teacher models. Instead, the main question is whether a compact student can be improved through careful supervision from larger models, especially when direct training labels are noisy and when deployment constraints make smaller models more useful.
