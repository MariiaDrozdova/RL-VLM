## Project Overview

**CirclesParity** probes the reasoning ability of Microsoft’s Florence-2 (base-FT) on a simple visual task: determining whether the number of colorful balls in an image is odd or even.  As an exercise, we implemented both **Supervised Fine-Tuning** and several **Reinforcement Learning** techniques on this Vision-Language Model.

- **Dataset**  
  – Synthetic images of colored circles, inspired by Apple’s Samy Bengio talk.  
  – Train / Val / Test / Test-Hard splits: 10 000 / 100 / 1 000 / 1 000 samples  
  – Binary labels: “odd” if there’s an odd number of balls, “even” otherwise.

- **Methods**  
  1. **Supervised Fine-Tuning (SFT)**  
     – Direct parity classification via a `CirclesQA` prompt.  
     – Count-then-infer heuristic (unstable generalization).  
  2. **Reinforcement Learning**  
     – **REINFORCE** (policy gradient with learned baseline)  
     – **PPO** (Proximal Policy Optimization actor-critic with clipping)  
     – **GRPO** (Group-based Relative Policy Optimization)  
  3. **Heuristic Baselines**  
     – Pairwise grouping into color-pairs + leftover count

- **Results (Accuracies)**  

  | Method                       | Test Acc (%) | Test-Hard Acc (%) |
  |------------------------------|-------------:|------------------:|
  | SFT (Direct classification)  |          XX  |               XX  |
  | SFT (Count-then-infer)       |          XX  |               XX  |
  | RL – REINFORCE               |          XX  |               XX  |
  | RL – PPO                     |          XX  |               XX  |
  | RL – GRPO                    |          XX  |               XX  |


First, we SFT fine-tuned the model for 100 epochs with a batch size of 20 on 5,000 image-text pairs using a learning rate of 1e-5.
