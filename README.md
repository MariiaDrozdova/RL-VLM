## Project Overview

**CirclesParity** probes the reasoning ability of Microsoft’s Florence-2 (base FT) on a simple visual task: determining whether the number of colorful balls in an image is odd or even.

- **Dataset**  
  – Synthetic images of colored circles, inspired by Apple’s Samy Bengio presentation. 
  – Train/Val/Test/Test-Hard splits: 10 000 / 100 / 1 000 / 1 000 samples  
  – Binary labels: “odd” if there’s an odd number of balls, “even” otherwise.

- **Method**  
  – **Supervised Fine-Tuning** of Florence-2 base FT via a new `CirclesQA` prefix  
  – **Direct Parity Classification** (best performer)  
  – Also explored:  
    1. Count-then-infer parity (unstable generalization)  
    2. Pairwise grouping into color-pairs + leftover
    
- **Results**  
  – **Test accuracy:** 91.6%  
  – **Test-Hard accuracy:** 56.2%  

