# EN3160 – Assignment 3: Neural Networks

**Name:** Saruka U. | **Index:** 230585C

Implementation of three neural-network experiments in PyTorch: a dense network written from scratch, a comparison of two CNNs on MNIST, and transfer learning with ResNet18.

## Tasks and results

| Task | Description | Dataset | Result |
|---|---|---|---|
| 1 | Dense network (3072 → 100 sigmoid → 10) with manual forward pass, cross-entropy loss and backpropagation, 10 epochs | CIFAR-10 | Train 39.71 %, test 39.71 % |
| 2 | LeNet-5 vs. a modern 3-layer CNN (batch normalization + dropout), same conditions, 10 epochs | MNIST | LeNet-5: test 98.57 %. Modern CNN: test 99.31 % |
| 3 | ResNet18 (ImageNet-1K) as fine-tuning vs. fixed feature extractor | Hymenoptera (ants vs. bees) | Fine-tuning: val 94.12 %. Feature extraction: val 94.77 % |

### Key findings

- **Task 1:** Training and test accuracy match, so the network underfits. Flattening the image discards spatial structure, which limits a dense network.
- **Task 2:** The modern CNN learns faster, has about half the test loss and was ahead at every epoch (143 vs. 69 errors). It has about 5.4× more parameters and took about 2.2× longer to train. This is a single run, and the models differ in several ways at once.
- **Task 3:** The two methods differ by one image out of 153, so they are effectively tied. Feature extraction trains only 1,026 parameters instead of about 11.2 M and ran about 2× faster.

## Repository contents

```
├── EN3160_Assignment_3.ipynb   # full report: code, plots, confusion matrices, discussion
└── README.md
```

## Setup and usage

```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn jupyter
jupyter notebook EN3160_Assignment_3.ipynb
```

Run **Kernel → Restart & Run All**. CIFAR-10, MNIST and the Hymenoptera dataset are downloaded automatically into `./data`. The reported results were produced on CPU with a fixed random seed (42).

## What the notebook contains

- **Task 1:** manual softmax cross-entropy and sigmoid backpropagation, per-epoch loss and accuracy curves, and the per-iteration loss history.
- **Task 2:** per-epoch training and test loss and accuracy for both models, a side-by-side comparison, and confusion matrices on the test set.
- **Task 3:** fine-tuning and feature-extraction runs, plus a comparison table and curves.
- A short discussion after each task and overall conclusions.