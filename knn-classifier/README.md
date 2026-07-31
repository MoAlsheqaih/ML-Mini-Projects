# KNN Classifier

Handwritten digit recognition with k-nearest neighbours, using scikit-learn's `KNeighborsClassifier`. The interesting question here is not accuracy but cost: how the underlying nearest-neighbour search structure behaves as dimensionality changes.

Coursework for ICS485 (Machine Learning), KFUPM.

---

## What's in the notebook

`knn-mnist-sklearn.ipynb` loads **MNIST** (60,000 train / 10,000 test, 28×28 greyscale), selects k on a held-out validation split rather than the test set, then evaluates the final model. It then compares the three search algorithms available to `KNeighborsClassifier` — `brute`, `kd_tree` and `ball_tree` — across three datasets of very different shape, recording evaluation CPU time alongside accuracy.

| Dataset | Dimensions | Best k | Test accuracy | brute | kd_tree | ball_tree |
|:---|:---|:---|:---|:---|:---|:---|
| MNIST | 784 | 1 | 96.91% | **9.4 s** | 46.7 s | 46.4 s |
| Abalone19 | 8 | 3 | 97.25% | **0.003 s** | 0.015 s | 0.014 s |
| Statlog / Satimage | 36 | 7 | 89.42% | **0.007 s** | 0.015 s | 0.014 s |

## The result worth noting

All three algorithms return identical predictions, as they must — they are exact methods that differ only in how they find neighbours. But on MNIST the tree structures are roughly **five times slower** than brute force.

That inversion is the point of the exercise. At 784 dimensions a kd-tree can no longer prune effectively: the bounding regions overlap so heavily that nearly every branch has to be visited anyway, so you pay traversal overhead on top of the distance computations you were trying to skip. Brute force, meanwhile, is a single vectorised matrix operation. The trees also lose on the two low-dimensional sets, but only narrowly and for a different reason — those problems are small enough that their asymptotic advantage never gets a chance to show.

The practical reading: tree-based neighbour search is worth reaching for in low dimensions with many points. In high dimensions, it isn't, and you want approximate methods or dimensionality reduction instead.

A closing section tests that idea directly, replacing raw pixels with hand-crafted features (mean intensity, symmetry, and similar). Accuracy drops to 85.07% at k=5, but in a drastically smaller feature space.

---

## Running it

```bash
pip install numpy pandas scikit-learn matplotlib
jupyter notebook knn-mnist-sklearn.ipynb
```

`186_satimage.csv` and `Abalone19.csv` are included and are read from the notebook's directory. The MNIST archives are **not** committed — the loader downloads them from a mirror on first run and caches them locally.
