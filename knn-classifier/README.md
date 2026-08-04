# KNN Classifier

Two notebooks on k-nearest neighbours, approached from opposite directions. One uses the library and studies the cost of the search; the other implements the algorithm from scratch and studies the distance metric.

Coursework for ICS485 (Machine Learning), KFUPM.

---

## `knn-mnist-sklearn.ipynb` — search structures and their cost

Handwritten digit recognition on **MNIST** (60,000 train / 10,000 test, 28×28 greyscale) with scikit-learn's `KNeighborsClassifier`. k is selected on a held-out validation split, never on the test set.

The interesting question here isn't accuracy, it's cost. The notebook compares the three search algorithms available to `KNeighborsClassifier` — `brute`, `kd_tree` and `ball_tree` — across three datasets of very different shape, recording evaluation CPU time alongside accuracy.

| Dataset | Dimensions | Best k | Test accuracy | brute | kd_tree | ball_tree |
|:---|:---|:---|:---|:---|:---|:---|
| MNIST | 784 | 1 | 96.91% | **9.4 s** | 46.7 s | 46.4 s |
| Abalone19 | 8 | 3 | 97.25% | **0.003 s** | 0.015 s | 0.014 s |
| Statlog / Satimage | 36 | 7 | 89.42% | **0.007 s** | 0.015 s | 0.014 s |

All three algorithms return identical predictions, as they must — they're exact methods differing only in how they find neighbours. But on MNIST the tree structures are roughly **five times slower** than brute force.

That inversion is the point. At 784 dimensions a kd-tree can no longer prune effectively: bounding regions overlap so heavily that nearly every branch gets visited anyway, so you pay traversal overhead on top of the distance computations you were trying to skip. Brute force, meanwhile, is one vectorised matrix operation. The trees also lose on the two low-dimensional sets, but narrowly and for a different reason — those problems are small enough that their asymptotic advantage never gets a chance to show.

The practical reading: tree-based neighbour search is worth reaching for in low dimensions with many points. In high dimensions it isn't, and you want approximate methods or dimensionality reduction instead. A closing section tests that idea directly, swapping raw pixels for hand-crafted features (mean intensity, symmetry, and similar): accuracy drops to 85.07% at k=5, but in a drastically smaller feature space.

## `knn-breast-cancer-numpy.ipynb` — the algorithm, written out

The **Breast Cancer Wisconsin** dataset (569 samples, 30 real-valued features, 212 malignant / 357 benign), with KNN implemented directly in numpy. No sklearn classifier is used.

Implemented from scratch: `NN_L2`, `KNN_L2`, `NN_L1`, `KNN_L1`, a unified `KNN` that switches on the distance metric, and a `confusion` matrix built by hand. The data is split 70 / 15 / 15 into train, validation and test, then k and the metric are grid-searched on the validation set.

| | Validation error |
|:---|:---|
| 1-NN, L2 | 5.88% |
| KNN, L2 (k=3) | **3.53%** |
| 1-NN, L1 | 5.88% |
| KNN, L1 (k=3) | **3.53%** |

Selected configuration: **L2 with k=3**, giving **5.81% test error**.

Two things stand out. L1 and L2 are indistinguishable here, which is what you'd expect on 30 continuous features of broadly comparable character — the metric matters far less than people assume once the geometry is well behaved. And going from 1-NN to 3-NN cuts the validation error by about a third, so most of the available gain came from the voting, not from the distance function.

An extra section adds distance-weighted voting (inverse distance) and an L∞ metric. Weighted L1 at k=7 ties the best plain result at 96.47% validation accuracy; L∞ does slightly worse at 95.29%. Neither beats plain 3-NN on this data.

All distance computations are vectorised, with no explicit loops over test points.

---

## Running these

```bash
pip install numpy pandas scikit-learn matplotlib
jupyter notebook
```

`186_satimage.csv` and `Abalone19.csv` are included and are read from the notebook's directory. The MNIST archives are **not** committed — the loader downloads them from a mirror on first run and caches them locally.
