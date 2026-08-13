# Decision Trees and Ensembles

Tree-based classifiers compared on a large multi-class problem, then the same algorithm rebuilt from scratch.

Coursework for ICS485 (Machine Learning), KFUPM.

---

## `covertype-ensembles.ipynb` — four tree methods on half a million samples

Predicting forest cover type from cartographic variables on the [UCI Covertype](https://archive.ics.uci.edu/ml/datasets/covertype) dataset: 581,012 observations, 54 features, seven tree-cover classes from Spruce/Fir to Krummholz. Loaded directly through `sklearn.datasets.fetch_covtype()`, so nothing needs downloading by hand.

The split is deliberately lopsided: **49,500 train / 5,500 validation / 526,012 test**. Holding back 90% of the data for testing is unusual, but it means every test figure below is measured on more than half a million examples and carries almost no sampling noise. Each classifier is grid-searched on validation, then scored once on test.

| Classifier | Best validation | Test | Winning configuration |
|:---|:---|:---|:---|
| Decision tree | 0.8365 | 0.8337 | gini, max_depth 30, min_samples_split 2, min_samples_leaf 1 |
| **Bagging** | **0.9075** | **0.9047** | 60 estimators, 80% samples, 80% features, bootstrap **False** |
| AdaBoost | 0.6696 | 0.6744 | 100 estimators, learning rate 0.5, depth-1 base learner |
| Random Forest | 0.8880 | 0.8844 | 200 trees, max_depth 40, max_features sqrt |

Validation and test track each other within about 0.3 points for every model, which is what a 526,012-sample test set should give you.

### AdaBoost lands below a single decision tree

0.674 against the plain tree's 0.834 is a big gap, and it is worth being precise about the cause: the base learner is fixed at `max_depth=1`, and the grid search only varied `n_estimators` and `learning_rate`. It never varied the depth of the weak learner.

A depth-1 stump makes exactly one binary split. Boosting stumps is the textbook AdaBoost setup and works well on binary problems, but Covertype has seven classes and highly non-linear boundaries in 54 dimensions. Reaching 67% by combining learners that individually manage barely more than chance is actually a lot of lift from boosting — the ceiling was set by the choice of base learner, not by the algorithm. Allowing depth 3 to 5 stumps would be the obvious next experiment.

### Bagging beats Random Forest, which is the reverse of the usual result

0.9047 against 0.8844 is a real gap, and the winning hyper-parameters explain it. Random Forest is pinned to `max_features='sqrt'`, roughly 7 of the 54 features at each split. The best bagging model considers **80%** of them, about 43.

Random Forest's feature subsampling exists to decorrelate the trees so that averaging them cancels more variance. That trade only pays when the discarded features are largely redundant. On Covertype most of the 54 features carry genuine signal, so restricting each split to 7 of them costs more in individual tree quality than it recovers through decorrelation.

One detail worth noticing in that row: the best configuration uses **`bootstrap=False`**, meaning subsets are drawn without replacement. The winning model is therefore technically *pasting* rather than bootstrap aggregation.

### Bonus: Vehicle Silhouettes

A single tuned decision tree on the [Statlog Vehicle Silhouettes](https://archive.ics.uci.edu/ml/datasets/Statlog+%28Vehicle+Silhouettes%29) data (18 features, four classes: opel, saab, bus, van), split 70/15/15 with seed 777. Best validation 0.7717 with entropy at max_depth 10; **0.7559 on test**. The learned rules are extracted as text and the tree is rendered as an image.

Two things stand out. `entropy` clearly beats `gini` here — 0.7717 against 0.7087 at best — which is the reverse of Covertype, where gini won. On 846 samples the split criterion actually matters; on half a million it washes out.

The comparison across folders is the more interesting one. The from-scratch neural network in `neural-networks-from-scratch/` reaches **85.04%** on this same dataset and split, against this tuned decision tree's **75.59%**. A single tree partitions the feature space with axis-aligned cuts, and vehicle silhouettes are described by continuous shape ratios where the informative boundaries are oblique. The network can form those boundaries directly; the tree has to approximate them with staircases, and 846 samples is not enough data to build a good staircase.

---

## Data

- `Vehicles.csv` — Statlog Vehicle Silhouettes, 846 samples, 18 features, four vehicle types
- `lending-club-data.csv.gz` — Lending Club loan data, 68 columns, **gzipped**. `pandas.read_csv()` reads it directly with no extra arguments. Used by the from-scratch decision tree work in this folder.

Covertype is fetched by scikit-learn on first run and cached locally.

## Running this

```bash
pip install numpy pandas scikit-learn matplotlib
jupyter notebook covertype-ensembles.ipynb
```

Be aware that the grid searches in this notebook are genuinely slow — bagging fits 60 trees per configuration across 54 configurations on 49,500 samples. Expect hours for a full re-run, not minutes.
