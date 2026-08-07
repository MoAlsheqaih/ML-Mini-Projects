# Logistic Regression and SVM

Classification with linear models and kernels, starting from a binary problem that a straight line cannot solve. Two notebooks reach nearly the same accuracy by opposite routes: one builds polynomial features explicitly, the other lets a kernel do it implicitly.

Coursework for ICS485 (Machine Learning), KFUPM.

---

## `synthetic-binary-logistic-regression.ipynb` — bending a linear boundary

A synthetic two-feature binary dataset (550 samples, roughly 45/55 class split). Plotting it first makes the difficulty obvious: the two classes are not linearly separable, so plain logistic regression has nowhere useful to draw its line.

The data is split 80-20 into train and test, then 80-20 again into train and validation, giving 352 / 88 / 110. Every split is stratified, so all three sets carry the same class balance, and `random_state=20211008` keeps the whole notebook reproducible.

The fix is to expand the features into polynomial terms. A model that is linear in its *parameters* can still produce a curved boundary in the original feature space, as long as the features themselves are non-linear. Sweeping the degree and selecting on validation accuracy:

| Degree | Train acc | Validation acc |
|:---|:---|:---|
| 1 | 0.58 | 0.56 |
| 2 | 0.53 | 0.51 |
| 3 | 0.65 | 0.62 |
| 4 | 0.83 | 0.80 |
| **5** | **0.83** | **0.81** |
| 6 | 0.82 | 0.81 |
| 8 | 0.83 | 0.81 |
| 9 | 0.83 | 0.81 |
| 10 | 0.83 | 0.81 |

Selected degree 5, giving **0.85 test accuracy** after refitting on train and validation combined.

Two things are worth reading off that table. Degrees 1 and 2 are near-useless — degree 2 actually scores *below* the 55% you'd get by always predicting the majority class, so the model is worse than a constant. And everything from degree 5 onward is flat: degree 10 has vastly more parameters than degree 5 and does not convert any of them into accuracy. The capacity the problem needed was reached at 5, and the validation curve says so plainly, which is exactly what a validation set is for.

Each polynomial expansion is standard-scaled with a scaler fit on the training split only, which matters more than it sounds — raw degree-10 terms span many orders of magnitude, and without scaling the optimizer struggles.

The notebook closes by plotting the learned decision boundary over the test points.


## `synthetic-binary-svm.ipynb` — letting the kernel do the expansion

The same dataset and the same splits, approached differently. Instead of building polynomial features by hand and feeding them to a linear model, an SVM can reach the same higher-dimensional space through its kernel, without ever constructing the features. Comparing kernels on the validation set:

| Kernel | Train acc | Validation acc |
|:---|:---|:---|
| Linear | 0.5455 | 0.5455 |
| Polynomial, degree 2 | 0.5824 | 0.5682 |
| Polynomial, degree 3 | 0.5994 | 0.5682 |
| Polynomial, degree 4 | 0.7131 | 0.6932 |
| Polynomial, degree 5 | 0.5938 | 0.5568 |
| Polynomial, degree 6 | 0.7045 | 0.6591 |
| **RBF** | **0.8466** | **0.8182** |

Selected the RBF kernel, giving **0.8364 test accuracy** after refitting on train and validation combined.

The linear kernel lands at 0.5455, essentially the majority-class baseline — the same verdict the logistic regression notebook reached at degree 1, arrived at from the other direction. The polynomial kernels are erratic rather than monotone: degree 4 does reasonably at 0.6932 while degree 5 collapses back to 0.5568, which is what happens when the kernel's implicit feature space is a poor match for the geometry and the fit becomes unstable.

RBF wins comfortably, and that makes sense for this data. Its implicit feature space is infinite-dimensional and its similarity measure is local, so it can wrap a boundary around clusters instead of trying to fit one global polynomial surface through them.

Set the two notebooks side by side and the comparison is the real result: degree-5 polynomial logistic regression reaches 0.85 test accuracy, the RBF SVM reaches 0.8364. Practically a tie — but the logistic regression needed an explicit search over polynomial degrees and materialised every one of those features in memory, while the SVM got there with one kernel choice and no feature construction at all.

The notebook ends by plotting the RBF decision boundary over the test points, which shows the closed, curved regions a local kernel produces.


---

## Data

- `synthetic_binary.csv` — the two-feature binary set used above
- `186_satimage.csv` — Statlog (Landsat Satellite), used by the multiclass work in this folder

## Running this

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
jupyter notebook
```

Data files are read from the notebook's directory.
