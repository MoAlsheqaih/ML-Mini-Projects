# Logistic Regression and SVM

Classification with linear models and kernels, starting from a binary problem that a straight line cannot solve. Four notebooks. The first two reach nearly the same accuracy on a binary problem by opposite routes, one expanding features explicitly and the other letting a kernel do it implicitly. The second pair repeats the comparison on real multi-class satellite data, where the kernel wins and it is possible to see exactly which classes the gain comes from.

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



## `satimage-multiclass-logistic-regression.ipynb` — six classes, real data

Landsat satellite imagery from the Statlog (Landsat Satellite) dataset: 6,435 samples, 36 features (four spectral bands for each of the nine pixels in a 3x3 neighbourhood), and six land-cover classes, from red soil to cotton crop to three grades of grey soil. Split 70 / 15 / 15, stratified, `random_state=777`.

Sweeping the inverse-regularization strength `C`:

| C | Validation accuracy |
|:---|:---|
| 0.001 | 0.8039 |
| 0.01 | 0.8216 |
| 0.1 | 0.8402 |
| 1 | 0.8548 |
| **10** | **0.8589** |
| 100 | 0.8568 |
| 1000 | 0.8579 |

Selected C=10, giving **0.8611 test accuracy**. The curve is shallow past C=1, so regularization strength is not what limits this model.

The per-class breakdown is where it gets interesting:

| Class | Precision | Recall | F1 | Support |
|:---|:---|:---|:---|:---|
| 1 red soil | 0.96 | 0.99 | 0.97 | 230 |
| 2 cotton crop | 0.95 | 0.94 | 0.95 | 105 |
| 3 grey soil | 0.88 | 0.92 | 0.90 | 204 |
| 4 damp grey soil | 0.53 | 0.34 | **0.42** | 94 |
| 5 veg stubble | 0.83 | 0.77 | 0.80 | 106 |
| 7 very damp soil | 0.81 | 0.90 | 0.85 | 226 |

86% overall accuracy hides a class that barely works. Damp grey soil is recovered only a third of the time, with 35 of its samples predicted as very damp soil. That is not a modelling accident — the three grey-soil classes differ by moisture content, which shifts the spectral signature only slightly, so a linear boundary in these 36 features cannot cleanly separate them.

The gap between macro average (0.81 F1) and weighted average (0.85 F1) is the same story in one number: the big, easy classes are carrying the headline figure.


## `satimage-multiclass-svm.ipynb` — where the non-linear gain actually lands

The same data and splits, with an SVM grid over `C` in {0.1, 1, 10, 100} crossed with linear, polynomial (degrees 2 to 4) and RBF kernels. Best configuration: **C=100 with an RBF kernel**, at 0.9098 validation and **0.9098 test accuracy**.

Two results are worth separating.

**The linear kernel confirms the previous notebook.** It peaks at 0.8703 validation, right alongside logistic regression's 0.8611. Two different algorithms drawing linear boundaries land in the same place, which is what you would hope for and a decent sanity check on both.

**The improvement is not spread evenly.** Overall accuracy rises from 86.11% to 90.98%, but look at where:

| Class | Logistic regression F1 | SVM F1 |
|:---|:---|:---|
| 1 red soil | 0.97 | 0.98 |
| 2 cotton crop | 0.95 | 0.96 |
| 3 grey soil | 0.90 | 0.92 |
| 4 damp grey soil | 0.42 | **0.66** |
| 5 veg stubble | 0.80 | **0.90** |
| 7 very damp soil | 0.85 | 0.90 |

The easy classes barely move. Nearly all of the gain sits in damp grey soil and vegetation stubble, precisely the two classes the linear model handled worst. Class 4 to Class 7 confusion drops from 35 cases to 14. The non-linear boundary is doing work exactly where a linear one was failing, which is the strongest evidence that the classes are not linearly separable rather than simply under-tuned.

Damp grey soil remains the hardest class at 0.66 F1 even after the improvement. Some of that boundary is genuinely ambiguous in this feature space.

---

## Data

- `synthetic_binary.csv` — the two-feature binary set used above
- `186_satimage.csv` — Statlog (Landsat Satellite), 6,435 samples across six land-cover classes

## Running this

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
jupyter notebook
```

Data files are read from the notebook's directory.
