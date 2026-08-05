# Linear Regression

Two notebooks on linear models. The first builds least-squares regression from the ground up and compares it against the library implementation. The second pushes a linear model into overfitting on purpose, then pulls it back with regularization.

Coursework for ICS485 (Machine Learning), KFUPM.

---

## `diabetes-least-squares.ipynb` — least squares, three ways

Predicting one-year diabetes progression from 10 physiological features (442 patients, the standard diabetes dataset). The notebook works up from the simplest possible model to a from-scratch solver, checking each step against `sklearn`.

**Starting point.** Predicting the mean of `y` and ignoring the features entirely gives MSE 5929.9. That's the number everything else has to beat.

**One feature at a time.** Fitting each feature alone identifies which carry signal:

| Feature | MSE | | Feature | MSE |
|:---|:---|---|:---|:---|
| body mass index | **3890.5** | | serum4 | 4831.1 |
| serum5 | **4031.0** | | serum6 | 5062.4 |
| blood pressure | 4774.1 | | serum1 | 5663.3 |
| serum3 | 5005.7 | | age | 5720.6 |
| | | | sex | 5918.9 |

Body mass index alone cuts the error by 34%, and serum5 is the next most informative. Sex is essentially worthless on its own, barely improving on the constant model. Combining BMI and serum5 reaches 3205.2, and using all ten features gets to 2859.7.

**Two solvers.** The closed-form normal equation and batch gradient descent are both implemented directly in numpy, then checked against `sklearn.linear_model.LinearRegression`:

| Method | Train MSE | Test MSE |
|:---|:---|:---|
| sklearn | 2905.19 | 2877.95 |
| Closed form (normal equation) | 2905.19 | 2877.95 |
| Gradient descent | 2920.90 | 2870.55 |

The closed-form solution matches sklearn to the decimal, which is the point — sklearn is solving the same equation. Gradient descent lands a fraction away in both directions, having stopped near, not exactly at, the optimum.

**Learning curve.** Refitting on growing subsets shows the variance problem directly:

| Training size | Train MSE | Test MSE |
|:---|:---|:---|
| 20 | 1683 | 5868 |
| 50 | 2740 | 3182 |
| 100 | 2968 | 3116 |
| 200 | 2872 | 3042 |
| 300 | 2921 | 2871 |

At n=20 the model fits its training data far better than anything it hasn't seen — a 3.5× gap that is pure overfitting, since 20 points and 10 features leaves almost no room to generalize. The two errors converge as data accumulates, and by n=300 test error has drawn level with train.

A closing section works through what changes when the units of `y` or of an individual feature change, and why no retraining is required in either case.

## `house-prices-ridge-lasso.ipynb` — overfitting on purpose, then fixing it

House price prediction on a deliberately small dataset (20 training rows, in `LandPriceTrain.csv` and `LandPriceTest.csv`). The notebook is structured as a sequence of diagnoses, each motivating the next step.

**Underfitting.** The constant model gives a test MSE of 3.54e9. Adding the raw features drops it to 1.59e8 — a large gain, but train and test errors are both still high, which is the signature of a model too simple for the data.

**Overcorrecting.** Expanding to polynomial features (20×9) cuts train MSE to 2.15e7 while test MSE sits at 9.45e7. Train error has fallen roughly 4× further than test error. The model is now memorizing 20 points.

**Ridge.** Sweeping the penalty:

| alpha | Train MSE | Test MSE |
|:---|:---|:---|
| 0.01 | 2.155e7 | 9.435e7 |
| 0.1 | 2.155e7 | 9.359e7 |
| 0.5 | 2.158e7 | 9.170e7 |
| 1.0 | 2.159e7 | 9.006e7 |
| 5.0 | 2.176e7 | 8.100e7 |
| 10.0 | 2.206e7 | **7.395e7** |

Train error barely moves while test error falls 22%. That asymmetry is the whole argument for regularization: the penalty costs almost nothing on data the model has already seen, and buys a lot on data it hasn't.

**Lasso.** With a much larger penalty scale, Lasso goes further, reaching a test MSE of 5.26e7 at alpha=2000 — better than any Ridge setting tried. The mechanism differs too: Lasso drives coefficients to exactly zero rather than merely shrinking them, so by alpha=1150 only a handful of features remain active and the model has performed its own feature selection. Push alpha too far and it discards useful features as well, and both errors climb again.

The closing section documents the bias–variance reading of these sweeps: how alpha trades one for the other, and why Ridge and Lasso arrive at different answers on the same data.

---

## Running these

```bash
pip install numpy scikit-learn matplotlib
jupyter notebook
```

`diabetes-data.csv`, `LandPriceTrain.csv` and `LandPriceTest.csv` are included and are read from the notebooks' directory.
