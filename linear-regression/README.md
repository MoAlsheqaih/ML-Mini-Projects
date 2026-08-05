# Linear Regression

Least-squares regression built from the ground up, then compared against the library implementation.

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

---

## Running this

```bash
pip install numpy scikit-learn matplotlib
jupyter notebook diabetes-least-squares.ipynb
```

`diabetes-data.csv` is included and is read from the notebook's directory.
