# Neural Networks From Scratch

Building the machinery of a neural network by hand in numpy: forward propagation, the cost function, backpropagation, and gradient descent, with no library model doing the work. Four notebooks, working up from a single neuron to a one-hidden-layer network, then taking that same network to multiclass classification and to regression by changing only its output layer and cost.

Coursework for ICS485 (Machine Learning), KFUPM.

---

## `breast-cancer-logistic-regression.ipynb` — a single neuron, built piece by piece

Logistic regression implemented end to end on the Breast Cancer Wisconsin dataset (569 samples, 30 features, malignant vs benign). scikit-learn supplies the data and the train/test split; everything after that is written from scratch.

The pieces, each built and tested in isolation before being assembled:

| Function | What it does |
|:---|:---|
| `sigmoid()` | squashes any real value into (0, 1) so the output reads as a probability |
| `initialize_with_zeros()` | creates the weight vector and bias |
| `propagate()` | one forward pass to the cost, one backward pass to the gradients |
| `optimize()` | the gradient descent loop |
| `predict()` | thresholds probabilities at 0.5 |
| `LogisticModel()` | calls the above in order and reports accuracy |

Trained for 1,500 iterations at a learning rate of 0.01, the cost falls from 0.693 to 0.095 and the model reaches **98.46% training accuracy and 96.49% test accuracy**.

### Why this is the on-ramp to a neural network

Framed this way, logistic regression *is* a one-neuron network: a linear combination of inputs, a sigmoid activation, a cost, and gradients flowing back to the parameters. Adding a hidden layer changes the bookkeeping, not the ideas.

One detail makes the difference visible. Here `w` is initialized to **zeros**, and that is perfectly safe, because with a single neuron there is no symmetry to break — every parameter still receives its own distinct gradient. The moment a hidden layer exists, zero initialization is fatal: every hidden unit computes the same thing, receives the same gradient, and stays identical forever. That is exactly why the neural network notebooks in this folder initialize randomly instead.

The training curve tells the other half of the story. Cost is still decreasing at iteration 1,500, so training longer would keep lifting training accuracy — but test accuracy would eventually stop following it. Number of iterations is a hyper-parameter to tune, not to maximize.

Everything is vectorised, with no Python loops over training examples.

## `breast-cancer-neural-network.ipynb` — adding a hidden layer

The same dataset, now with a one-hidden-layer network: 10 `tanh` units feeding a sigmoid output, trained for 10,000 iterations at a learning rate of 0.01. Written in numpy throughout, including backpropagation.

The cost falls from 0.693 to 0.049, and the model reaches **98.90% training accuracy and 96.49% test accuracy**.

### The result worth being honest about

The logistic regression notebook reached **96.49%** on the same test set. The neural network reaches **96.49%**.

Training accuracy did improve, from 98.46% to 98.90%, so the extra capacity is being used — it is just being spent fitting the training set rather than generalizing. This dataset is close to linearly separable in its 30 standardized features, so a hidden layer has little structure left to discover. Added capacity only helps when there is non-linearity to capture, and this is a clean demonstration of what happens when there isn't.

### Comparing activation functions

Holding everything else fixed and swapping only the hidden activation:

| Activation | Cost at 5,000 iterations | Train accuracy | Test accuracy |
|:---|:---|:---|:---|
| tanh | 0.0555 | 98.90% | 96.49% |
| sigmoid | 0.1291 | 98.46% | 96.49% |
| ReLU | 0.0552 | 98.90% | **97.37%** |

Sigmoid converges far more slowly, sitting at more than double the cost of the other two after the same number of iterations. That is the vanishing gradient problem in miniature: sigmoid saturates toward 0 and 1, where its derivative approaches zero, so gradients arriving at the hidden layer are repeatedly scaled down. `tanh` is centred at zero and steeper through its middle, and ReLU passes gradient through unchanged wherever its input is positive.

ReLU comes out ahead on test accuracy, but the honest reading is that 96.49% and 97.37% differ by exactly **one test sample** out of 114. That is well inside the noise of a test set this small, and it would be overclaiming to present it as evidence that ReLU generalizes better here. What the table does support cleanly is the convergence difference, where the gap is large and consistent.

### Why initialization changes between the two notebooks

Logistic regression initializes its weights to zeros, and that is safe. This network cannot: with all weights at zero every hidden unit computes the same value, receives the same gradient, and stays identical forever, leaving a network no more expressive than the single neuron it was supposed to improve on. Small random values break that symmetry, and the 0.01 scale keeps `tanh` in its steep central region rather than saturated at the tails.

## `vehicle-silhouettes-multiclass.ipynb` — four classes instead of two

The same network extended to multiclass, on the Vehicle Silhouettes dataset: 846 samples, 18 shape features extracted from vehicle outlines at varying angles, four classes (double-decker bus, Chevrolet van, Saab 9000, Opel Manta). Split 70/15/15 with seed 777, stratified.

Two changes turn the binary network into a multiclass one. **Softmax** replaces the sigmoid at the output, so four units produce a distribution summing to one instead of a single probability, and **categorical cross-entropy** replaces binary cross-entropy as the cost.

Everything else is untouched, and the reason is worth stating: the gradient of categorical cross-entropy with respect to the output pre-activation collapses to `dZ2 = A2 - Y`, exactly the expression the binary network already used. The softmax Jacobian and the derivative of the log cancel. The backward pass genuinely does not change.

Trained with 20 hidden units at a learning rate of 0.1 for 5,000 iterations, the cost falls from 1.386 to 0.157 and the model reaches **85.04% test accuracy**.

That starting cost is a useful check in itself. $\ln(4) = 1.3863$ is the cross-entropy of a uniform distribution over four classes, which is exactly what a network initialized with near-zero weights should produce. Seeing 1.3860 at iteration 0 confirms the softmax and the cost function are wired up correctly before a single gradient step has been taken.

### The errors are not spread evenly

| Class | Precision | Recall | F1 | Support |
|:---|:---|:---|:---|:---|
| bus | 1.00 | 1.00 | **1.00** | 32 |
| van | 1.00 | 0.93 | **0.97** | 30 |
| opel | 0.73 | 0.69 | 0.71 | 32 |
| saab | 0.70 | 0.79 | 0.74 | 33 |

Bus and van are essentially solved: 32 of 32 buses correct, and not a single false positive for either class. Opel and Saab sit around 0.71 and 0.74, and the confusion matrix shows they are being mistaken almost entirely for each other.

That split is a property of the data rather than a failure of the model. A double-decker bus and a van have silhouettes unlike anything else in the set, while the Saab 9000 and the Opel Manta are both saloon cars of similar proportion — from many viewing angles their outlines genuinely overlap, and the 18 shape features do not carry enough information to separate them. Any classifier working from these features would run into the same wall.

The classes are close to balanced (30 to 33 samples each in the test set), which is why macro and weighted averages agree at 0.85. No class is being propped up by size.

**One caveat worth stating plainly:** a validation split is created but never used. The hyper-parameters here are fixed by hand rather than tuned, so the 85.04% is an honest test result but not an optimized one.

## `airfoil-noise-regression.ipynb` — the same network, predicting a number

The last variation, on the NASA Airfoil Self-Noise dataset: 1,503 wind tunnel measurements, five features (frequency, angle of attack, chord length, free-stream velocity, suction side displacement thickness), predicting the scaled sound pressure level in decibels. Split 70/15/15 with seed 777.

Going from classification to regression takes two changes, and they mirror the multiclass ones:

- **The output activation is removed entirely.** `A2 = Z2`, a linear output, because the target is an unbounded real number rather than a probability.
- **Mean squared error replaces cross-entropy** as the cost.

And once again the backward pass is unchanged. `dZ2 = A2 - Y` holds for a linear output with MSE just as it did for sigmoid with binary cross-entropy and softmax with categorical cross-entropy. That is not a coincidence: each of those three activation and cost pairings is chosen precisely so the terms cancel this way, which is why one backpropagation implementation serves all three notebooks.

**The target is scaled too**, not just the features. Sound pressure levels average about 125 dB, and feeding a target that large into a network whose weights start near zero produces enormous initial gradients. Standardizing the target keeps the optimization well conditioned; predictions are then inverse-transformed back to decibels before the error is reported, so the final numbers are in real physical units.

Trained with 25 hidden units at a learning rate of 0.05 for 15,000 iterations, the cost falls from 0.4999 to 0.0853 and the model reaches **6.73 MSE and 2.02 MAE on the test set, in decibels**.

The starting cost is another free correctness check, in the same spirit as $\ln(4)$ in the multiclass notebook. With a standardized target the variance is 1, and with near-zero weights the network predicts approximately zero, so the MSE cost of $\frac{1}{2m}\sum(A^{[2]} - Y)^2$ should begin at about **0.5**. It begins at 0.4999.

An average error of about 2 dB on measurements spanning 103.4 to 141.0 dB (standard deviation 6.9 dB) is a reasonable result for a network with one hidden layer and no tuning. MAE being well below the square root of MSE (2.02 against 2.59) indicates the error distribution has a tail: most predictions are closer than 2 dB, with a smaller number of larger misses pulling the squared error up.

**Same caveat as the multiclass notebook:** the validation split is created but never used, so these are honest test numbers from hand-picked hyper-parameters rather than tuned ones.

---

## Data

- `Airfoil.csv` — NASA Airfoil Self-Noise, 1,503 samples, five features, sound pressure level in decibels
- `Vehicles.csv` — Vehicle Silhouettes, 846 samples, 18 features, four vehicle types

The breast cancer data loads directly from scikit-learn.

## Running these

```bash
pip install numpy scikit-learn matplotlib pandas seaborn
jupyter notebook
```

The CSV files are read from the notebook's directory.
