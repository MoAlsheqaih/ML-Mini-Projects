# Neural Networks From Scratch

Building the machinery of a neural network by hand in numpy: forward propagation, the cost function, backpropagation, and gradient descent, with no library model doing the work. Two notebooks so far, working up from a single neuron to a one-hidden-layer network on the same dataset, which makes the comparison between them a controlled one.

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

---

## Data

- `Airfoil.csv` — NASA airfoil self-noise measurements, used for the regression work in this folder
- `Vehicles.csv` — Vehicle Silhouettes, used for the multiclass work in this folder

## Running this

```bash
pip install numpy scikit-learn matplotlib pandas seaborn
jupyter notebook
```

The breast cancer data loads directly from scikit-learn; the CSV files are read from the notebook's directory.
