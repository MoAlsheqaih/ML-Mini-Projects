# Neural Networks From Scratch

Building the machinery of a neural network by hand in numpy: forward propagation, the cost function, backpropagation, and gradient descent, with no library model doing the work.

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

---

## Data

- `Airfoil.csv` — NASA airfoil self-noise measurements, used for the regression work in this folder
- `Vehicles.csv` — Vehicle Silhouettes, used for the multiclass work in this folder

## Running this

```bash
pip install numpy scikit-learn matplotlib pandas seaborn
jupyter notebook breast-cancer-logistic-regression.ipynb
```

The breast cancer data loads directly from scikit-learn; the CSV files are read from the notebook's directory.
