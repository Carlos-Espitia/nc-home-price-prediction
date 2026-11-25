# ITCS 3156 (Fall 2026): Topics Covered in Class

Goal: use these course methods in the project and name them in the report, so it's clear the project builds on what we learned.
Source: Canvas Modules page (copied Nov 24, 2026).

## Course modules
| Module | Topic | Class notes |
|---|---|---|
| 1 | Intro to Machine Learning (ML landscape, ML code sketch) | |
| 2 | ML Basics, ML pipeline, data exploration (HW: Data Exploration) | Hyperplane, Aug 25 |
| 3 | Matrix operations, **K-Nearest Neighbors (KNN)** | Matrix Operations, Sep 3 |
| 4 | Derivatives, **Linear Regression, Ordinary Least Squares (closed form)** | Derivatives, Sep 10; Linear Regression, Sep 15 |
| 5 | **Gradient Descent**, Least Mean Squares (LMS) | Gradient Descent, Sep 22 and Sep 24 |
| 6 | **Curve fitting (polynomial features), Regularization (Ridge/Lasso)**, overfitting | Lasso Regression, Oct 1 |
| 7 | **Perceptron**, kernels | Perceptron, Oct 6 / Oct 8 / Oct 20 (kernels) |
| 8 | Probability theory, **Naive Bayes** | Naive Bayes, Oct 27 and Oct 29 |
| 9 | **Logistic Regression** | Logistic Regression, Nov 3 and Nov 5 |
| 10 | Unsupervised learning: **K-Means clustering**, **PCA** | PCA, Nov 12 |
| 11–12 | **Neural networks** (single layer, multilayer/deep NN, partial derivatives, backprop) | course notebooks |
| 13 | Reinforcement learning | course notebook |

Also in the course: a notation table (`3156_Notation_Table.pdf`; use its notation in the report's equations) and an **AI Acknowledgement Example** (follow it for the Acknowledgement section).

## Ideas for using these in the project (written while the project was still about rent; see the house-price plan in NOTES.md)
- **ML pipeline / data exploration (M1–2):** structure the notebook and report as the pipeline taught in class.
- **Linear regression, OLS (M4):** baseline model. Could also implement the closed-form solution, w = (XᵀX)⁻¹Xᵀy, with NumPy and check it against scikit-learn.
- **Gradient descent / LMS (M5):** train linear regression with our own GD loop and plot the loss curve. This shows why feature scaling matters.
- **Polynomial features and Ridge/Lasso (M6):** test whether extra features overfit; Lasso shows which features it drops (feature selection).
- **KNN (M3):** KNN regression is a natural fit for prices ("similar homes nearby sell for similar prices"). Tune k with cross-validation.
- **Neural network (M11–12):** a multilayer network for regression, compared against the linear models.
- **K-Means (M10):** cluster homes or ZIP codes into market segments and use the cluster as a feature or for analysis.
- **PCA (M10):** visualize the feature space in 2D, or test whether reducing dimensions helps.
- **Logistic regression / Naive Bayes (M8–9):** classification versions of the problem, e.g. "is this home above or below the median price for its city?"
- **Perceptron (M7), RL (M13):** probably not a natural fit; skip unless a use comes up.

## My own homework code that can be reused (in `../`, checked Nov 24)
Reusing my own from-scratch implementations shows the project builds on class work. Each one can be checked against scikit-learn's version.

| Homework notebook | What I implemented | Use in the project |
|---|---|---|
| `hw1/hw-data-exploration-1.ipynb` | `value_counts` bar charts, null check, `describe`, boxplots, **log-transforming a skewed target** (`np.log(area + 1)`), scatter matrix, target-vs-feature scatter grid, correlation matrix | EDA template. Home prices are right-skewed like fire area, so the same log trick applies |
| `hw-derivatives-and-ols.ipynb` | `add_bias`, `sse`/`mse`/`rmse`, **`OrdinaryLeastSquares`** (normal equation with `pinv`), seaborn pairplot with regression lines | Baseline linear regression from scratch |
| `hw-gd-and-lms.ipynb` | **`Standardize`** class (fit on train only), `get_batches`, **`LeastMeanSquares`** (mini-batch gradient descent), learning curves | Train linear regression with GD and show the loss curve |
| `hw-cf-and-reg.ipynb` | `l1_norm`/`squared_l2_norm`, **`poly_features`**, train/valid/test split, **`RidgeOrdinaryLeastSquares`**, **`RidgeLeastMeanSquares`** | Polynomial features plus Ridge; tune lambda on a validation set |
| `hw-la-and-knn.ipynb` | `euclidean_distance`, **`KNearestNeighbors`** class, sweeping k | KNN; the prediction step needs changing from majority vote to **average** for regression |
| `hw_kmeans.ipynb` | `squared_distances`, `compute_inertia`, **`KMeans`** class, **elbow method** | Cluster homes or ZIP codes into market segments |
| `logistic_regression/hw-logistic-regression.ipynb` | `sigmoid`, `softmax`, NLL loss, `SoftmaxRegression`, confusion matrix plot | Only if we add a classification version (e.g. price tier) |
| `hw-naive-bayes.ipynb` | Gaussian Naive Bayes (log priors, log likelihoods) | Same: classification only |
| `hw-perceptron.ipynb` | Perceptron, accuracy/precision/recall/specificity | Probably not needed |
| `ml-sketch-updated.ipynb` | Train/test split → StandardScaler → model pipeline; **`MLPRegressor`** example (4×25 ReLU layers) | Pipeline structure; neural network regression starting point |

No homework covered PCA or neural networks (those exist only as lecture notes/notebooks), so those would use scikit-learn directly.

### Past Acknowledgment/Reference style (worth improving for the project)
Homework acknowledgments were one line ("google search AI mode", "google search and gemini"), and references were "Scikit-learn and NumPy documentation." The project grades these at 10 pts each, so it needs more: say specifically which AI tools were used and for what, follow the course's **AI Acknowledgement Example** file, and give full MLA entries with in-text citations.

## Important note
Random Forest and XGBoost (in the original project plan) are **not** on this list. If we use them, the report should say they go beyond the course. Alternatively, swap them for course methods (KNN, neural network) so the models clearly come from class.
