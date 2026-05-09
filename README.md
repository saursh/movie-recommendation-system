# 🎬 Movie Recommendation System

A hands-on series building a personalized recommendation system from scratch — starting with foundational concepts and progressing through classical ML to deep learning.

| Part | Topic | Status |
|------|-------|--------|
| [Part A](#-part-a-how-recommendation-systems-work) | How Recommendation Systems Work | ✅ Complete |
| [Part B](#%EF%B8%8F-part-b-svd-implementation) | SVD Implementation | ✅ Complete · [Open Notebook »](./movie_recommender_svd.ipynb) |
| Part C | Neural Network Implementation | 🔜 Coming Soon |

---

## 🧠 Part A: How Recommendation Systems Work




### The Problem

Given a history of user ratings, recommend movies a user **hasn't seen yet** — without asking them directly.

| User | Star Wars | Titanic | The Matrix |
|------|-----------|---------|------------|
| Alice | ⭐ 5 | ⭐ 2 | ? |
| Bob | ⭐ 4 | — | ⭐ 5 |

Should we recommend *The Matrix* to Alice? The system has to infer this from behavior alone.

---

### Collaborative Filtering

The core idea: **users who agreed in the past will agree in the future.**

Alice and Bob both love Star Wars → Bob also loves The Matrix → recommend The Matrix to Alice.

This approach requires no knowledge of what a movie is *about*. It learns purely from collective behavior — the same signal Netflix, Spotify, and Amazon use at scale.

---

### Representing Users and Movies as Vectors (Embeddings)

The model converts every user and every movie into a list of numbers called an **embedding**. Each number captures a hidden preference dimension — things like "tends toward action" or "prefers older films" — that the model discovers entirely on its own during training. These dimensions are never labeled by humans.

**Example (simplified to 2 dimensions for illustration):**

| | Dim 1 | Dim 2 |
|--|-------|-------|
| Alice (user) | 0.9 | 0.1 |
| The Matrix (movie) | 0.85 | 0.15 |
| Titanic (movie) | 0.1 | 0.9 |

> **In practice**, embeddings have far more dimensions — typically **50 to 200**. More dimensions let the model capture subtler patterns (e.g., "prefers ensemble casts" or "responds to slow-burn narratives"), but increase training cost and the risk of overfitting. Production systems at Netflix or Spotify operate in the hundreds of dimensions.

Users and movies that share similar patterns end up with similar vectors — which is exactly what makes recommendations possible.

---

### Predicting a Rating

A predicted rating is computed as the **dot product** of a user vector and a movie vector — a measure of how aligned they are.

- Alice × The Matrix → `(0.9 × 0.85) + (0.1 × 0.15)` ≈ **0.78** → strong match ✅
- Alice × Titanic → `(0.9 × 0.1) + (0.1 × 0.9)` ≈ **0.18** → weak match ❌

---

### How the Model Learns: Embedding Alignment

Embeddings are **not predefined** — they are learned by repeatedly adjusting vectors to reduce prediction error.

**Step 1 — Random initialization**

All vectors start as random numbers. They carry no meaning yet.

```
Alice      = [0.2, 0.4]
The Matrix = [0.3, 0.1]
```

**Step 2 — Predict and measure error**

```
Predicted = dot(Alice, The Matrix) = (0.2×0.3) + (0.4×0.1) = 0.10
Actual rating = 5
Error = 5 − 0.10 = 4.9   ← very large; vectors are poorly aligned
```

**Step 3 — Nudge vectors toward each other**

Both vectors are adjusted slightly in the direction that reduces the error:

```
Alice      = [0.4, 0.5]   ← moved toward The Matrix
The Matrix = [0.5, 0.3]   ← moved toward Alice
```

After this nudge:
```
Predicted = (0.4×0.5) + (0.5×0.3) = 0.35
Error = 5 − 0.35 = 4.65  ← still large, but smaller than before
```

This process repeats across every known rating in the dataset, for many passes (called epochs), until predictions stabilize.

**Training loop (pseudocode)**

```
initialize all user_vectors and movie_vectors randomly

for epoch in 1 to max_epochs:
    for each (user, movie, actual_rating) in training_data:
        predicted = dot(user_vectors[user], movie_vectors[movie])
        error = actual_rating - predicted

        user_vectors[user]   += learning_rate × error × movie_vectors[movie]
        movie_vectors[movie] += learning_rate × error × user_vectors[user]

        # Regularization: prevent vectors from growing too large
        user_vectors[user]   -= regularization × user_vectors[user]
        movie_vectors[movie] -= regularization × movie_vectors[movie]

    # Termination check — stop early if improvement has plateaued
    if improvement_this_epoch < tolerance:
        break
```

**When does training stop?**
- A fixed number of epochs is reached, OR
- Improvement per epoch falls below a threshold (early stopping), OR
- Validation error starts rising — meaning the model is beginning to overfit

At the end of training, each user vector has drifted to encode that user's taste, and each movie vector encodes that movie's latent character — purely from the pattern of ratings, with no human labels.

---

### Where the Approaches Diverge

Everything above — collaborative filtering, embeddings, dot product scoring, gradient-based learning — is shared by all modern recommendation systems. What differs is **what happens after the dot product**:

- **Classical methods** (Part B) keep the dot product as-is. Simple, interpretable, and fast to train.
- **Neural network methods** (Part C) pass the user and movie vectors through non-linear layers, allowing the model to capture conditional patterns like *"this user likes action, but only when it's combined with comedy."*

Both approaches use the same conceptual foundation. Part B shows the classical path first.



---

## ⚙️ Part B: SVD Implementation

<details>
<summary><strong>▶ Expand · <a href="./movie_recommender_svd.ipynb">Open Notebook »</a></strong></summary>

### What is SVD?

SVD (Singular Value Decomposition) is a matrix factorization technique that applies the learning loop from Part A directly to a user–movie ratings matrix. It was the breakthrough method behind the winning entry in the **Netflix Prize competition** (2009) and remains a strong baseline in production systems today.

---

### Dataset

**MovieLens 100K** — a standard benchmark dataset containing 100,000 ratings from 943 users across 1,682 movies (rated on a 1–5 scale). Loaded directly via `scikit-surprise`, which handles download and formatting.

---

### Two-Stage Pipeline

The notebook mirrors how recommender systems are actually deployed in production:

**Stage 1 — Evaluate (train on 80%, test on 20%)**

The model is trained on a subset of ratings, then asked to predict ratings it has never seen. This gives an honest measure of whether the model has learned generalizable patterns or just memorized the training data.

**Stage 2 — Deploy (retrain on 100%)**

Once the model passes evaluation, it is retrained on the full dataset before generating recommendations. Every rating is a signal — leaving 20% out would produce slightly worse recommendations for no benefit once we've already validated the approach.

---

### Evaluation Metrics

| Metric | What it measures | Why it matters |
|--------|-----------------|----------------|
| **RMSE** | Average magnitude of rating prediction error | Penalizes large errors more heavily |
| **MAE** | Average absolute rating prediction error | Easier to interpret ("off by X stars on average") |
| **Precision@K** | Of the top-K recommendations, what fraction does the user actually like? | Closest to what the user actually experiences |

Precision@K is the most business-relevant metric — a user never sees whether a predicted rating was 4.1 vs 4.3, but they absolutely notice whether the top 5 recommendations are good.

---

### Recommendation Logic

For a given user, the system:
1. Identifies all movies the user has already rated and excludes them
2. Runs SVD prediction on every remaining movie
3. Sorts by predicted rating (descending)
4. Returns the top N

This ensures every recommendation is a movie the user hasn't already seen.

---

### Example Output

```
Top 5 recommendations for User 196:

1. Star Wars (1977)          — Predicted Rating: 4.82
2. The Godfather (1972)      — Predicted Rating: 4.78
3. Schindler's List (1993)   — Predicted Rating: 4.75
4. Casablanca (1942)         — Predicted Rating: 4.71
5. Rear Window (1954)        — Predicted Rating: 4.68
```

---

### Tech Stack

Python · Pandas · NumPy · scikit-surprise

</details>

---

## 🔜 Part C: Neural Network Implementation *(Coming Soon)*

Part C will extend this system using deep learning — replacing the dot product with a neural network that can model non-linear user-movie interactions, handle cold-start users, and incorporate side features like genre and release year.

---

## About This Project

Built to demonstrate practical ML system design — from problem framing and model selection through evaluation methodology and deployment strategy. The concepts here underpin recommendation engines at Netflix, Spotify, Amazon, and YouTube.
