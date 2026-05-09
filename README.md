# 🎬 Movie Recommendation System using Machine Learning (SVD)

A production-style movie recommendation system built using collaborative filtering (SVD) on the MovieLens dataset.

---

## 🧠 Part A: Conceptual Understanding
### 1. Problem Statement (with Example)

We are given historical user–movie ratings and want to recommend movies a user has **not seen yet**.

Example:

| User  | Movie        | Rating |
|------|--------------|--------|
| Alice | Star Wars    | 5      |
| Alice | Titanic      | 2      |
| Bob   | Star Wars    | 4      |
| Bob   | The Matrix   | 5      |

Question:  
**Should we recommend _The Matrix_ to Alice?**

The system must infer preferences **without Alice explicitly rating The Matrix**.
---

### 2. Why Collaborative Filtering? (with Example)

Collaborative filtering is based on the idea:

> Users who behaved similarly in the past will behave similarly in the future.

Example:
- Alice and Bob both like **Star Wars**
- Bob also likes **The Matrix**

👉 It is reasonable to recommend **The Matrix** to Alice.

This approach:
- Does **not** require movie metadata
- Learns directly from **user behavior**
- Captures collective patterns

---

### 3. Embeddings and Dimensions (Core Concept)

The model represents each user and each movie as an **embedding** — a low‑dimensional numerical vector.

Each element of the embedding vector is called a **dimension**.

Each dimension captures a hidden preference or characteristic learned from data, such as:
- preference for action movies
- preference for romance
- preference for older movies

These dimensions are **not explicitly labeled** and are learned automatically during training.
---

### 4. Users and Movies as Embedding Vectors

SVD maps users and movies into a shared embedding space.

Example (2 embedding dimensions):

**User embeddings**

| User  | Dim 1 | Dim 2 |
|------|-------|-------|
| Alice | 0.9   | 0.1   |
| Bob   | 0.8   | 0.2   |

**Movie embeddings**

| Movie        | Dim 1 | Dim 2 |
|--------------|-------|-------|
| Star Wars    | 0.9   | 0.1   |
| Titanic      | 0.1   | 0.9   |
| The Matrix   | 0.85  | 0.15  |

---

### 5. How a Rating Is Predicted (Concrete Example)

A rating is predicted using a **dot product** of embeddings:
Predicted Rating = User Embedding ⋅ Movie Embedding
To predict Alice’s rating for **The Matrix**:
(0.9 × 0.85) + (0.1 × 0.15) ≈ 0.78  → high score
✅ The model predicts Alice will like **The Matrix**.

To predict Alice’s rating for **Titanic**:
(0.9 × 0.1) + (0.1 × 0.9) = 0.18 → low score
✅ The model predicts Alice will not like Titanic.

---

### 6. How Embeddings Are Learned: Alignment During Training

Embeddings are **not predefined**.  
They are learned through an iterative process called **alignment**.

#### Step 1: Random Initialization
Alice = [0.2, 0.4]
The Matrix = [0.3, 0.1]
These values have no meaning yet.

#### Step 2: Predict and Measure Error
Prediction = Alice · The Matrix = 0.10
Actual Rating = 5
Error = 5 − 0.10 = 4.9
The large error indicates poor alignment.

#### Step 3: Adjust (Align) the Embeddings

To reduce error:
- User embedding moves **toward** the movie embedding
- Movie embedding moves **toward** the user embedding

Alice = [0.4, 0.5]
The Matrix = [0.5, 0.3]
The embeddings are now more aligned.


---

### 7. Pseudocode: Embedding Learning Loop

initialize user_embeddings randomly; initialize movie_embeddings randomly
for epoch in range(num_epochs):
for (user, movie, rating) in training_data:
    prediction = dot(user_embedding[user], movie_embedding[movie])
    error = rating - prediction
    update user_embedding[user] to reduce error
    update movie_embedding[movie] to reduce error
    apply regularization
    
---

### 8. When Does Training Stop?

Training stops when:
- A fixed number of epochs is reached
- Error reduction becomes marginal
- Regularization prevents embeddings from growing too large

---

### 9. Key Limitation (Motivation for Neural Models)

SVD combines embeddings using a **linear dot product**.

This limits the model’s ability to capture:
- conditional preferences
- non-linear interactions

Neural recommender systems extend this idea by applying **non-linear functions** on top of embeddings.
    
---
## ⚙️ Part B: Implementation & Results
## 🚀 Project Overview

This project implements a personalized recommendation system that learns user preferences from past ratings and predicts movies users are likely to enjoy.

### The system:
- Learns latent user and movie representations using matrix factorization  
- Predicts ratings for unseen movies  
- Recommends top‑N movies tailored to each user  

---

## 🧠 Key Concepts Demonstrated

- ✅ Collaborative Filtering  
- ✅ Matrix Factorization (SVD)  
- ✅ Embedding learning (latent factors)  
- ✅ Model evaluation (RMSE, MAE)  
- ✅ Ranking metrics (Precision@K)  
- ✅ Real-world ML workflow (train → evaluate → deploy)  

---

## 📂 Dataset

We use the **MovieLens 100K dataset**, containing:

- 100,000 ratings  
- 943 users  
- 1,682 movies  

### Files:
- `u.data` → ratings (user–movie interactions)  
- `u.item` → movie titles and metadata  

---

## ⚙️ Tech Stack

- Python  
- Pandas  
- NumPy  
- scikit-surprise  

---

## 🤖 Model: SVD (Matrix Factorization & Embeddings)

This project uses **Singular Value Decomposition (SVD)**, a widely used machine learning technique for recommender systems.

### 🧠 Core Idea

Instead of memorizing ratings, the model learns:

- **User embeddings** → represent user preferences  
- **Movie embeddings** → represent latent movie features  

These embeddings are low-dimensional vectors capturing hidden patterns in the data.

---

## 📌 Intuition: Recommendation as Similarity

Each user and movie is mapped into a shared latent space:

- Users are placed near movies they are likely to enjoy  
- Similar movies appear close to each other  

👉 Recommendation becomes:

> Find movies whose embeddings are **closest to the user embedding**

---

## 📐 How Prediction Works

The predicted rating is computed as:

- ✅ High similarity → high predicted rating  
- ❌ Low similarity → low predicted rating  

---

## 🎯 Why This Works

This allows the model to:

- Generalize beyond known ratings  
- Recommend unseen movies  
- Capture patterns like:
  - “user prefers sci‑fi + action”  
  - “movie belongs to similar latent features”  

---

## 🏆 Industry Relevance

This approach:

- Was popularized in the **Netflix Prize competition**  
- Remains foundational in modern recommender systems  
- Forms the basis for many deep learning recommendation models  

---

## 📊 Model Evaluation

The model is evaluated using a train/test split.

### Metrics:
- **RMSE (Root Mean Squared Error)**  
- **MAE (Mean Absolute Error)**  

✅ These measure how accurately the model predicts user ratings.

---

## 🎯 Recommendation Quality

We also evaluate ranking performance using:

### ✅ Precision@K

Measures how many of the top‑K recommended movies are relevant.
👉 Higher values indicate better recommendation quality.

---

## 🔄 Real-World Workflow (Important Design)

This project follows a real-world ML pipeline:

### ✅ Stage 1 — Evaluation
- Split data into train/test  
- Train model on known data  
- Evaluate on unseen data  

### ✅ Stage 2 — Deployment Model
- Retrain model on full dataset  
- Use all available data for recommendations  

👉 This ensures both:
- unbiased evaluation  
- best-performing recommendation system  

---

## 🍿 Recommendation Strategy

For a given user:

1. Identify movies already rated (seen)  
2. Filter them out  
3. Predict ratings for unseen movies  
4. Rank movies by predicted rating  
5. Return top‑N recommendations  

---

## ✅ Example Output

Top 5 recommendations:

1. Star Wars (1977) — Predicted Rating: 4.82  
2. The Godfather (1972) — Predicted Rating: 4.78  
3. Shawshank Redemption, The (1994) — Predicted Rating: 4.75  
...  

---

## 🧩 Key Features

- ✅ Personalized recommendations using ML  
- ✅ Clean evaluation pipeline (RMSE, MAE, Precision@K)  
- ✅ Excludes already watched movies  
- ✅ Movie ID → title mapping  
- ✅ Production-style workflow  
- ✅ Embedding-based recommendation logic  

---

## 📈 Future Improvements

- Add Recall@K, NDCG  
- Build REST API (FastAPI)  
- Create UI (Streamlit)  
- Implement neural recommender (PyTorch)  
- Handle cold-start users  

---

## 🧠 What I Learned

- Difference between prediction accuracy vs recommendation quality  
- Importance of train/test separation in recommender systems  
- How recommender systems learn latent embeddings  
- How recommendations can be viewed as a similarity problem in vector space  
- Why real systems use full data after evaluation  

---

## 🚀 How to Run

1. Open the notebook in Google Colab  
2. Install dependencies  
3. Run all cells  
4. View evaluation metrics and recommendations  

---

## 📘 Notebook

You can view the full implementation here:

- `movie_recommender_svd.ipynb`
---

## 📬 Conclusion

This project demonstrates a complete pipeline for building a recommendation system:

- ✅ Train a model using collaborative filtering  
- ✅ Evaluate it correctly  
- ✅ Deploy a full-data model for recommendations  

---

## ⭐ Support

If you found this useful, consider giving the repository a ⭐
