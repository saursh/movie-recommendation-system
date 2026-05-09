# 🎬 Movie Recommendation System using Machine Learning (SVD)

A production-style movie recommendation system built using collaborative filtering (SVD) on the MovieLens dataset.

---

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
