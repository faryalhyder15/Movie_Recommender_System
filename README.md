# 🎬 AI Movie Recommender System

## 📌 Abstract

This project is an AI-powered movie recommendation system that suggests personalized movies based on user preferences such as genre, release year, duration, and IMDb rating. It integrates multiple Artificial Intelligence concepts including search algorithms, clustering, neural networks, and decision-making techniques to simulate an intelligent recommendation pipeline. The system is built using a modern interactive web interface to provide a smooth user experience.

---

## 📖 Introduction

The AI Movie Recommender is designed to mimic real-world recommendation engines used by platforms like Netflix and Amazon Prime. It takes user preferences and processes a dataset of movies through several AI techniques to filter, analyze, and rank the best possible recommendations. The system emphasizes explainability by showing how each recommendation is generated.

---

## 🎯 Objectives

* To build an intelligent movie recommendation system.
* To apply multiple AI algorithms in a single pipeline.
* To provide personalized movie suggestions based on user preferences.
* To demonstrate clustering, search algorithms, and neural networks in practice.
* To create an interactive and visually appealing web application.

---

## ⚙️ System Workflow

The recommendation process follows a multi-stage AI pipeline:

1. **Constraint Satisfaction (CSP Filtering)**

   * Filters movies based on genre, rating, year, and duration.

2. **Search Algorithms**

   * BFS: Explores movie dataset in queue-based order.
   * DFS: Explores dataset using stack-based approach.
   * A* Search: Prioritizes movies using heuristic (IMDb rating).

3. **Clustering (K-Means)**

   * Groups similar movies based on features like year, rating, and duration.

4. **Neural Network (ANN)**

   * Predicts personalized ratings for each movie.

5. **Game Theory (Minimax)**

   * Combines user preference and IMDb rating for balanced scoring.

6. **Ranking System**

   * Final movies are ranked using a combined match score.

---

## 🧠 AI Techniques Used

* Constraint Satisfaction Problem (CSP)
* Breadth-First Search (BFS)
* Depth-First Search (DFS)
* A* Search Algorithm
* K-Means Clustering
* Artificial Neural Network (ANN)
* Minimax Decision Strategy

---

## 🗂 Dataset

The system uses a dataset of IMDb top movies containing:

* Title
* Genre
* Year of release
* Duration
* IMDb rating
* Description

---

## 🎮 Features

* Interactive Streamlit UI
* Personalized filtering system
* Multi-algorithm AI pipeline
* Real-time recommendation generation
* Explainable results for each movie
* Top-N recommendation selection

---

## 📊 Output

The system displays:

* Top recommended movies
* IMDb rating vs predicted rating
* Match score
* Heuristic score (A*)
* Minimax decision score
* Explanation for each recommendation

---

## 🚀 Future Improvements

* Add user login and profile-based recommendations
* Integrate collaborative filtering (user-user similarity)
* Use deep learning models for better predictions
* Add movie poster API integration
* Deploy system on cloud (Streamlit Cloud / AWS)
* Add watch history tracking

---

## 👨‍💻 Conclusion

This project demonstrates how multiple AI techniques can be combined into a single intelligent system for movie recommendations. It not only provides accurate suggestions but also explains the reasoning behind each recommendation, making it transparent and educational.

