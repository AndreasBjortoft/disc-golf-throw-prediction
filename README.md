# Neural Network-Based Prediction of Disc Golf Throw Speed

## 📌 Overview  
This project explores the use of **machine learning models** to predict disc golf throw speed using multi-modal biomechanical data.  
We combined **motion capture (3D kinematics), EMG (muscle activation), and force plate data** to analyze biomechanics and estimate disc velocity.  

Two predictive approaches were compared:  
- **Artificial Neural Network (ANN)**  
- **Markov Chain Monte Carlo (Slice Sampling Regression)**  

The work was part of a group project at *Luleå University of Technology*.  

---

## 🎯 Goals  
- Identify biomechanical factors influencing disc golf throw velocity.  
- Develop machine learning models that predict disc speed from motion and muscle data.  
- Compare ANN vs. Slice Sampling regression in terms of prediction accuracy.  

---

## 🧠 Methods  
- **Data collection**:  
  - Motion capture with Qualisys (300 Hz)  
  - Surface EMG (1500 Hz) from 4 arm/shoulder muscles  
  - Force plate (300 Hz) for bracing foot  
  - Disc velocity measured with smart disc + radar gun  

- **Preprocessing**:  
  - Synchronization of different sampling rates  
  - Normalization of EMG against max-activation tests  
  - Manual labeling of motion markers  
  - Train/validation split: 70/30  

- **Models**:  
  - ANN: Fully connected feedforward NN (PyTorch), RMSE ≈ 7.6  
  - Slice Sampling regression (Matlab), RMSE ≈ 6.6  

---

## 📊 Results  
- Both models could predict disc throw speed **far better than random guessing** (RMSE ≈ 20).  
- **Slice Sampling performed slightly better** than ANN.  
- Hand speed was the strongest predictor of disc velocity.  
- Some EMG variables contributed little, likely due to limitations in muscle activation tests.  

<p align="center">
  <img src="results/scatter_ann_vs_slice.png" alt="ANN vs Slice Sampling results" width="500"/>
</p>

---

## 🚀 Future Work  
- Collect more data across multiple athletes (current dataset = 1 participant, ~60 throws).  
- Explore Spiking Neural Networks (SNNs) for continuous time-series data.  
- Improve EMG max-activation tests for sport-specific validity.  
- Integrate additional force plate and timing features.  

---

## 📂 Repository Structure  
