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
- Extend dataset to multiple athletes (current dataset = 1 participant, ~60 throws).  
- Apply Spiking Neural Networks (SNNs) for continuous time-series data.  
- Improve EMG max-activation tests for sport-specific validity.  
- Explore real-time feedback applications for athletes.  

---

## 📂 Repository Structure  
```
├── Report_discgolf_project.pdf #report of the project
├── src/ # Code for preprocessing, models, training
│ ├── models/
│ └── visualization/
└── README.md
```

---

## 🧑‍🤝‍🧑 Team & Contributions  
This project was developed by:  
- **Andreas G. Björtoft**  
- **Felix S. Woxblom**  
- **Oscar A. Östensson**  

**My contributions included:**  
- Collaborated on the implementation and training of the ANN model in PyTorch  
- Preprocessed EMG signals (normalization, synchronization, and feature preparation)  
- Preprocessed and analyzed force plate data for integration with other modalities  
- Contributed to report writing, data visualization, and final project presentation
- Provided disc golf expertise for experimental design and data interpretation
---

## ⚙️ Tech Stack  
- **Python 3.10**: PyTorch, NumPy, Matplotlib, Scikit-learn  
- **Matlab 2024b**: Slice Sampling regression  
- **Qualisys QTM** (motion capture)  
- **Noraxon EMG system**  
- **Kistler force plate**
- **GameProofer Disc**

---

## Documentation

For a detailed explanation of the experimental setup, preprocessing methods, and model development, see the full report:

📄 [Neural Network-Based Prediction of Disc Golf Throw Speed (PDF)](./Report_discgolf_project.pdf)

