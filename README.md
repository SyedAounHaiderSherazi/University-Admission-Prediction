# 🎓 University Admission Prediction - Logistic Regression

<img width="1375" height="632" alt="Screenshot 2025-08-08 050610" src="https://github.com/user-attachments/assets/e6884557-1de1-450f-ab8b-efbc696a24d8" />
<img width="1410" height="622" alt="Screenshot 2025-08-08 050632" src="https://github.com/user-attachments/assets/b887d94f-61ab-466a-915c-b3d4e4b75d8c" />
<img width="1370" height="676" alt="Screenshot 2025-08-08 050703" src="https://github.com/user-attachments/assets/bdd4fd51-90a5-46a1-80ad-00f384129952" />


This project is part of Course 1 of the Machine Learning Specialization by Andrew Ng (DeepLearning.AI). It focuses on binary classification using logistic regression to predict student admissions based on exam scores.

---

## 📌 Problem Statement

Given two exam scores for university applicants, predict whether each student gets admitted (1) or not admitted (0).

---

## 📈 Concepts & Formulas

### Sigmoid Function:

    h(x) = 1 / (1 + e^(-θᵀ * x))

Maps output to a probability between 0 and 1.

---

### Logistic Cost Function:

    J(θ) = -(1 / m) * Σ [ yᶦ * log(h(xᶦ)) + (1 - yᶦ) * log(1 - h(xᶦ)) ]

---

### Gradient Descent Update:

    θj := θj - α * (1 / m) * Σ ((h(xᶦ) - yᶦ) * xⱼᶦ)

---

### Regularized Logistic Regression:

To avoid overfitting when dealing with complex feature mappings, regularization is applied:

#### Regularized Cost Function:

    J(θ) = Original Cost + (λ / (2 * m)) * Σ θj²  for j ≥ 1

#### Regularized Gradient:

    θj := θj - α * [ (1 / m) * Σ ((h(xᶦ) - yᶦ) * xⱼᶦ) + (λ / m) * θj ]

---

## 🔧 Implementation

- Language: Python
- Libraries: NumPy, Matplotlib
- Implemented sigmoid, cost function, and gradients manually
- Visualized decision boundaries
- Used regularization to reduce overfitting

---

## 📊 Outputs

- Decision boundary plotted on exam score data
- Accuracy of the model printed
- Regularized model improves classification on complex datasets

---

## 🧠 Skills Learned

- Binary classification theory
- Logistic regression from scratch
- Feature mapping and regularization
- Visualizing and evaluating classification models
