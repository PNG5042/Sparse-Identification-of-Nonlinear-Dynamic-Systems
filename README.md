# Sparse Equation Discovery for High-Temperature Alloy Behavior

**Using machine learning and SINDy to recover interpretable material equations from experimental creep and tensile datasets**

---

## Project Overview

This project explores whether **Sparse Identification of Nonlinear Dynamics (SINDy)** can recover physically meaningful constitutive equations from noisy real-world material datasets.

Traditional material models such as creep and tensile relationships are often derived through regression and empirical fitting. However, these approaches may struggle when data becomes noisy, highly correlated, or high-dimensional.

This project builds a physics-informed pipeline that combines machine learning, sparse regression, and identifiability analysis to determine whether interpretable governing equations can still be recovered.

---

## Why This Matters

High-temperature alloys such as SS316H and Alloy 617 are used in extreme engineering environments where long-term deformation and material strength are critical.

Understanding whether sparse equation discovery methods can recover known physical relationships helps researchers:

* Improve explainability in data-driven material science
* Compare machine learning predictions with classical constitutive laws
* Evaluate model robustness under noise and collinearity
* Identify which features remain physically meaningful

---

## Research Question

> Can sparse equation discovery recover physically interpretable material equations from noisy creep and tensile datasets?

This project investigates:

* Whether SINDy recovers known constitutive terms
* How feature identifiability affects equation discovery
* How QR decorrelation improves numerical conditioning
* The effect of noise on sparse equation similarity

---

## Workflow

```text
Raw Experimental Data
        ↓
Feature Engineering
        ↓
Machine Learning Models
        ↓
QR Decorrelation
        ↓
SINDy Sparse Regression
        ↓
Equation Discovery + Similarity Scoring
```

---

## Key Features

### Physics-Informed Sparse Regression

Uses SINDy to identify interpretable equations instead of black-box predictions.

### Machine Learning Benchmarking

Compares sparse equation discovery against multiple predictive models:

* Ridge Regression
* Random Forest
* Ensemble Models

### Feature Identifiability Analysis

Measures whether candidate physical features remain statistically meaningful.

### QR-Based Decorrelation

Reduces collinearity between engineered variables before sparse regression.

### Noise Sensitivity Testing

Tests whether recovered equations remain stable under increasing noise.

---

## Dataset Overview

### SS316H Creep Dataset

* 144 experimental samples
* Temperature range: 811–1255 K
* Stress range: 10–177 MPa
* Heat correction included through engineered feature H_c

### Alloy 617 Tensile Dataset

* 28 retained room-temperature specimens
* Over 1,300 point-wise measurements
* Multiple strain-rate filter levels tested

---

## Results

### SS316H Creep Results

| Model                 | Test R² | Equation Similarity |
| --------------------- | ------- | ------------------- |
| Ridge + SINDy         | 0.9561  | 0.9793              |
| Random Forest + SINDy | 0.8687  | 0.6997              |
| Ensemble + SINDy      | 0.9554  | 0.8873              |
| Actual Data + SINDy   | 0.7881  | 0.9985              |

### Key Finding

SINDy successfully recovered the dominant Norton–Bailey creep relationship using only sparse terms.

Recovered equation:

```text
log(t) = -21.279 + 51430/T − 5.934·log(σ)
```

This result closely matches the original OLS constitutive relationship.

---

### Alloy 617 Results

| Filter Level | R²      | Similarity |
| ------------ | ------- | ---------- |
| Tight        | -0.3115 | -0.3294    |
| Moderate     | 0.9044  | 0.6189     |
| Percentile   | 0.9036  | 0.6187     |

### Key Finding

Moderate filtering produced the most stable sparse equations while preserving physically meaningful terms.

---

## Visual Results

Add your generated figures here.

Example:

![SINDy Pipeline Results 316 SS]([image-2.png](https://kommodo.ai/i/ncs7jnJfyswgKG02oX2G))

![SINDy Pipeline Results 617 Alloy](image-3.png)

---

## Repository Structure

```text
/project
│
├── Archive Code/
├── Project_base_files/
├──── 316 Stainless Steel tests
├──── Alloy 617 Test
├── pysindy/
├── Unit_testing/
└── README.md
```

---

## How to Run

```bash
git clone https://github.com/PNG5042/Sparse-Identification-of-Nonlinear-Dynamic-Systems?tab=contributing-ov-file
cd Sparse-Identification-of-Nonlinear-Dynamic-Systems
pip install -r requirements.txt
python main.py
```
For more info check Contributing

---

## Technologies Used

* Python
* NumPy
* pandas
* scikit-learn
* PySINDy
* Matplotlib

---

## Team Members

Developed by:

* Philip Nguyen - nguyphi4@oregonstate.edu
* Phuc Tran -tranchau@oregonstate.edu
* Prathmesh Nitin - gitep@oregonstate.edu

Mentor by:

* Ramon Ken Yoshiura (INL) - ramon.yoshiura@inl.gov


GitHub profiles:

* [https://github.com/thegooda](https://github.com/thegooda)
* [https://github.com/Phuctran24102002](https://github.com/Phuctran24102002)
* [https://github.com/PNG5042](https://github.com/PNG5042)

---

## Contact / Feedback

For questions, collaboration, or feedback:

* Open a GitHub Issue
* Contact the project team through GitHub or Email

---

## Future Work

Potential extensions include:

* Expanding to additional alloy systems
* Adding symbolic regression comparison
* Applying the framework to time-dependent constitutive modeling
* Improving feature selection under strong collinearity

---

## License

This repository is intended for academic and research purposes.
