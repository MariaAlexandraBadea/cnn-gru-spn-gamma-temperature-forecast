# CNN-GRU-SPN Gamma for Joint Multi-Horizon Temperature Anomaly Forecasting

This repository contains the implementation and example outputs for a **joint multi-horizon probabilistic forecasting** model of **monthly temperature anomalies**.

---

## 📌 Project summary

**Model: CNN1D + skip connection + GRU encoder + conditional SPN-Gamma decoder**

- **Encoder**
  - 1D CNN with skip connection
  - GRU encoder to obtain a context embedding \(h(x)\)

- **Decoder (conditional SPN head)**
  - Root **Sum** over \(K\) components with neural gating \(\pi_k(h)\)
  - Each component is a **Product over horizons**, yielding a mixture-of-products
  - This induces **cross-horizon dependence** via the shared latent component \(k\)

- **Per horizon & component**
  - Signed magnitude model:
    - Bernoulli sign gate \(s_{k,h}(h)\)
    - Gamma mixtures for \(|y_h|\) with sign-specific parameters (pos/neg)
  - Designed to capture **asymmetry** and **heavy tails**

---

## 🧾 Dataset

We use the **Daily Temperature of Major Cities** dataset (public):

- Kaggle link: https://www.kaggle.com/datasets/sudalairajkumar/daily-temperature-of-major-cities

Daily records are aggregated into **monthly values**, then **monthly anomalies** are computed following the pipeline described below.

> Note: The dataset is hosted on Kaggle, so you may need to be logged in to download it.

---

## 🔧 Preprocessing pipeline (leakage-safe)

The pipeline is designed to avoid data leakage and preserve a correct forecasting protocol:

- Monthly aggregation is performed first.
- Anomalies are computed **per city** using a fixed climatology estimated **only from training years**.
- Input gaps are imputed **causally inside the past window only** (window-only, short-gap limited).
- Targets are **never imputed** (windows with missing targets are discarded).
- **Geographic generalization:** train/val/test are disjoint **by cities**.
- **Time splits** enforce horizon boundaries (`train_end`, `val_end`) using target-year constraints.

---

## ✅ Evaluation

We report both point and probabilistic metrics:

- **Point accuracy:** RMSE, MAE  
- **Probabilistic quality:** CRPS, NLL (per horizon)  
- **Multivariate dependence-aware scoring:** Energy Score  
  - computed for (i) joint sampling
  - and (ii) an independent-k baseline (per-horizon latent \(k\)) to isolate the effect of cross-horizon coupling

Additional robustness reporting:
- Results are shown globally on pooled test windows
- and grouped by decade/regime (first horizon year) for non-stationarity checks
- multiple stratified disjoint test sets are used for sensitivity analysis

---

## ▶️ How to run (Google Colab)

The code was developed and tested in **Google Colab**.
1. Open **Google Colab**
2. `File → Open notebook → GitHub`
3. Paste this repository URL:
   - https://github.com/MariaAlexandraBadea/cnn-gru-spn-gamma-temperature-forecast
4. Open the notebook/script and run the cells top-to-bottom
