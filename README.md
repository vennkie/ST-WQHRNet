# 🌊 ST-WQHRNet: A Spatio-Temporal Transformer for Multi-Task Groundwater Quality and Health Risk Assessment in Tamil Nadu

> **Product link:** https://stwqhrnettamiladu.pythonanywhere.com/

---

## 🌐 Project Overview

Groundwater quality degradation poses significant risks to public health, agricultural sustainability, and environmental stability — particularly in Tamil Nadu, where rapid urbanization, industrial discharge, and intensive agriculture continuously degrade groundwater across districts, blocks, and villages.

This project proposes **ST-WQHRNet**, a **Physics-Informed Spatio-Temporal Deep Learning Framework** for multi-level groundwater risk assessment and forecasting across **31 districts of Tamil Nadu**. Unlike traditional static models, ST-WQHRNet models contamination evolution through time and space using a Transformer-based architecture with physics-guided constraints.

The framework delivers:
- **Forecast** Water Quality Index (WQI) and Health Hazard Index (HHI) at district, block, and village scales
- **Capture** long-range seasonal contamination dynamics via spatio-temporal Transformer encoding
- **Enforce** hydrogeological plausibility through physics-based temporal smoothness and spatial Gaussian regularization
- **Classify** groundwater into Excellent / Good / Poor / Very Poor / Unsuitable (WQI) and Low / Medium / High / Severe (HHI)
- **Provide** actionable, policy-aligned risk insights for water resource authorities and public health planning

---

## 📦 Installation

**1. Clone or download the project**

**2. Install Python dependencies:**
```bash
pip install -r requirements.txt
```

**3. Run data preprocessing and tensor construction:**
```bash
python data_preprocessing.py
```

**4. Train the ST-WQHRNet model:**
```bash
python train.py
```

**5. Start the Flask application:**
```bash
python app.py
```

**6. Open browser and navigate to:**
```
http://localhost:5000
```

### Optional: Enable Groq AI Explainability in Analysis Tab

Set these environment variables to enable AI-generated narrative insights per district (Possible Health Impacts, Vulnerable Population, Recommended Action, Recommended Health Intervention):

**Windows PowerShell (current session):**
```powershell
$env:GROQ_API_KEY = "<your_api_key>"
$env:GROQ_API_URL = "https://your-groq-endpoint"  # POST endpoint returning JSON
python app.py
```

**Make it permanent:**
```powershell
setx GROQ_API_KEY "<your_api_key>"
setx GROQ_API_URL "https://your-groq-endpoint"
```

**Expected JSON response keys from your endpoint:**
```
possible_impacts                (string or [string])
vulnerable_population           (string or [string])
recommended_action              (string or [string])
recommended_health_intervention (string or [string])
```

---

## 📁 Project Structure

```
ST-WQHRNet/
├── model.py                     # ST-WQHRNet architecture (Transformer + Physics layers)
├── train.py                     # Model training pipeline (staged WQI → WQI+HHI)
├── data_preprocessing.py        # Data loading, encoding, tensor construction
├── predict.py                   # Inference engine for district/block/village forecasting
├── evaluate.py                  # Metrics: Accuracy, Macro F1, RMSE, R², NSE
├── app.py                       # Flask web application
├── config.yaml                  # Hyperparameter and experiment configuration
├── requirements.txt             # Python dependencies
├── templates/
│   └── dashboard.html           # Tamil Nadu GIS dashboard UI
├── static/
│   ├── css/
│   │   └── style.css            # Dashboard styling
│   └── js/
│       └── dashboard.js         # Interactive map and chart logic
├── models/
│   └── stwqhrnet_best.pt        # Best saved model checkpoint
├── data/
│   ├── groundwater_tn.csv       # Primary spatio-temporal dataset (2018–2025)
│   └── metadata.json            # Vocab maps, class weights, feature indices
└── outputs/
    ├── confusion_matrix_wqi.png
    ├── confusion_matrix_hhi.png
    └── loss_curves.png
```

---

## 🏗️ System Architecture

The ST-WQHRNet architecture is a layered deep learning pipeline consisting of three major stages:

**Stage I — Representation Learning & Feature Embedding:** Chemical parameters, spatio-temporal metadata, and environmental context are separately encoded and fused into a unified latent representation using hierarchical embeddings, positional encoding, and denoising autoencoders.

**Stage II — Spatio-Temporal & Physics-Guided Modeling:** The unified embedding is processed by a multi-head self-attention Transformer encoder with causal masking, followed by a Contamination State Modeling Module and a Physical Consistency Module enforcing temporal smoothness and spatial Gaussian diffusion constraints.

**Stage III — Multi-Task Risk Assessment:** Physically consistent contamination states feed into parallel WQI and HHI classification heads trained jointly using a weighted multi-task cross-entropy loss.

![ST-WQHRNet Architecture](https://github.com/user-attachments/assets/119765ff-c1e0-48d8-ae57-0f5853085f18)

---

## 🔬 Methodology

### Step 1 — Data Collection & Synthesis

A structured, multi-year, seasonally continuous groundwater dataset was developed using a hybrid approach combining hydrogeological domain knowledge with **Generative AI–based statistical simulation**. It covers:
- **31 districts** of Tamil Nadu at district → block → village hierarchy
- **4 seasons** per year: Pre-Monsoon, SW Monsoon, NE Monsoon, Post-Monsoon
- **Temporal range:** 2018–2025 (32 seasonal time steps)

---

### Step 2 — Data Preprocessing & Tensor Construction

- Parsing of chemical exceedance values into numerical format
- Encoding of categorical features (district, block, village, season, source type)
- Normalization using training-year statistics to prevent data leakage
- Construction of spatio-temporal tensors of shape **(Locations × Time Steps × Features)**
- Compressed `.npz` storage with JSON metadata for reproducible training

---

### Step 3 — WQI Classification Formula

For each water sample with parameter *j*:

```
Sub-index:    Qj = (Cj / Sj) × 100

WQI:          WQIi = Σ(Wj × Qj) / Σ(Wj)
```

| WQI Range  | Water Quality Category  |
|------------|--------------------------|
| 0 – 50     | ✅ Excellent / Good       |
| 51 – 75    | 🟡 Moderate               |
| 76 – 100   | 🟠 Poor                   |
| > 100      | 🔴 Very Poor / Unsuitable |

---

### Step 4 — HHI Computation Formula

```
HHIi = Σ (Cij / RfDj)
```

Where `RfDj` = reference dose / maximum safe exposure for contaminant *j* (as per WHO / BIS standards).

| HHI Range   | Health Risk Level |
|-------------|-------------------|
| < 0.15      | 🟢 Low             |
| 0.15 – 0.25 | 🟡 Moderate        |
| > 0.25      | 🔴 High / Severe   |

---

### Step 5 — Physics-Guided Spatio-Temporal Learning

**Layer 1: Numerical Projection**
```
Znum = Wnum · Xnum + bnum
```

**Layer 2–4: Hierarchical Spatial + Seasonal + Source Embeddings**
```
Espatial = Ed + Eb + Ev          (district + block + village)
Eseason  = Embedding(season_id)
Esource  = Embedding(source_id)
```

**Layer 5: Feature Fusion with Positional Encoding**
```
Zt = Znum + Espatial + Eseason + Esource + PE(t)
```

**Layer 6: Multi-Head Self-Attention (Causal Transformer)**
```
Attention(Q, K, V) = softmax( Q·Kᵀ / √dk ) · V
```

**Layer 8: Contamination State Projection**
```
Sl,t = Ws · Zl,t + bs
```

**Layer 9: Temporal Smoothness Constraint**
```
S'l,t = α · Sl,t + (1 − α) · Sl,t−1
```

**Layer 10: Spatial Gaussian Diffusion Constraint**
```
S''l,t = Σ wlj · S'j,t       where wlj = exp(−d²lj / 2σ²)
```

**Layer 11: Multi-Task Classification**
```
Pwqi = softmax(Wwqi · S'' + bwqi)
Phhi = softmax(Whhi · S'' + bhhi)

Loss = λ1 · CE(ywqi, ŷwqi) + λ2 · CE(yhhi, ŷhhi)
```

---

### Step 6 — Training Strategy

- **Stage 1:** WQI pretraining — model learns core contamination patterns first
- **Stage 2:** Joint WQI + HHI training with weighted multi-task loss
- **Optimizer:** AdamW with weight decay
- **Regularization:** Gradient clipping, early stopping, class-balanced focal loss
- **Evaluation:** Accuracy, Macro F1-Score, RMSE, R², Nash–Sutcliffe Efficiency (NSE)

---

## 📊 Dataset

### Primary Dataset

| Field               | Details                                                                                       |
|---------------------|-----------------------------------------------------------------------------------------------|
| **Source**          | Synthetic dataset generated using Generative AI + hydrogeological domain constraints         |
| **Coverage**        | 31 districts of Tamil Nadu — district → block → village hierarchy                            |
| **Year Range**      | 2018–2025                                                                                     |
| **Seasons**         | Pre-Monsoon, SW Monsoon, NE Monsoon, Post-Monsoon                                            |
| **Total Records**   | 348,800                                                                                       |
| **Time Steps**      | 32 (4 seasons × 8 years)                                                                     |
| **Key Parameters**  | pH, TDS, EC, DO, Nitrate, Fluoride, Iron, Alkalinity, Heavy Metals, Chloride, Hardness       |
| **Env. Features**   | Seasonal Rainfall (mm), Average Temperature (°C), Population Density, Agricultural Intensity |
| **WQI Classes**     | Excellent, Good, Poor, Very Poor, Unsuitable                                                  |
| **HHI Levels**      | Low, Medium, High, Severe                                                                     |

### External Reference Standards

| Dataset                               | Purpose                                                |
|---------------------------------------|--------------------------------------------------------|
| WHO Drinking Water Quality Guidelines | Permissible limits for WQI/HHI threshold definition    |
| Bureau of Indian Standards (BIS)      | National drinking water quality standards              |
| CGWB Groundwater Quality Reports      | Cross-reference for parameter distribution validation  |
| Tamil Nadu Climate / Rainfall Records | Seasonal rainfall and temperature constraint modeling  |
| Block-wise Census / Admin Boundaries  | Hierarchical spatial segmentation (district–block–village) |

### Derived Outputs

| Output                    | Description                                                        |
|---------------------------|--------------------------------------------------------------------|
| WQI Class                 | Multi-class groundwater suitability label (5 categories)          |
| HHI Level                 | Health risk classification based on exceedance-ratio modeling     |
| Contamination State (S)   | Latent physics-consistent intensity representation per location/season |
| Seasonal Time Index       | Year × Season encoding for causal temporal modeling               |
| Spatial Embedding ID      | Hierarchical district–block–village encoding                      |

---

## 🧩 Modules

### Module 1 — Feature Embedding Module

Converts heterogeneous groundwater attributes into a unified latent representation for spatio-temporal learning:
- Hierarchical embeddings encode district, block, and village to capture geographic structure
- Seasonal and source-type embeddings represent temporal variation and water source characteristics
- Chemical and environmental numerical features are projected into a shared latent space via dense layers
- Latitude and longitude encoding preserves spatial continuity and regional heterogeneity
- A denoising autoencoder pre-trains chemical parameter embeddings for noise robustness

---

### Module 2 — Spatio-Temporal Transformer Module

Models seasonal groundwater dynamics using multi-head self-attention with causal masking:
- Captures long-range temporal dependencies while preserving chronological observation order
- Spatial position embeddings account for regional variability across locations
- Feed-forward sub-layers with layer normalization apply non-linear feature transformation
- The resulting contextual representations encode evolving contamination patterns across seasons and years

---

### Module 3 — Contamination State Modeling Module

Projects Transformer outputs into a compact latent contamination intensity variable:
- Transforms high-dimensional spatio-temporal features into a continuous 1D contamination score per location/season
- Captures both short-term seasonal fluctuations and long-term contamination persistence
- Reflects the combined influence of chemical parameters, climatic conditions, and spatial factors
- Shared representation feeds into both WQI and HHI prediction heads

---

### Module 4 — Physical Consistency Module

Enforces hydrogeological plausibility through two mathematical constraints:
- **Temporal Smoothness:** Recursive formulation ensures gradual contamination transitions across consecutive seasons
- **Spatial Gaussian Regularization:** Geographic proximity weights maintain consistency between neighboring districts/blocks
- Together, these constraints align predictions with real-world groundwater flow and diffusion principles
- A physics-guided loss term (`L_phys`) is added to the training objective during joint optimization

---

### Module 5 — Latent Contamination Aggregation Module

Aggregates physically consistent contamination states into structured intensity vectors:
- Attention-based pooling integrates seasonal signals and long-term trends into a stable summary vector
- Captures contamination persistence, accumulation, and gradual variation
- Contextual auxiliary features (rainfall, population density) are fused via a non-linear projection layer
- Outputs a unified enriched latent vector `Z` for the downstream risk assessment heads

---

### Module 6 — Risk Assessment Module

Converts contamination states into normalized, actionable risk scores:
- Feature focus mechanism selects the most discriminative latent dimensions
- Non-linear mapping produces normalized risk scores in [0.0 – 1.0]
- Categorical risk level (`R_cat`) and normalized risk score (`R_score`) are computed in parallel
- Threshold-aligned boundaries ensure regulatory interpretability of outputs

---

### Module 7 — WQI and HHI Prediction Module

Final parallel classification heads generating groundwater quality and health risk predictions:
- WQI head: 5-class classification (Excellent / Good / Poor / Very Poor / Unsuitable)
- HHI head: 4-class classification (Low / Medium / High / Severe)
- Weighted multi-task cross-entropy loss balances both objectives during joint training
- Outputs aligned with BIS/WHO regulatory thresholds for policy-ready reporting

---

### Module 8 — API Management Layer

| Endpoint          | Method | Function                                        |
|-------------------|--------|-------------------------------------------------|
| `/api/predict`    | POST   | Run WQI/HHI prediction for a given location    |
| `/api/forecast`   | POST   | Future-year forecast for district/block/village |
| `/api/upload`     | POST   | Upload new seasonal groundwater CSV             |
| `/api/stats`      | GET    | Fetch district-level statistics and summaries   |
| `/api/status`     | GET    | API health check                                |

- Built with **Flask** for modular request handling
- JSON-formatted responses with WQI class, HHI level, risk score, and trend data
- CSV log export supported for each forecast session

---

## 🛠️ Implementation

![Dashboard Screenshot](https://github.com/user-attachments/assets/9e7294f6-5109-4753-a766-d92cb168384a)

### Quality Monitoring Dashboard
- Interactive GIS choropleth map of Tamil Nadu with all **31 districts** color-coded by WQI risk level
- Risk legend: Excellent (green) → Good → Poor → Very Poor → Unsuitable (dark red)
- Click-to-select district functionality showing live WQI and HHI forecast values
- Toggle between Live System mode and Dark mode
- Groq AI integration for district-wise narrative health impact summaries

---

### AI Model Prediction Interface
Input parameters accepted (hierarchical location + year selection):
- State → District → Block → Village → Year (forecast target)

Sample output:
```
WQI Score:              76.00 – 82.00 (Very Poor)
Health Hazard Index:    1.00 (Low Risk)
Historical WQI Trend:   ±5-year trend chart
Historical HHI Trend:   ±5-year trend chart
Download CSV Log:       Exportable forecast record
```

---

### Index Calculators (Manual Computation Tools)

**WQI Calculator** — accepts pH, DO, TDS, NO₃, Cl, F inputs → returns WQI score and class

**HHI Calculator** — accepts contaminant concentrations (As, Pb, Cd, Cr, Fe), exposure factors (IR, EF, ED, BW, AT), and RfD values → returns HHI score and risk level

Sample WQI output:
```
WQI: 51.00 – 59.36 → Poor
```

Sample HHI output:
```
HHI: 0.54 – 0.94 → Low Risk
```

---

### Health Hazard & Risk Visualization
- Choropleth overlay showing district-level WQI/HHI scores on Tamil Nadu GIS map
- Click any district to view: WQI score, WQI class, HHI score, HHI level
- Color gradient from green (safe) to red (high risk) for instant spatial insight
- Example: **Dindigul** → WQI: 76 (Very Poor) | HHI: 1 (Low Risk)

---

## 📈 Results and Discussion

![Results](https://github.com/user-attachments/assets/4b3923f8-6a2f-4b82-bdf3-c5e6de4d7e91)

### Model Performance

| Metric              | ST-WQHRNet (WQI) | ST-WQHRNet (HHI) |
|---------------------|------------------|------------------|
| **Accuracy**        | 93.1%            | 94.2%            |
| **Macro F1-Score**  | 0.889            | 0.918            |
| **Macro ROC-AUC**   | 0.953            | 0.953            |
| **R² Score**        | 0.79             | 0.83             |
| **RMSE**            | Low              | Low              |
| **NSE**             | Stable           | Stable           |

---

### District-wise Forecast (10-Year Window: 2026–2036, Centred on 2031)

| S.No | District          | Avg WQI Score | Avg HHI Score |
|------|-------------------|---------------|---------------|
| 1    | Ariyalur          | 21.46         | 0.537         |
| 2    | Coimbatore        | 25.07         | 0.627         |
| 3    | Cuddalore         | 21.60         | 0.540         |
| 4    | Dharmapuri        | 19.93         | 0.499         |
| 5    | Dindigul          | 21.29         | 0.532         |
| 6    | Erode             | 26.39         | 0.660         |
| 7    | Kancheepuram      | 19.37         | 0.484         |
| 8    | Kanniyakumari     | 22.02         | 0.550         |
| 9    | Karur             | 21.86         | 0.547         |
| 10   | Krishnagiri       | 21.79         | 0.545         |
| 11   | Madurai           | 22.87         | 0.571         |
| 12   | Nagapattinam      | 20.04         | 0.501         |
| 13   | Namakkal          | 20.87         | 0.521         |
| 14   | Perambalur        | 21.12         | 0.528         |
| 15   | Pudukkottai       | 21.03         | 0.526         |
| 16   | Ramanathapuram    | 20.47         | 0.511         |
| 17   | Salem             | 25.48         | 0.637         |
| 18   | Sivagangai        | 21.12         | 0.528         |
| 19   | Thanjavur         | 25.53         | 0.639         |
| 20   | The Nilgiris      | 22.06         | 0.551         |
| 21   | Theni             | 20.46         | 0.511         |
| 22   | Thoothukkudi      | 22.86         | 0.571         |
| 23   | Tiruchirappalli   | 21.74         | 0.543         |
| 24   | Tirunelveli       | 24.25         | 0.606         |
| 25   | Tiruppur          | 23.88         | 0.597         |
| 26   | Tiruvallur        | 22.78         | 0.570         |
| 27   | Tiruvannamalai    | 19.17         | 0.480         |
| 28   | Tiruvarur         | 26.37         | 0.660         |
| 29   | Vellore           | 26.41         | 0.660         |
| 30   | Vilupuram         | 20.81         | 0.520         |
| 31   | Virudhunagar      | 25.61         | 0.640         |

---

![Confusion Matrix WQI](https://github.com/user-attachments/assets/aad5a134-5603-4543-831a-f989e68a32f7)
![Confusion Matrix HHI](https://github.com/user-attachments/assets/a0530011-49f8-4661-b60f-480bb08e0a21)

### Key Findings

- **All 31 districts** fall within the "Excellent to Good" WQI category (WQI < 50), with a state-wide mean of **22.44**
- Despite good WQI, **all districts** register an HHI > 0.25 ("High concentration"), with a state-wide mean HHI of **0.561** — indicating widespread latent health risks
- **Highest HHI risk districts:** Vellore (0.660), Erode (0.660), Tiruvarur (0.660), Virudhunagar (0.640), Thanjavur (0.639)
- **Lowest HHI (safest) districts:** Tiruvannamalai (0.480), Kancheepuram (0.484), Dharmapuri (0.499)
- **Pre-monsoon season** consistently showed elevated contamination levels across districts
- The divergence between good WQI and high HHI reveals that even "drinkable" groundwater may carry long-term health hazards, highlighting the need for dual-index monitoring

---

## 🔁 Comparative Analysis

### Performance Across Models

| Metric                   | Logistic Regression | Random Forest | **ST-WQHRNet (Ours)** |
|--------------------------|:-------------------:|:-------------:|:---------------------:|
| Spatial Component        | ❌                  | ❌            | ✅                    |
| Temporal Component       | ❌                  | ❌            | ✅                    |
| Physics-Guided Component | ❌                  | ❌            | ✅                    |
| WQI Accuracy             | 0.812               | 0.874         | **0.931**             |
| WQI F1-Score             | 0.798               | 0.861         | **0.889**             |
| WQI ROC-AUC              | 0.846               | 0.901         | **0.953**             |
| HHI Accuracy             | 0.828               | 0.882         | **0.942**             |
| HHI F1-Score             | 0.815               | 0.869         | **0.918**             |
| HHI ROC-AUC              | 0.846               | 0.901         | **0.953**             |

### Comparison with Existing Approaches

| Approach                                            | WQI Pred. | HHI Pred. | Temporal | Spatial Hierarchy | Physics-Informed | Multi-Task |
|-----------------------------------------------------|:---------:|:---------:|:--------:|:-----------------:|:----------------:|:----------:|
| Manual / Lab Testing (Traditional)                  | ❌        | ❌        | ❌       | ❌                | ❌               | ❌         |
| Static ML (Logistic Regression / RF)                | ✅        | Partial   | ❌       | ❌                | ❌               | ❌         |
| LSTM / GRU Recurrent Models                         | ✅        | Partial   | ✅       | ❌                | ❌               | Partial    |
| Transformer w/o Physics (Alizadeh et al., 2024)     | ✅        | ✅        | ✅       | Partial           | ❌               | ✅         |
| Graph Neural Networks (Li et al., 2024)             | ✅        | ❌        | Partial  | ✅                | ❌               | ❌         |
| **ST-WQHRNet (Ours)**                               | ✅        | ✅        | ✅       | ✅                | ✅               | ✅         |

**Advantages of ST-WQHRNet:**
- Integrates WQI and HHI in a single unified multi-task pipeline
- Hierarchical district → block → village spatial embeddings capture geographic structure
- Physics-informed temporal smoothness and spatial Gaussian diffusion ensure hydrogeological plausibility
- Causal Transformer encoding preserves chronological contamination evolution
- Scalable to additional districts, states, contaminants, or real-time sensor feeds without redesigning the core system

---

## ⚙️ System Requirements

### Software Requirements

| Component              | Specification                                           |
|------------------------|---------------------------------------------------------|
| **OS**                 | Windows 10/11, Ubuntu 20.04+, macOS (64-bit)           |
| **Language**           | Python 3.9 or above                                     |
| **Deep Learning**      | PyTorch (GPU-accelerated)                               |
| **Data Processing**    | NumPy, Pandas                                           |
| **Visualization**      | Matplotlib                                              |
| **Configuration**      | PyYAML                                                  |
| **Web Framework**      | Flask                                                   |
| **Storage Format**     | NumPy `.npz` tensors, JSON metadata, YAML configs       |
| **GPU Support**        | CUDA Toolkit (optional, recommended)                    |
| **Version Control**    | Git                                                     |

### Hardware Requirements

| Component      | Minimum                        | Recommended                      |
|----------------|--------------------------------|----------------------------------|
| **CPU**        | Intel Core i5 / AMD equivalent | Intel Core i7 / AMD Ryzen 7+     |
| **GPU**        | NVIDIA CUDA GPU (4GB VRAM)     | GTX/RTX series (6GB+ VRAM)       |
| **RAM**        | 8 GB                           | 16 GB or higher                  |
| **Storage**    | 5 GB free (HDD)                | SSD (faster tensor I/O)          |
| **Network**    | Required for package install   | Required for cloud GPU execution |

---

## 🔮 Future Enhancements

- **Real-Time IoT Integration:** Connect to live groundwater sensor networks for dynamic model updating and early contamination alerts
- **Climate Projection Modeling:** Integrate rainfall variability, drought indices, and temperature trends for long-horizon forecasting under climate change scenarios
- **Pan-India Expansion:** Scale the hierarchical spatial embedding to multi-state groundwater monitoring using the same modular architecture
- **Explainability Modules:** Attention weight visualization, feature attribution maps, and contamination pathway explanation for policy transparency
- **Policy Decision-Support Dashboard:** Deploy as a full-scale governance tool with automated district-wise sustainability reports and health risk advisories

---

## 🎥 Implementation Video

[![Watch Demo on Google Drive](https://img.shields.io/badge/▶%20Watch%20Demo-Google%20Drive-blue?style=for-the-badge&logo=google-drive)](https://drive.google.com/file/d/13OqrraxRf4aajwzO4FncQ4xlKfB_1g-A/view?usp=sharing)
