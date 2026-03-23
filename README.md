# ST-WQHRNet

ST-WQHRNet is a spatio-temporal transformer model that predicts district-level Water Quality Index (WQI) classes and Human Health Impact (HHI) levels for Tamil Nadu groundwater data. It follows the 11-layer design you specified and includes a Flask web app for interactive predictions and trend visualization.

**Quick Start**

```powershell
python st_wqhrnet\data_prep.py
python st_wqhrnet\train.py
python st_wqhrnet\eval.py
python st_wqhrnet\predict.py --district "Ariyalur" --year 2030
```

**Web Application**

```powershell
python st_wqhrnet\app.py
```

Open:
```
http://127.0.0.1:5000
```

---

**Project Structure**

- `data_prep.py` preprocesses the CSV and creates `artifacts/`
- `model.py` defines the ST-WQHRNet layers
- `train.py` trains and saves the model to `outputs/best_model.pt`
- `eval.py` evaluates Accuracy and Macro-F1
- `predict.py` runs model-based predictions for a district/year
- `app.py` runs the Flask web app
- `static/` and `templates/` contain the UI

---

**Model Architecture (11 Layers)**

Layer 1 - Input Layer  
Input: Raw structured groundwater record  
Process: Accepts numeric and categorical features  
Output: Raw feature vector

Layer 2 - Embedding Layer  
Input: District, Block, Village, Season, Source Type  
Process: Converts categorical IDs into dense embeddings  
Output: Embedded feature representation

Layer 3 - Spatial Encoding Layer  
Input: Embeddings + Latitude/Longitude  
Process: Learns geographic relationships and spatial similarity  
Output: Spatial feature representation

Layer 4 - Temporal Encoding Layer  
Input: Spatial features ordered by time index  
Process: Adds seasonal order and time context  
Output: Time-aware feature sequence

Layer 5 - Spatio-Temporal Transformer Encoder  
Input: Time-aware spatial sequence  
Process: Multi-head attention across time and space  
Output: Context-aware spatio-temporal representation

Layer 6 - Contamination State Layer  
Input: Transformer output  
Process: Compresses features to a contamination severity signal  
Output: Latent contamination state

Layer 7 - Physical Consistency Layer  
Input: Current state + previous season + nearby states  
Process: Smooths unrealistic spikes across time and space  
Output: Physically consistent contamination state

Layer 8 - Risk Scoring Layer  
Input: Consistent contamination state  
Process: Normalizes severity to a risk score  
Output: Risk score (0 to 1)

Layer 9 - WQI Prediction Head  
Input: Risk score  
Process: Maps to WQI classes  
Output: Good / Moderate / Poor / Unfit

Layer 10 - HHI Prediction Head  
Input: Risk score + emerging contamination indicators  
Process: Maps to health impact levels  
Output: Low / Medium / High

Layer 11 - Forecasting Layer  
Input: Contamination history up to year t-1  
Process: Predicts future WQI and HHI for selected year  
Output: WQI(t) and HHI(t)

---

**Data and Preprocessing**

- CSV path is configured in `st_wqhrnet/config.yaml`
- Preprocessing creates:
  - `artifacts/data.npz` (model tensors)
  - `artifacts/meta.json` (vocabularies and mappings)

Splits:
- Train: 2015 to 2022
- Validation: 2023
- Test: 2024

Forecast years:
- 2025 to 2050

---

**Training**

```powershell
python st_wqhrnet\data_prep.py
python st_wqhrnet\train.py
```

Model checkpoint:
- `st_wqhrnet\outputs\best_model.pt`

---

**Evaluation Metrics**

`eval.py` now computes and saves:
- Accuracy (overall + per class, for WQI and HHI)
- Precision / Recall / F1 (per class)
- Confusion matrices (`.png` + `.npy`)
- ROC-AUC (macro OVR + per class, optional)
- Proxy regression metrics (RMSE, R2, NSE)

Run:

```powershell
python st_wqhrnet\eval.py
```

Skip ROC-AUC:

```powershell
python st_wqhrnet\eval.py --skip-roc
```

Saved outputs:
- `st_wqhrnet\outputs\eval_metrics.json`
- `st_wqhrnet\outputs\eval_summary.csv`
- `st_wqhrnet\outputs\eval_per_class.csv`
- `st_wqhrnet\outputs\wqi_confusion_matrix.png`
- `st_wqhrnet\outputs\hhi_confusion_matrix.png`
- `st_wqhrnet\outputs\wqi_confusion_matrix.npy`
- `st_wqhrnet\outputs\hhi_confusion_matrix.npy`

---

**Prediction**

Model-based predictions:

```powershell
python st_wqhrnet\predict.py --district "Ariyalur" --year 2030
```

This returns the most frequent WQI and HHI class across the district for the selected year and each season.

---

**Web App Behavior**

The web UI lets the user select:
- District
- Year (2025 to 2050)

It shows:
- WQI and HHI values with class labels
- 11-year trend charts (5 years before and after)
- WQI and HHI calculators under their respective charts

Important note about UI prediction mode:
- UI prediction behavior is controlled by `inference.use_jitter` in `st_wqhrnet/config.yaml`.
- `true` -> jittered synthetic-like outputs.
- `false` -> model-driven outputs.
- Restart Flask app after changing this flag.

---

**Configuration**

All parameters are in `st_wqhrnet/config.yaml`:

- Dataset paths
- Train/val/test years
- Model hyperparameters
- Training parameters

---

**Troubleshooting**

- If `train.py` fails on Windows with multiprocessing errors, set `num_workers: 0` in `config.yaml`.
- If you see no variation in UI predictions, confirm the selected district and year are being sent to `/api/predict`.


---

**Validation Baseline (Current Run)**

Config snapshot:
- `st_wqhrnet\outputs\validation_baseline_config.yaml`

Metrics (test split):
- WQI Overall Accuracy: `0.8295`
- WQI Macro F1: `0.3504`
- WQI Macro ROC-AUC: `0.8589`
- WQI RMSE: `13.5633`
- HHI Overall Accuracy: `0.9320`
- HHI Macro F1: `0.9141`
- HHI Macro ROC-AUC: `0.9945`
- HHI RMSE: `0.3947`

Training artifacts:
- `st_wqhrnet\outputs\best_model.pt`
- `st_wqhrnet\outputs\train_history.json`
- `st_wqhrnet\outputs\loss_curve.png`

python st_wqhrnet\eval.py --study-demo
python ANALYSIS\baseline_eval.py --study-demo
