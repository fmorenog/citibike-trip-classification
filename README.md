# Classifying Commuter vs. Recreational Citi Bike Trips

**Programming Tools for Urban Analytics — Final Project**  
University of Glasgow | Dr Qunshan Zhao | Deadline: 30 March 2026

---

## Research Question

> *What trip-level features best predict whether a Citi Bike journey is made for commuting or recreational purposes, and what does the spatial and temporal distribution of classified trips reveal about urban mobility patterns in New York City?*

---

## Project Overview

This project builds a supervised machine learning pipeline to classify Citi Bike trips in New York City as either **commuter** or **recreational**. Because no ground-truth labels exist in the raw trip data, a multi-dimensional proxy heuristic combining departure time, trip duration, and station land use is used to generate labels. Three classifiers — Logistic Regression, Random Forest, and a Multi-Layer Perceptron — are trained and compared.

---

## Repository Structure

```
citibike/
├── data/
│   ├── raw/              # Downloaded source files (gitignored)
│   └── processed/        # Cleaned Parquet files (gitignored)
├── notebooks/
│   └── citibike_analysis.ipynb   # Main deliverable notebook
├── src/
│   ├── __init__.py
│   ├── data_download.py          # Phase 2: data collection scripts
│   ├── database.py               # Phase 3: DuckDB setup and queries
│   ├── pipeline.py               # Phase 6: TripClassifierPipeline OOP class
│   └── utils.py                  # Shared helper functions
├── .env.example                  # API key template (never commit .env)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Data Sources

| Source | Data | Access |
|--------|------|--------|
| [Citi Bike S3 (Lyft)](https://citibikenyc.com/system-data) | Historical trip records 2024 | Public CSV/Parquet download |
| [GBFS API](https://gbfs.lyft.com/gbfs/1.1/bkn/en/station_information.json) | Station metadata (lat, lon, capacity) | REST API (JSON) |
| [OpenWeatherMap API](https://openweathermap.org/api) | Hourly historical weather for NYC | REST API — requires free API key |
| [NYC PLUTO](https://www.nyc.gov/site/planning/data-maps/open-data/dwn-pluto-mappluto.page) | Land use by tax lot | NYC Open Data / CSV |

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/citibike-trip-classification.git
cd citibike-trip-classification
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate       # macOS/Linux
venv\Scripts\activate          # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your API key
```bash
cp .env.example .env
# Then edit .env and add your OpenWeatherMap API key
```

### 5. Run data collection
```bash
python src/data_download.py
```

### 6. Open the notebook
```bash
jupyter notebook notebooks/citibike_analysis.ipynb
```

---

## Methods Summary

1. **Proxy labeling** — Combined heuristic: rush-hour timing + trip duration 5–30 min + employment-anchored station land use
2. **Feature engineering** — Temporal, trip, spatial, member type, and weather features via `TripClassifierPipeline` OOP class
3. **Models** — Logistic Regression (baseline) → Random Forest (GridSearchCV) → MLP neural network
4. **Interpretability** — SHAP values on Random Forest; spatial misclassification map

---

## Course Tools Coverage

| Tool | Implementation |
|------|----------------|
| GitHub | This repository — version controlled throughout |
| Pandas | Data manipulation, merging, feature engineering |
| APIs | OpenWeatherMap + Citi Bike GBFS |
| Data extraction | S3 download, REST API calls, Socrata API |
| OOP | `TripClassifierPipeline` class in `src/pipeline.py` |
| Machine learning | Logistic Regression, Random Forest, GridSearchCV |
| Deep learning | MLP Classifier (sklearn) |
| Database | DuckDB — schema, SQL queries, Parquet I/O |

---

## Results

*(To be completed after modelling)*

---

## References

- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
- Chen, T. & Guestrin, C. (2016). XGBoost. *Proc. KDD*, 785–794.
- Fishman et al. (2014). Bike share's impact on car use. *Transportation Research Part D*, 31, 13–20.
- Lundberg, S. & Lee, S. (2017). A unified approach to interpreting model predictions. *NeurIPS*, 30.
- Ursaki, J. & Aultman-Hall, L. (2015). Quantifying the equity of bikeshare access. *Transportation Research Record*, 2512(1), 28–38.
- Xing et al. (2020). Exploring travel patterns of dockless bike-sharing. *Journal of Transport Geography*, 87.
