# Predicting NBA Player Value

A machine learning pipeline to predict whether an NBA rookie will still be
active in the league 5 years later, combined with a Flask web app for
live predictions.

**Tech stack:** Flask · scikit-learn · pandas · seaborn · SciPy · joblib

## Approach

- **Dataset:** 1,340 NBA rookies × 21 columns (target: `TARGET_5Yrs`)
- **EDA:** distributions, NaN check, outlier detection (IQR)
- **Feature selection:** Mann-Whitney U tests + `SelectKBest` inside the pipeline
- **Modelling:** SVM with `StandardScaler`, optimized via `GridSearchCV`
  (recall as scoring metric)
- **Deployment:** Flask app loading the serialized model (`best_model.pkl`)

The objective was to **maximize recall** — missing a future star is more
costly than passing on a borderline candidate.

## Installation

```bash
git clone https://github.com/MEMAUDATA/nba-player-value.git
cd nba-player-value

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# Run the full analysis notebook
jupyter notebook nba.ipynb

# Launch the Flask app
python app.py
```

Then open `http://127.0.0.1:5000` to predict player value from stats.

## License

This project is licensed under the **MIT License** —
see the [LICENSE](LICENSE) file for details.

## Author

**Nicolas Vannson** — Hearing Data Scientist
[memaudata.github.io](https://memaudata.github.io)