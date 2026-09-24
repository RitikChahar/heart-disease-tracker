# models/

This folder contains the trained machine learning model used by the Streamlit dashboard.

## Contents

| File | Description |
|------|-------------|
| `heart_disease_model.pkl` | Serialised scikit-learn `Pipeline` (StandardScaler + best classifier) |

## What is saved

The `.pkl` file is a Python `dict` with two keys:

```python
{
    "model_name": str,          # name of the winning algorithm, e.g. "Random Forest"
    "pipeline":   Pipeline,     # sklearn Pipeline: StandardScaler → Classifier
}
```

The `Pipeline` bundles the fitted `StandardScaler` and the best classifier together so that raw (unscaled) input can be passed directly to `pipeline.predict()` without a separate scaling step.

## How it was produced

Run the training script from the project root:

```bash
python train_and_save_model.py
```

The script:
1. Loads `heart.csv`
2. Performs an 80 / 20 stratified train–test split (`random_state=42`)
3. Trains four models: Logistic Regression, KNN, Decision Tree, Random Forest — each inside a `Pipeline` with `StandardScaler`
4. Evaluates all models on the test set and selects the one with the highest **F1-Score**
5. Saves the winning pipeline to `models/heart_disease_model.pkl`

## Regenerating the model

Delete `heart_disease_model.pkl` and re-run `python train_and_save_model.py` from the project root.  
The result is fully reproducible because `random_state=42` is fixed throughout.

## Notes

- The model file is tracked in version control so the Streamlit app works without re-running the training script.
- The model was trained on the **UCI Heart Disease (Cleveland) dataset** (`heart.csv`, 303 rows, 13 features).
- This model is for **academic / educational purposes only** and must not be used for clinical decisions.
