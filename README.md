# Sensor Anomaly Detection API

A machine learning API that uses Isolation Forest to identify unusual
sensor observations from the NASA C-MAPSS FD001 dataset.

## Features
- Trains an Isolation Forest anomaly detection model.
- Analyzes training and test sensor observations.
- Saves the trained model and feature metadata.
- Provides REST API endpoints using FastAPI.
- Includes automated API tests.

## Dataset
NASA C-MAPSS FD001 turbofan engine simulation data.

Training observations: 20631

Test observations: 13096

Selected sensor features: 15

Training anomaly rate: 5.0%

Test anomaly rate: 1.05%

## API Endpoints

- GET /health — checks API health and model loading.
- POST /predict — predicts whether a sensor observation is normal or anomalous.

## Example Request

Send a JSON object to POST /predict with a `values` list containing exactly
15 numeric sensor values, in the feature order
listed in metadata.json.

Example structure:

    {
      "values": [0.0, 0.0, 0.0]
    }

The three values above illustrate the JSON structure only. Replace them with
exactly 15 numeric values for a valid request.

## Run Locally

Install the dependencies:

    pip install fastapi uvicorn scikit-learn pandas joblib

Start the API from this project directory:

    uvicorn app:app --reload

Open the interactive API documentation at:

    http://127.0.0.1:8000/docs

## Model Details

Algorithm: Isolation Forest

Number of estimators: 200

Contamination parameter: 0.05

Random state: 42

## Project Files

- app.py
- isolation_forest_model.joblib
- metadata.json
- results.json
- README.md

## Important Note

The anomaly rates are model-generated flags, not ground-truth accuracy
measurements. The test dataset does not provide row-by-row anomaly labels.
