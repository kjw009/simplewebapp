# CC Underwriting API

Machine learning model for automated credit card underwriting decisions, deployed as a REST API on Azure.

## What It Does

A Random Forest classifier trained on 5,000 historical credit card applications with 176 engineered features. It accepts applicant data via a REST API and returns:

- **Approval Decision** (Approved / Declined)
- **Approval Probability** (0.0 - 1.0)
- **FICO-Style Score** (600 - 750)
- **Risk Band** (Very High / High / Medium / Low / Excellent)

**Model Performance:** AUC = 0.9955 | Gini = 0.9909 | F1 = 0.9621

## Architecture

```
Push to main
    |
    v
GitHub Actions CI/CD
    |
    +--> Install deps
    +--> Login to Azure (OIDC)
    +--> Set MLflow tracking URI (Azure ML Studio)
    +--> Train model (train.py)
    +--> Deploy to Azure Web App
    |
    v
FastAPI API (https://ccuw-api-mahi80.azurewebsites.net)
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/model` | GET | Model metadata and metrics |
| `/predict` | POST | Underwriting prediction |

### Example: Predict

```bash
curl -X POST https://ccuw-api-mahi80.azurewebsites.net/predict \
  -H "Content-Type: application/json" \
  -d '{"annual_income": 75000, "fico_score": 720, "age": 35}'
```

Response:
```json
{
  "decision": "Approved",
  "probability": 0.87,
  "score": 715,
  "risk_band": "Low"
}
```

## Project Structure

```
.
├── app.py                              # FastAPI inference server
├── train.py                            # Model training pipeline
├── requirements.txt                    # Python dependencies
├── cc_underwriting_5k_stratified11.csv # Training data (5K records)
├── model/
│   ├── rf/model.pkl                    # Trained Random Forest model
│   ├── features.json                   # 176 feature names
│   ├── scaler.json                     # StandardScaler params
│   └── metrics.json                    # Training metrics
└── .github/workflows/
    └── deploy.yml                      # CI/CD pipeline
```

## Azure Resources

| Resource | Name | Purpose |
|----------|------|---------|
| Resource Group | `zerotohero` | Container for all resources |
| App Service Plan | `ccuw-plan` (B2 Linux) | Compute for the web app |
| Web App | `ccuw-api-mahi80` (Python 3.11) | Hosts the FastAPI API |
| ML Workspace | `ccuw-mlflow` | MLflow experiment tracking |
| App Registration | `github-ccuw` | OIDC auth for GitHub Actions |

## GitHub Secrets

| Secret | Purpose |
|--------|---------|
| `AZURE_CLIENT_ID` | OIDC app registration ID |
| `AZURE_TENANT_ID` | Azure AD tenant |
| `AZURE_SUBSCRIPTION_ID` | Azure subscription |
| `AZURE_WEBAPP_NAME` | Target Web App name |
| `AZURE_ML_WORKSPACE` | MLflow workspace name |
| `AZURE_ML_RESOURCE_GROUP` | ML workspace resource group |

## MLflow Experiment Tracking

Every push to main trains the model and logs metrics to Azure ML Studio:
- **Portal:** https://ml.azure.com → Experiments → `cc-underwriting`
- **Tracked Metrics:** AUC, Gini, F1, Accuracy, feature count, git SHA

## Local Development

```bash
pip install -r requirements.txt
python train.py          # Train model
uvicorn app:app --reload # Start API locally at http://localhost:8000
```

## Setup (From Scratch)

1. Create Azure resources (App Service Plan, Web App, ML Workspace)
2. Set up OIDC (App Registration + Federated Credential + Service Principal)
3. Assign Contributor role to the service principal on the resource group
4. Set 6 GitHub secrets listed above
5. Push to main — GitHub Actions handles the rest
