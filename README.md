# CC Underwriting API

Machine learning model for automated credit card underwriting decisions, deployed as a REST API on Azure.

## What It Does

A Random Forest classifier trained on 5,000 historical credit card applications with 176 engineered features. It accepts applicant data via a REST API and returns:

- **Approval Decision** (Approved / Declined)
- **Approval Probability** (0.0 - 1.0)
- **FICO-Style Score** (600 - 750)
- **Risk Band** (Very High / High / Medium / Low / Excellent)

**Model Performance:** AUC = 0.9955 | Gini = 0.9909 | F1 = 0.9621

---

## Architecture

```
Push to main
    |
    v
GitHub Actions CI/CD
    |
    +--> Install deps
    +--> Login to Azure (OIDC)
    +--> Set MLflow tracking URI
    +--> Train model (train.py)
    +--> Deploy to Azure Web App
    |
    v
FastAPI API (https://ccuw-api-mahi80.azurewebsites.net)
```

---

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

---

## Project Structure

```
.
+-- app.py                              # FastAPI inference server
+-- train.py                            # Model training pipeline
+-- requirements.txt                    # Python dependencies
+-- cc_underwriting_5k_stratified11.csv # Training data (5K records)
+-- model/
|   +-- rf/model.pkl                    # Trained Random Forest model
|   +-- features.json                   # 176 feature names
|   +-- scaler.json                     # StandardScaler params
|   +-- metrics.json                    # Training metrics
+-- .github/workflows/
    +-- deploy.yml                      # CI/CD pipeline
```

---

## Complete Setup Guide (Step-by-Step from Scratch)

This guide assumes you are starting from zero. Follow every step in order.

### Prerequisites

You need these tools installed on your machine:

```bash
# 1. Install Azure CLI
# Windows: Download from https://aka.ms/installazurecliwindows
# Mac:     brew install azure-cli
# Linux:   curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# 2. Install GitHub CLI
# Windows: winget install GitHub.cli
# Mac:     brew install gh
# Linux:   sudo apt install gh

# 3. Install Git
# Windows: https://git-scm.com/download/win
# Mac:     xcode-select --install
# Linux:   sudo apt install git

# 4. Install Python 3.11+
# Download from https://python.org/downloads
```

### Step 1: Login to Azure

Open your terminal and run:

```bash
# Login to Azure (this opens a browser window)
az login

# Verify you are in the right subscription
az account show --query "{name:name, subscriptionId:id, tenantId:tenantId}" -o table
```

You should see your subscription name, ID, and tenant ID. Note these down:
- **Subscription ID**: (e.g., 3a72be92-287b-4f1e-840a-5e3e71100139)
- **Tenant ID**: (e.g., 2b32b1fa-7899-482e-a6de-be99c0ff5516)

If you have multiple subscriptions, set the correct one:
```bash
az account set --subscription "YOUR_SUBSCRIPTION_ID"
```

### Step 2: Login to GitHub CLI

```bash
# Login to GitHub (this opens a browser window)
gh auth login

# Verify you are logged in
gh auth status
```

### Step 3: Create Azure Resource Group

A resource group is a container for all your Azure resources.

```bash
# Create resource group (change location if needed)
az group create --name zerotohero --location eastus

# Verify it exists
az group show --name zerotohero --query "{name:name, location:location}" -o table
```

### Step 4: Create App Service Plan

This is the virtual server that will run your web app.

```bash
# Create a B2 Linux plan (2 vCPUs, 3.5 GB RAM)
az appservice plan create \
  --name ccuw-plan \
  --resource-group zerotohero \
  --sku B2 \
  --is-linux \
  --location uksouth

# Verify it was created
az appservice plan show --name ccuw-plan --resource-group zerotohero \
  --query "{name:name, sku:sku.name}" -o table
```

> **Note:** If you get a quota error, try a different region: `--location westeurope` or `--location centralus`

### Step 5: Create Web App

This is the application that will host your API.

```bash
# Create the web app (name must be globally unique!)
az webapp create \
  --name ccuw-api-mahi80 \
  --resource-group zerotohero \
  --plan ccuw-plan \
  --runtime "PYTHON:3.11"

# Verify it was created
az webapp show --name ccuw-api-mahi80 --resource-group zerotohero \
  --query "{name:name, hostName:defaultHostName}" -o table
```

Your API will be at: `https://ccuw-api-mahi80.azurewebsites.net`

### Step 6: Create Azure ML Workspace (for MLflow Tracking)

This gives you a managed MLflow server to track model experiments.

**6a. Create storage account:**
```bash
az storage account create \
  --name ccuwmlflowstorage \
  --resource-group zerotohero \
  --location uksouth \
  --sku Standard_LRS
```

**6b. Create Key Vault:**
```bash
az keyvault create \
  --name ccuw-mlflow-kv \
  --resource-group zerotohero \
  --location uksouth
```

**6c. Create App Insights (via REST API):**
```bash
az rest --method PUT \
  --url "https://management.azure.com/subscriptions/YOUR_SUBSCRIPTION_ID/resourceGroups/zerotohero/providers/Microsoft.Insights/components/ccuw-mlflow-ai?api-version=2020-02-02" \
  --body '{"location":"uksouth","kind":"web","properties":{"Application_Type":"web"}}'
```

**6d. Create the ML Workspace (via ARM template):**

Save this as `ml_deploy.json`:
```json
{
  "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
  "contentVersion": "1.0.0.0",
  "resources": [
    {
      "type": "Microsoft.MachineLearningServices/workspaces",
      "apiVersion": "2024-04-01",
      "name": "ccuw-mlflow",
      "location": "uksouth",
      "identity": { "type": "SystemAssigned" },
      "properties": {
        "storageAccount": "/subscriptions/YOUR_SUBSCRIPTION_ID/resourceGroups/zerotohero/providers/Microsoft.Storage/storageAccounts/ccuwmlflowstorage",
        "keyVault": "/subscriptions/YOUR_SUBSCRIPTION_ID/resourceGroups/zerotohero/providers/Microsoft.KeyVault/vaults/ccuw-mlflow-kv",
        "applicationInsights": "/subscriptions/YOUR_SUBSCRIPTION_ID/resourceGroups/zerotohero/providers/microsoft.insights/components/ccuw-mlflow-ai",
        "friendlyName": "ccuw-mlflow"
      }
    }
  ]
}
```

Then deploy it:
```bash
az deployment group create \
  --resource-group zerotohero \
  --template-file ml_deploy.json
```

**6e. Verify:**
```bash
az rest --method GET \
  --url "https://management.azure.com/subscriptions/YOUR_SUBSCRIPTION_ID/resourceGroups/zerotohero/providers/Microsoft.MachineLearningServices/workspaces/ccuw-mlflow?api-version=2024-04-01" \
  --query "{name:name, mlFlowTrackingUri:properties.mlFlowTrackingUri}" -o table
```

### Step 7: Set Up OIDC Authentication (GitHub to Azure)

This allows GitHub Actions to deploy to Azure without storing passwords.

**7a. Create Azure AD App Registration:**
```bash
CLIENT_ID=$(az ad app create --display-name github-ccuw --query appId -o tsv)
echo "Your Client ID: $CLIENT_ID"
# SAVE THIS - you need it for GitHub secrets!
```

**7b. Create Federated Credential:**
```bash
az ad app federated-credential create \
  --id "$CLIENT_ID" \
  --parameters '{
    "name": "github-main",
    "issuer": "https://token.actions.githubusercontent.com",
    "subject": "repo:mahi80/simplewebapp:ref:refs/heads/main",
    "audiences": ["api://AzureADTokenExchange"]
  }'
```

**7c. Create Service Principal:**
```bash
az ad sp create --id "$CLIENT_ID"
```

**7d. Get Service Principal Object ID:**
```bash
SP_OBJ_ID=$(az ad sp show --id "$CLIENT_ID" --query id -o tsv)
echo "SP Object ID: $SP_OBJ_ID"
```

**7e. Assign Contributor role:**
```bash
ROLE_ASSIGNMENT_ID=$(python -c "import uuid; print(uuid.uuid4())")

az rest --method PUT \
  --url "https://management.azure.com/subscriptions/YOUR_SUBSCRIPTION_ID/resourceGroups/zerotohero/providers/Microsoft.Authorization/roleAssignments/${ROLE_ASSIGNMENT_ID}?api-version=2022-04-01" \
  --body "{\"properties\":{\"roleDefinitionId\":\"/subscriptions/YOUR_SUBSCRIPTION_ID/providers/Microsoft.Authorization/roleDefinitions/b24988ac-6180-42a0-ab88-20f7382dd24c\",\"principalId\":\"${SP_OBJ_ID}\",\"principalType\":\"ServicePrincipal\"}}"
```

### Step 8: Clone Repo and Add Code

```bash
# Clone the repository
git clone https://github.com/mahi80/simplewebapp.git
cd simplewebapp

# Unzip ccuw-simple.zip and copy files
# (Replace /path/to/ with actual path to your zip)
unzip /path/to/ccuw-simple.zip -d /tmp/ccuw-simple
cp -r /tmp/ccuw-simple/ccuw-simple/* .
cp -r /tmp/ccuw-simple/ccuw-simple/.github .
cp /tmp/ccuw-simple/ccuw-simple/.gitignore .
```

### Step 9: Set GitHub Secrets

```bash
gh secret set AZURE_CLIENT_ID         --repo mahi80/simplewebapp --body "YOUR_CLIENT_ID"
gh secret set AZURE_TENANT_ID         --repo mahi80/simplewebapp --body "YOUR_TENANT_ID"
gh secret set AZURE_SUBSCRIPTION_ID   --repo mahi80/simplewebapp --body "YOUR_SUBSCRIPTION_ID"
gh secret set AZURE_WEBAPP_NAME       --repo mahi80/simplewebapp --body "ccuw-api-mahi80"
gh secret set AZURE_ML_WORKSPACE      --repo mahi80/simplewebapp --body "ccuw-mlflow"
gh secret set AZURE_ML_RESOURCE_GROUP --repo mahi80/simplewebapp --body "zerotohero"

# Verify all 6 secrets are set
gh secret list --repo mahi80/simplewebapp
```

### Step 10: Push and Deploy

```bash
git add .
git commit -m "feat: CC underwriting model + Azure Web App deploy"
git push origin main
```

Watch the deployment:
```bash
gh run watch --repo mahi80/simplewebapp
# Or visit: https://github.com/mahi80/simplewebapp/actions
```

### Step 11: Verify Everything Works

```bash
# Health check
curl https://ccuw-api-mahi80.azurewebsites.net/health

# Model info
curl https://ccuw-api-mahi80.azurewebsites.net/model

# Test prediction
curl -X POST https://ccuw-api-mahi80.azurewebsites.net/predict \
  -H "Content-Type: application/json" \
  -d '{"annual_income": 75000, "fico_score": 720, "age": 35}'

# MLflow: https://ml.azure.com -> Experiments -> cc-underwriting
```

---

## Azure Resources Summary

| Resource | Name | Purpose |
|----------|------|---------|
| Resource Group | `zerotohero` | Container for all resources |
| App Service Plan | `ccuw-plan` (B2 Linux) | Compute for the web app |
| Web App | `ccuw-api-mahi80` (Python 3.11) | Hosts the FastAPI API |
| Storage Account | `ccuwmlflowstorage` | Storage for ML workspace |
| Key Vault | `ccuw-mlflow-kv` | Secrets for ML workspace |
| App Insights | `ccuw-mlflow-ai` | Monitoring for ML workspace |
| ML Workspace | `ccuw-mlflow` | MLflow experiment tracking |
| App Registration | `github-ccuw` | OIDC auth for GitHub Actions |

## GitHub Secrets Summary

| Secret | Purpose |
|--------|---------|
| `AZURE_CLIENT_ID` | OIDC app registration ID |
| `AZURE_TENANT_ID` | Azure AD tenant |
| `AZURE_SUBSCRIPTION_ID` | Azure subscription |
| `AZURE_WEBAPP_NAME` | Target Web App name |
| `AZURE_ML_WORKSPACE` | MLflow workspace name |
| `AZURE_ML_RESOURCE_GROUP` | ML workspace resource group |

## CI/CD Pipeline Steps

1. Push to `main` triggers GitHub Actions
2. Checks out code
3. Sets up Python 3.11
4. Installs dependencies
5. Logs into Azure using OIDC (passwordless)
6. Installs Azure ML CLI extension
7. Gets MLflow tracking URI from Azure ML workspace
8. Trains model - logs metrics to Azure ML
9. Deploys to Azure Web App
10. API goes live

## Local Development

```bash
pip install -r requirements.txt
python train.py          # Train model (logs to local mlruns/)
uvicorn app:app --reload # Start API at http://localhost:8000
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Quota error on App Service Plan | Try a different region: `--location westeurope` |
| OIDC login fails in GitHub Actions | Verify federated credential subject matches your repo |
| MLflow tracking fails | Ensure `azureml-mlflow` is in requirements.txt |
| `model/rf already exists` | Fixed in train.py - it clears the dir before saving |
| `parse_version` import error | Use `imbalanced-learn>=0.12.3` |
