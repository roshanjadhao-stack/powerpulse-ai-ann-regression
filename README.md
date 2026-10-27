# PowerPulse AI — ANN Regression for Power Plant Output

An interactive deep-learning web application that estimates net electrical output (`PE`) from four environmental/operating inputs using a trained PyTorch Artificial Neural Network (ANN).

## Live demo
After deployment, add your Streamlit Community Cloud URL here.

## Project overview
- **Task:** Regression — estimate power plant net electrical output.
- **Model:** PyTorch ANN with architecture `4 → 6 → 6 → 1`, ReLU hidden activations.
- **Features:** Ambient Temperature (`AT`), Exhaust Vacuum (`V`), Ambient Pressure (`AP`), Relative Humidity (`RH`).
- **Target:** Net Electrical Output (`PE`).
- **Dataset:** `powerplant_data.csv` (9,568 rows, 5 columns).
- **Preprocessing:** `StandardScaler` fit on the training split only.
- **Evaluation:** 80/20 train-test split, `random_state=42`.

## Model evaluation
Metrics below are calculated using the included `best_model.pt` checkpoint and the reproduced 20% held-out test split. Model results can vary if the checkpoint or preprocessing changes.

| Metric | Result |
|---|---:|
| R² | 0.9320 |
| MAE | 3.4672 |
| RMSE | 4.4116 |
| MSE | 19.4621 |

## Features
- Interactive sliders for all four model inputs.
- Instant output prediction.
- Dataset preview, feature correlation and output distribution.
- Model metrics and ANN architecture view.
- Downloadable prediction report and source dataset.
- Responsive Streamlit interface.

## Repository structure
```text
.
├── app.py
├── ANN_Regression.ipynb
├── best_model.pt
├── scaler.pkl
├── model_metrics.json
├── powerplant_data.csv
├── requirements.txt
├── .gitignore
└── .streamlit/
    └── config.toml
```

## Run locally (Windows PowerShell)

1. Install Python 3.11 or 3.12 if not already installed.
2. Open PowerShell in this project folder.
3. Create and activate a virtual environment:

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, you can run the app directly without activating:

   ```powershell
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   .\.venv\Scripts\python.exe -m streamlit run app.py
   ```

4. Install dependencies:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. Launch the app:

   ```powershell
   python -m streamlit run app.py
   ```

Streamlit will print a local URL, typically `http://localhost:8501`.

## Deploy online with Streamlit Community Cloud

1. Create a new **public** GitHub repository, for example `powerpulse-ai-ann-regression`.
2. Upload/push all project files from this folder to the repository root.
3. Open https://share.streamlit.io/ and sign in with GitHub.
4. Select **Create app**, choose the repository and branch `main`.
5. Set the main file path to `app.py`, then deploy.
6. Wait for the build to finish and test the generated public app URL.

Keep `best_model.pt`, `scaler.pkl`, `powerplant_data.csv`, and `model_metrics.json` in the repository root, as the app expects these paths.

## Git commands (run inside this folder)

```powershell
git init
git add .
git commit -m "Build PowerPulse AI ANN regression web app"
git branch -M main
git remote add origin https://github.com/YOUR-GITHUB-USERNAME/powerpulse-ai-ann-regression.git
git push -u origin main
```

Replace `YOUR-GITHUB-USERNAME` with your GitHub username. If Git says the remote already exists, use `git remote set-url origin YOUR_REPOSITORY_URL`.

## LinkedIn project post template

🚀 Excited to share **PowerPulse AI — ANN-Based Power Plant Output Prediction**!

I transformed my JupyterLab deep-learning project into an interactive web application using PyTorch and Streamlit. The app estimates net electrical output from ambient temperature, exhaust vacuum, ambient pressure, and relative humidity.

**Highlights**
- Built an Artificial Neural Network using PyTorch
- Applied feature scaling with StandardScaler
- Evaluated the model using R², MAE, and RMSE
- Developed an interactive prediction dashboard with dataset exploration
- Prepared the project for GitHub and online deployment

🔗 Live Demo: [Add your deployed app URL]
💻 GitHub: [Add your repository URL]

Tech Stack: Python | PyTorch | Pandas | NumPy | scikit-learn | Streamlit

#MachineLearning #DeepLearning #ArtificialIntelligence #PyTorch #Streamlit #Python #DataScience

## Important note
This is an educational prediction demo. It is not intended for certified industrial control, safety decisions, or guaranteed operational forecasts.
