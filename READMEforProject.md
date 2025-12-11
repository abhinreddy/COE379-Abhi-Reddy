# Used Car Price Prediction Proejct 3

My project trains and compares several classical machine learning models to predict used car selling prices from basic listing attributes (age/year, mileage, fuel type, transmission, seller type, ownership history, and brand).

## Files
- `Initial Proposal.pdf` - the intial proposal for the project.
- `AbhiReddyProject3Notebook.ipynb` — the main notebook.
- `Use_of_AI.md` — a log of AI-assisted code generation, with numbered entries referenced inside the notebook.
- `Predicting Used Car Prices with Classical Machine Learning Report.pdf` - the final report for the project.

## How to run (Google Colab)
1. Open `AbhiReddyProject3Notebook.ipynb`
2. Download the CarDekho CSV from Kaggle to your computer. (CAR DETAILS FROM CAR DEKHO.csv) https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho?resource=download
3. Run the dataset upload cell and upload the CSV when prompted.
4. Run the remaining cells top-to-bottom.

## Outputs
The notebook produces:
- A model comparison table (MAE / RMSE / R² on train and validation)
- Final test set evaluation for the best model
- Diagnostic plots (predicted vs actual; residuals)
- Top features (coefficients or importances)
- A saved model pipeline: `best_used_car_price_model.joblib`

