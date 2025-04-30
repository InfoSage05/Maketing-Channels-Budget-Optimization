# Marketing Mix Model and Budget Optimization

## Problem Statement

This project aims to build a Marketing Mix Model (MMM) to understand the impact of various marketing channel investments on Gross Merchandise Value (GMV). Subsequently, it utilizes the model to optimize the marketing budget allocation across different channels (TV, Digital, Sponsorship, Content Marketing, Online Marketing, Affiliates, SEM, Radio, Other) to maximize the predicted GMV, given a fixed total budget.

## Data

The analysis uses data from an Excel file (`Media data-Sale Calendar-NPS Scores_Data (1).xlsx`), specifically the "Media Investment" sheet. This sheet contains monthly investment data (in INR Cr.) for various marketing channels spanning from July 2023 to June 2024.

The Gross Merchandise Value (GMV) data for the corresponding months is manually provided and combined with the media investment data.

## Methodology

1.  **Data Loading and Preparation**:
    * The "Media Investment" sheet is loaded using pandas.
    * Initial rows containing metadata are skipped.
    * Columns are renamed for clarity (e.g., 'TV', 'Digital', 'Sponsorship', etc.).
    * Investment values, originally in INR Cr., are converted to absolute numeric values (multiplied by 10^7).
    * Missing values in 'Radio' and 'Other' channels are filled (imputed with 1, likely representing minimal or baseline presence).
    * The GMV data is loaded into a separate DataFrame.
    * The GMV and Media Investment data are concatenated into a final DataFrame (`final_df`).

2.  **Feature Engineering**:
    * To account for diminishing returns and capture the non-linear relationship often observed between marketing spend and sales, the investment features (TV, Digital, etc.) are log-transformed.
    * The log-transformed investment features are then scaled using `MinMaxScaler` to bring them into a common range (0 to 1). This helps improve the stability and performance of the regression model.

3.  **Modeling**:
    * A Ridge Regression model is chosen to predict GMV based on the log-transformed and scaled marketing investments. Ridge Regression is suitable here as it helps prevent overfitting by adding L2 regularization, especially relevant when dealing with potentially multicollinear marketing channels.
    * The data is split into training and testing sets (80/20 split) to evaluate the model's performance on unseen data.
    * The model is trained on the training set.
    * Model performance is evaluated using:
        * Mean Squared Error (MSE)
        * R-squared (R²)
        * Mean Absolute Error (MAE)

4.  **Budget Optimization**:
    * The coefficients and intercept from the trained Ridge Regression model, which represent the relationship between the log-transformed/scaled spend and GMV, are extracted.
    * An optimization process (using `scipy.optimize.differential_evolution` and apparently an attempt with Gradient Ascent) is employed to determine the optimal allocation weights (`w`) for the total marketing budget across the 9 channels.
    * The objective function for the optimization aims to maximize the predicted GMV based on the model's learned relationship, subject to the constraint that the sum of allocation weights equals 1 (representing 100% of the budget).
    * The final optimized weights indicate the percentage of the total budget that should be allocated to each channel to maximize predicted GMV.
    * The notebook calculates the predicted GMV based on this optimal allocation and compares it to the original budget to estimate the potential Return on Investment (ROI) improvement.

## Results

* The Ridge Regression model is trained and evaluated (MSE, R2, MAE reported in the notebook).
* The optimization process yields optimal budget allocation weights for the marketing channels. *Note: The specific weights derived from the Differential Evolution method in the notebook show an almost equal allocation (approx. 11.1% each), which might require further investigation or tuning of the optimization parameters/objective function.*
* The notebook calculates the potential percentage increase in GMV (ROI improvement) achievable by shifting to the optimized budget allocation compared to the historical spend.

## How to Use

1.  Ensure the required libraries (pandas, numpy, scikit-learn, scipy, matplotlib) are installed.
2.  Place the data file (`Media data-Sale Calendar-NPS Scores_Data (1).xlsx`) in the same directory as the notebook or update the `file_path` variable.
3.  Run the notebook cells sequentially to load data, preprocess features, train the model, and perform budget optimization.
4.  Review the model evaluation metrics and the final optimal budget allocation percentages.
