import logging
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

class DataCleaningAgent:
    """AI agent responsible for cleaning and preprocessing data."""
    
    def __init__(self, log_level=logging.INFO):
        """Initialize the data cleaning agent."""
        # Set up logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('DataCleaningAgent')
        self.logger.info("Data cleaning agent initialized")
    
    def process(self, data):
        """Process and clean the input data."""
        if data is None or data.empty:
            self.logger.error("No data provided for cleaning")
            return None
        
        self.logger.info(f"Starting data cleaning process on dataset with shape {data.shape}")
        
        # Create a copy of the dataframe to avoid modifying the original
        df = data.copy()
        
        # 1. Handle duplicates
        original_rows = len(df)
        df = self.remove_duplicates(df)
        rows_after_dedup = len(df)
        self.logger.info(f"Removed {original_rows - rows_after_dedup} duplicate rows")
        
        # 2. Handle missing values
        df = self.handle_missing_values(df)
        
        # 3. Fix data types
        df = self.fix_data_types(df)
        
        # 4. Handle outliers
        df = self.handle_outliers(df)
        
        # 5. Feature engineering
        df = self.feature_engineering(df)
        
        self.logger.info(f"Data cleaning completed. Final shape: {df.shape}")
        return df
    
    def remove_duplicates(self, df):
        """Remove duplicate rows from the dataframe."""
        return df.drop_duplicates(keep='first')
    
    def handle_missing_values(self, df):
        """Handle missing values in the dataframe."""
        # Get missing value statistics before handling
        total_cells = np.product(df.shape)
        missing_cells = df.isnull().sum().sum()
        missing_percentage = (missing_cells / total_cells) * 100
        
        self.logger.info(f"Missing values: {missing_cells} cells ({missing_percentage:.2f}%)")
        
        # For each column, handle missing values based on the data type
        for column in df.columns:
            missing_count = df[column].isnull().sum()
            if missing_count == 0:
                continue
                
            column_type = df[column].dtype
            missing_ratio = missing_count / len(df)
            
            # If more than 50% of values are missing, drop the column
            if missing_ratio > 0.5:
                self.logger.info(f"Dropping column '{column}' with {missing_ratio:.2f}% missing values")
                df = df.drop(columns=[column])
                continue
            
            # Handle missing values based on column type
            if np.issubdtype(column_type, np.number):
                # For numeric columns, impute with median
                median_value = df[column].median()
                df[column] = df[column].fillna(median_value)
                self.logger.info(f"Filled {missing_count} missing values in '{column}' with median: {median_value}")
            
            elif column_type == 'object' or column_type == 'category':
                # For categorical columns, impute with mode
                mode_value = df[column].mode()[0]
                df[column] = df[column].fillna(mode_value)
                self.logger.info(f"Filled {missing_count} missing values in '{column}' with mode: {mode_value}")
            
            elif pd.api.types.is_datetime64_any_dtype(df[column]):
                # For datetime columns, forward fill (use previous value)
                df[column] = df[column].fillna(method='ffill')
                # If still missing values (at the beginning), backward fill
                df[column] = df[column].fillna(method='bfill')
                self.logger.info(f"Filled {missing_count} missing values in datetime column '{column}' with fill methods")
        
        return df
    
    def fix_data_types(self, df):
        """Fix data types in the dataframe."""
        # Find date-like columns that might be stored as strings
        for column in df.select_dtypes(include=['object']).columns:
            # Skip columns with too many unique values or too few values
            if df[column].nunique() > min(100, len(df) * 0.5) or df[column].nunique() < 3:
                continue
            
            # Try to convert to datetime
            try:
                test_conversion = pd.to_datetime(df[column].dropna().iloc[0:10])
                df[column] = pd.to_datetime(df[column], errors='coerce')
                self.logger.info(f"Converted column '{column}' to datetime")
            except (ValueError, TypeError):
                pass
        
        # Convert numeric strings to numbers
        for column in df.select_dtypes(include=['object']).columns:
            try:
                # Check if the column looks like a numeric column
                numeric_values = pd.to_numeric(df[column], errors='coerce')
                if numeric_values.notnull().sum() > 0.8 * len(df):
                    df[column] = numeric_values
                    self.logger.info(f"Converted column '{column}' to numeric")
            except:
                pass
        
        return df
    
    def handle_outliers(self, df):
        """Handle outliers in numeric columns using IQR method."""
        numeric_columns = df.select_dtypes(include=['number']).columns
        
        for column in numeric_columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
            
            if outliers > 0:
                self.logger.info(f"Found {outliers} outliers in column '{column}'")
                
                # Cap outliers instead of removing them
                df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
                self.logger.info(f"Capped outliers in column '{column}' to range [{lower_bound}, {upper_bound}]")
        
        return df
    
    def feature_engineering(self, df):
        """Perform basic feature engineering on the dataset."""
        # Extract date components from datetime columns
        for column in df.select_dtypes(include=['datetime64']).columns:
            df[f"{column}_year"] = df[column].dt.year
            df[f"{column}_month"] = df[column].dt.month
            df[f"{column}_day"] = df[column].dt.day
            df[f"{column}_dayofweek"] = df[column].dt.dayofweek
            self.logger.info(f"Created date features from column '{column}'")
        
        # Create interaction features for numeric columns (limited to prevent explosion)
        numeric_columns = df.select_dtypes(include=['number']).columns
        if len(numeric_columns) >= 2:
            important_numeric = numeric_columns[:5]  # Limit to first 5 numeric columns to avoid explosion
            for i in range(len(important_numeric)):
                for j in range(i+1, len(important_numeric)):
                    col1 = important_numeric[i]
                    col2 = important_numeric[j]
                    df[f"{col1}_times_{col2}"] = df[col1] * df[col2]
                    self.logger.info(f"Created interaction feature '{col1}_times_{col2}'")
        
        return df
    
    def answer_question(self, question, data):
        """Answer a natural language question about data cleaning."""
        if data is None:
            return "No dataset is currently loaded. Please load a dataset first."
        
        # Extract keywords from the question
        question = question.lower()
        
        # Questions about missing values
        if "missing" in question:
            missing_values = data.isnull().sum()
            missing_columns = missing_values[missing_values > 0]
            
            if len(missing_columns) == 0:
                return "There are no missing values in the dataset."
            
            result = "Missing values by column:\n"
            for column, count in missing_columns.items():
                percentage = (count / len(data)) * 100
                result += f"- {column}: {count} values ({percentage:.2f}%)\n"
            
            return result
        
        # Questions about duplicates
        elif "duplicate" in question:
            duplicate_count = len(data) - len(data.drop_duplicates())
            percentage = (duplicate_count / len(data)) * 100
            
            return f"There are {duplicate_count} duplicate rows ({percentage:.2f}% of the dataset)."
        
        # Questions about data types
        elif "data type" in question or "dtype" in question:
            result = "Data types by column:\n"
            for column, dtype in data.dtypes.items():
                result += f"- {column}: {dtype}\n"
            
            return result
        
        # Questions about outliers
        elif "outlier" in question:
            result = "Potential outliers in numeric columns:\n"
            numeric_columns = data.select_dtypes(include=['number']).columns
            
            for column in numeric_columns:
                Q1 = data[column].quantile(0.25)
                Q3 = data[column].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = ((data[column] < lower_bound) | (data[column] > upper_bound)).sum()
                percentage = (outliers / len(data)) * 100
                
                result += f"- {column}: {outliers} outliers ({percentage:.2f}%)\n"
            
            return result
        
        # Default response
        else:
            return "I can answer questions about missing values, duplicates, data types, or outliers in your dataset. Please specify what you'd like to know."

if __name__ == "__main__":
    # Example usage
    agent = DataCleaningAgent()
    
    # Create a sample dataframe with issues
    data = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5, 1],
        'B': ['x', 'y', 'z', 'x', None, 'x'],
        'C': pd.date_range('2020-01-01', periods=6)
    })
    
    # Process the data
    cleaned_data = agent.process(data)
    print(cleaned_data) 