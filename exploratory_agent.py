import logging
import pandas as pd
import numpy as np
from scipy import stats

class ExploratoryAgent:
    """AI agent responsible for exploratory data analysis."""
    
    def __init__(self, log_level=logging.INFO):
        """Initialize the exploratory agent."""
        # Set up logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('ExploratoryAgent')
        self.logger.info("Exploratory agent initialized")
    
    def process(self, data):
        """Perform exploratory analysis on the input data."""
        if data is None or data.empty:
            self.logger.error("No data provided for exploration")
            return None
        
        self.logger.info(f"Starting exploratory analysis on dataset with shape {data.shape}")
        
        results = {
            "summary_statistics": self.get_summary_statistics(data),
            "correlation_analysis": self.analyze_correlations(data),
            "distribution_analysis": self.analyze_distributions(data),
            "categorical_analysis": self.analyze_categorical_variables(data),
            "time_series_analysis": self.analyze_time_series(data),
            "key_metrics": self.calculate_key_metrics(data)
        }
        
        self.logger.info("Exploratory analysis completed")
        return results
    
    def get_summary_statistics(self, data):
        """Calculate summary statistics for each column."""
        self.logger.info("Calculating summary statistics")
        
        summary = {}
        
        # For numeric columns
        numeric_cols = data.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            numeric_summary = data[numeric_cols].describe().transpose()
            # Add additional metrics
            numeric_summary['skewness'] = data[numeric_cols].skew()
            numeric_summary['kurtosis'] = data[numeric_cols].kurtosis()
            numeric_summary['missing_values'] = data[numeric_cols].isnull().sum()
            numeric_summary['missing_percent'] = (data[numeric_cols].isnull().sum() / len(data)) * 100
            
            # Convert to dictionary for easier serialization
            summary['numeric'] = numeric_summary.to_dict()
        
        # For categorical columns
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            categorical_summary = {}
            for col in categorical_cols:
                value_counts = data[col].value_counts().to_dict()
                unique_count = data[col].nunique()
                missing_values = data[col].isnull().sum()
                missing_percent = (missing_values / len(data)) * 100
                
                categorical_summary[col] = {
                    'unique_values': unique_count,
                    'missing_values': missing_values,
                    'missing_percent': missing_percent,
                    'top_values': dict(list(value_counts.items())[:5])  # Only include top 5 values
                }
            
            summary['categorical'] = categorical_summary
        
        # For datetime columns
        datetime_cols = data.select_dtypes(include=['datetime64']).columns
        if len(datetime_cols) > 0:
            datetime_summary = {}
            for col in datetime_cols:
                datetime_summary[col] = {
                    'min': data[col].min().strftime('%Y-%m-%d %H:%M:%S') if not pd.isna(data[col].min()) else None,
                    'max': data[col].max().strftime('%Y-%m-%d %H:%M:%S') if not pd.isna(data[col].max()) else None,
                    'range_days': (data[col].max() - data[col].min()).days if not pd.isna(data[col].min()) and not pd.isna(data[col].max()) else None,
                    'missing_values': data[col].isnull().sum(),
                    'missing_percent': (data[col].isnull().sum() / len(data)) * 100
                }
            
            summary['datetime'] = datetime_summary
        
        self.logger.info("Summary statistics calculation completed")
        return summary
    
    def analyze_correlations(self, data):
        """Analyze correlations between variables."""
        self.logger.info("Analyzing correlations")
        
        correlation_results = {}
        
        # For numeric columns
        numeric_cols = data.select_dtypes(include=['number']).columns
        if len(numeric_cols) >= 2:
            # Calculate Pearson correlation
            pearson_corr = data[numeric_cols].corr(method='pearson')
            
            # Find top positive and negative correlations
            correlations = pearson_corr.unstack().sort_values(ascending=False)
            # Remove self-correlations (which are always 1.0)
            correlations = correlations[correlations < 1.0]
            
            correlation_results['top_positive'] = correlations.head(10).to_dict()
            correlation_results['top_negative'] = correlations.tail(10).to_dict()
            
            # Calculate Spearman rank correlation (better for non-linear relationships)
            try:
                spearman_corr = data[numeric_cols].corr(method='spearman')
                spearman_correlations = spearman_corr.unstack().sort_values(ascending=False)
                spearman_correlations = spearman_correlations[spearman_correlations < 1.0]
                
                correlation_results['top_nonlinear'] = spearman_correlations.head(10).to_dict()
            except:
                self.logger.warning("Spearman correlation calculation failed")
        else:
            self.logger.info("Not enough numeric columns for correlation analysis")
            correlation_results['message'] = "Not enough numeric columns for correlation analysis"
        
        self.logger.info("Correlation analysis completed")
        return correlation_results
    
    def analyze_distributions(self, data):
        """Analyze the distributions of numeric variables."""
        self.logger.info("Analyzing distributions of numeric variables")
        
        distribution_results = {}
        
        numeric_cols = data.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            # Skip columns with too many missing values
            if data[col].isnull().sum() / len(data) > 0.5:
                continue
                
            # Basic distribution statistics
            distribution_results[col] = {
                'mean': data[col].mean(),
                'median': data[col].median(),
                'std': data[col].std(),
                'skewness': data[col].skew(),
                'kurtosis': data[col].kurtosis(),
                'min': data[col].min(),
                'max': data[col].max()
            }
            
            # Test for normality
            try:
                _, p_value = stats.shapiro(data[col].dropna())
                distribution_results[col]['is_normal'] = bool(p_value > 0.05)
                distribution_results[col]['normality_p_value'] = p_value
            except:
                self.logger.warning(f"Normality test failed for column {col}")
                distribution_results[col]['is_normal'] = None
                distribution_results[col]['normality_p_value'] = None
            
            # Calculate percentiles
            percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
            distribution_results[col]['percentiles'] = {
                f"p{p}": data[col].quantile(p/100) for p in percentiles
            }
        
        self.logger.info("Distribution analysis completed")
        return distribution_results
    
    def analyze_categorical_variables(self, data):
        """Analyze categorical variables."""
        self.logger.info("Analyzing categorical variables")
        
        categorical_results = {}
        
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            # Skip columns with too many unique values (might not be truly categorical)
            if data[col].nunique() > min(50, len(data) * 0.1):
                continue
                
            # Calculate frequency and proportion
            value_counts = data[col].value_counts()
            value_props = data[col].value_counts(normalize=True)
            
            # Limit to top categories
            top_n = min(10, len(value_counts))
            
            categorical_results[col] = {
                'unique_count': data[col].nunique(),
                'top_categories': {
                    'values': value_counts.head(top_n).to_dict(),
                    'proportions': value_props.head(top_n).to_dict()
                },
                'entropy': stats.entropy(value_props) if len(value_props) > 1 else 0
            }
            
            # Check if the column could be binary/boolean
            if data[col].nunique() == 2:
                categorical_results[col]['is_binary'] = True
                categorical_results[col]['binary_values'] = value_counts.index.tolist()
        
        self.logger.info("Categorical analysis completed")
        return categorical_results
    
    def analyze_time_series(self, data):
        """Analyze time series patterns if datetime columns exist."""
        self.logger.info("Analyzing time series patterns")
        
        time_series_results = {}
        
        # Look for datetime columns
        datetime_cols = data.select_dtypes(include=['datetime64']).columns
        if len(datetime_cols) == 0:
            self.logger.info("No datetime columns found for time series analysis")
            return {"message": "No datetime columns found for time series analysis"}
        
        # Analyze each datetime column
        for date_col in datetime_cols:
            # For each datetime column, try to identify numeric columns to analyze over time
            numeric_cols = data.select_dtypes(include=['number']).columns
            
            if len(numeric_cols) == 0:
                continue
                
            # Select a few important numeric columns for time analysis
            selected_numeric_cols = numeric_cols[:5]  # Limit to prevent excessive analysis
            
            time_series_results[date_col] = {}
            
            # Group by year and month
            try:
                data['temp_year'] = data[date_col].dt.year
                data['temp_month'] = data[date_col].dt.month
                
                # Analyze metrics by month
                monthly_trends = {}
                for metric in selected_numeric_cols:
                    monthly_agg = data.groupby(['temp_year', 'temp_month'])[metric].agg(['mean', 'count']).reset_index()
                    # Convert to simple dict format for easier serialization
                    monthly_trends[metric] = monthly_agg.to_dict(orient='records')
                
                time_series_results[date_col]['monthly_trends'] = monthly_trends
                
                # Clean up temporary columns
                data = data.drop(columns=['temp_year', 'temp_month'])
            except Exception as e:
                self.logger.warning(f"Monthly trend analysis failed for {date_col}: {str(e)}")
            
            # Analyze day of week patterns
            try:
                data['temp_dayofweek'] = data[date_col].dt.dayofweek
                
                dow_trends = {}
                for metric in selected_numeric_cols:
                    dow_agg = data.groupby('temp_dayofweek')[metric].agg(['mean', 'count']).reset_index()
                    dow_trends[metric] = dow_agg.to_dict(orient='records')
                
                time_series_results[date_col]['day_of_week_trends'] = dow_trends
                
                # Clean up temporary column
                data = data.drop(columns=['temp_dayofweek'])
            except Exception as e:
                self.logger.warning(f"Day of week analysis failed for {date_col}: {str(e)}")
        
        self.logger.info("Time series analysis completed")
        return time_series_results
    
    def calculate_key_metrics(self, data):
        """Calculate key business metrics from the data."""
        self.logger.info("Calculating key metrics")
        
        key_metrics = {}
        
        # Total count of records
        key_metrics['record_count'] = len(data)
        
        # If we have timestamp/date columns, find the date range
        datetime_cols = data.select_dtypes(include=['datetime64']).columns
        if len(datetime_cols) > 0:
            main_date_col = datetime_cols[0]  # Use the first datetime column
            key_metrics['date_range'] = {
                'start_date': data[main_date_col].min().strftime('%Y-%m-%d') if not pd.isna(data[main_date_col].min()) else None,
                'end_date': data[main_date_col].max().strftime('%Y-%m-%d') if not pd.isna(data[main_date_col].max()) else None,
                'total_days': (data[main_date_col].max() - data[main_date_col].min()).days if not pd.isna(data[main_date_col].min()) and not pd.isna(data[main_date_col].max()) else None
            }
        
        # If we have numeric columns, provide overall statistics
        numeric_cols = data.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            # Look for columns that might be amount/revenue/price columns
            amount_cols = [col for col in numeric_cols if any(term in col.lower() for term in ['amount', 'price', 'revenue', 'cost', 'value', 'income', 'sale'])]
            
            if amount_cols:
                # For each potential amount column, calculate total and average
                for col in amount_cols:
                    key_metrics[f'total_{col}'] = data[col].sum()
                    key_metrics[f'average_{col}'] = data[col].mean()
            
            # Look for percentage columns
            pct_cols = [col for col in numeric_cols if any(term in col.lower() for term in ['percent', 'rate', 'ratio', 'pct', '%'])]
            
            if pct_cols:
                # For each percentage column, calculate average
                for col in pct_cols:
                    key_metrics[f'average_{col}'] = data[col].mean()
        
        # Look for potential binary outcome columns (e.g., success/failure)
        binary_cols = []
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            if data[col].nunique() == 2:
                binary_cols.append(col)
        
        for num_col in numeric_cols:
            # Check if column contains only 0 and 1
            if set(data[num_col].dropna().unique()).issubset({0, 1}):
                binary_cols.append(num_col)
        
        # Calculate proportion for binary columns
        for col in binary_cols:
            # Determine the values
            values = data[col].dropna().unique()
            if len(values) == 2:
                try:
                    # Try to interpret numerically if possible
                    numeric_values = pd.to_numeric(values, errors='coerce')
                    if not np.isnan(numeric_values).any():
                        positive_value = max(numeric_values)
                        key_metrics[f'{col}_rate'] = (data[col] == positive_value).mean()
                    else:
                        # If not numeric, use alphabetically last value as "positive"
                        positive_value = sorted(values)[1]
                        key_metrics[f'{col}_rate'] = (data[col] == positive_value).mean()
                except:
                    # If conversion fails, use alphabetically last value as "positive"
                    positive_value = sorted(values)[1]
                    key_metrics[f'{col}_rate'] = (data[col] == positive_value).mean()
        
        self.logger.info("Key metrics calculation completed")
        return key_metrics
    
    def answer_question(self, question, data):
        """Answer a natural language question about the data."""
        if data is None:
            return "No dataset is currently loaded. Please load a dataset first."
        
        # Extract keywords from the question
        question = question.lower()
        
        # Questions about basic data description
        if any(term in question for term in ['what does the data contain', 'what is in the data', 'data overview', 'data summary']):
            numeric_count = len(data.select_dtypes(include=['number']).columns)
            categorical_count = len(data.select_dtypes(include=['object', 'category']).columns)
            datetime_count = len(data.select_dtypes(include=['datetime64']).columns)
            
            response = f"The dataset contains {len(data)} rows and {len(data.columns)} columns.\n"
            response += f"Column types: {numeric_count} numeric, {categorical_count} categorical, and {datetime_count} datetime columns."
            
            # Include sample column names
            if len(data.columns) > 0:
                sample_cols = data.columns[:5].tolist()
                response += f"\nSample columns: {', '.join(sample_cols)}"
                
                if len(data.columns) > 5:
                    response += f" (and {len(data.columns) - 5} more)"
            
            return response
        
        # Questions about specific column
        elif "column" in question and any(col.lower() in question for col in data.columns):
            # Find which column was mentioned
            mentioned_col = next((col for col in data.columns if col.lower() in question), None)
            
            if mentioned_col:
                col_type = data[mentioned_col].dtype
                
                response = f"Column '{mentioned_col}' is of type {col_type}.\n"
                
                # Provide specific details based on column type
                if np.issubdtype(col_type, np.number):
                    response += f"Range: {data[mentioned_col].min()} to {data[mentioned_col].max()}\n"
                    response += f"Mean: {data[mentioned_col].mean():.2f}, Median: {data[mentioned_col].median():.2f}\n"
                    response += f"Standard deviation: {data[mentioned_col].std():.2f}"
                
                elif col_type == 'object' or col_type == 'category':
                    unique_vals = data[mentioned_col].nunique()
                    response += f"It has {unique_vals} unique values.\n"
                    
                    # Show top values
                    top_values = data[mentioned_col].value_counts().head(3).to_dict()
                    response += "Top values:\n"
                    for val, count in top_values.items():
                        pct = (count / len(data)) * 100
                        response += f"- {val}: {count} ({pct:.1f}%)\n"
                
                elif pd.api.types.is_datetime64_any_dtype(data[mentioned_col]):
                    start_date = data[mentioned_col].min()
                    end_date = data[mentioned_col].max()
                    response += f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
                
                return response
            
            return "I couldn't identify which column you're asking about. Please specify the column name clearly."
        
        # Questions about correlations
        elif any(term in question for term in ['correlate', 'correlation', 'relationship', 'related']):
            numeric_cols = data.select_dtypes(include=['number']).columns
            
            if len(numeric_cols) < 2:
                return "There are not enough numeric columns to analyze correlations."
            
            # Calculate correlations
            corr = data[numeric_cols].corr()
            
            # Get top correlations (excluding self-correlations)
            corr_unstack = corr.unstack()
            corr_unstack = corr_unstack[corr_unstack < 1.0]  # Remove self-correlations
            top_corrs = corr_unstack.sort_values(ascending=False).head(5)
            
            response = "Top correlations between variables:\n"
            for (col1, col2), corr_val in top_corrs.items():
                response += f"- {col1} and {col2}: {corr_val:.3f}\n"
            
            return response
        
        # Questions about trends or patterns
        elif any(term in question for term in ['trend', 'pattern', 'change over time']):
            datetime_cols = data.select_dtypes(include=['datetime64']).columns
            
            if len(datetime_cols) == 0:
                return "No datetime columns found to analyze trends over time."
            
            date_col = datetime_cols[0]  # Use first datetime column
            
            # Look for numeric columns to analyze
            numeric_cols = data.select_dtypes(include=['number']).columns
            
            if len(numeric_cols) == 0:
                return "No numeric columns found to analyze trends."
            
            # Choose a sample numeric column
            sample_col = numeric_cols[0]
            
            # Group by year and month
            data['year'] = data[date_col].dt.year
            data['month'] = data[date_col].dt.month
            
            monthly_trend = data.groupby(['year', 'month'])[sample_col].mean().reset_index()
            
            # Format the response with trend information
            response = f"Monthly trend of {sample_col}:\n"
            for _, row in monthly_trend.head(10).iterrows():
                response += f"- {int(row['year'])}/{int(row['month']):02d}: {row[sample_col]:.2f}\n"
            
            if len(monthly_trend) > 10:
                response += f"(Showing first 10 of {len(monthly_trend)} months)"
            
            return response
        
        # Default response
        else:
            return "I can answer questions about the data overview, specific columns, correlations, or trends over time. Please specify what aspect of the data you'd like to explore."

if __name__ == "__main__":
    # Example usage
    agent = ExploratoryAgent()
    
    # Create a sample dataframe with time series data
    dates = pd.date_range('2020-01-01', periods=100)
    data = pd.DataFrame({
        'date': dates,
        'value': np.random.normal(100, 20, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    })
    
    # Process the data
    results = agent.process(data)
    print("Exploration results:", results.keys()) 