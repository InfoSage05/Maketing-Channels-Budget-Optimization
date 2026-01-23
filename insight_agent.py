import logging
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

class InsightAgent:
    """AI agent responsible for generating insights from data analysis."""
    
    def __init__(self, log_level=logging.INFO):
        """Initialize the insight agent."""
        # Set up logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('InsightAgent')
        self.logger.info("Insight agent initialized")
    
    def process(self, data, exploratory_results=None):
        """Generate insights from the data and exploratory analysis results."""
        if data is None or data.empty:
            self.logger.error("No data provided for insight generation")
            return None
        
        self.logger.info(f"Starting insight generation on dataset with shape {data.shape}")
        
        insights = {
            "key_findings": self.extract_key_findings(data, exploratory_results),
            "anomalies": self.detect_anomalies(data),
            "segments": self.identify_segments(data),
            "trends": self.identify_trends(data),
            "recommendations": self.generate_recommendations(data, exploratory_results)
        }
        
        self.logger.info("Insight generation completed")
        return insights
    
    def extract_key_findings(self, data, exploratory_results=None):
        """Extract key findings from the exploratory analysis results."""
        self.logger.info("Extracting key findings")
        
        findings = []
        
        # If we have exploratory results, use them to extract findings
        if exploratory_results and "correlation_analysis" in exploratory_results:
            # Extract insights from correlations
            corr_results = exploratory_results["correlation_analysis"]
            
            if "top_positive" in corr_results:
                # Get top 3 positive correlations
                top_pos = list(corr_results["top_positive"].items())[:3]
                for (col1, col2), corr_val in top_pos:
                    if corr_val > 0.7:  # Only report strong correlations
                        findings.append({
                            "type": "correlation",
                            "description": f"Strong positive correlation of {corr_val:.2f} between {col1} and {col2}",
                            "importance": "high" if corr_val > 0.8 else "medium"
                        })
            
            if "top_negative" in corr_results:
                # Get top 3 negative correlations
                top_neg = list(corr_results["top_negative"].items())[:3]
                for (col1, col2), corr_val in top_neg:
                    if corr_val < -0.7:  # Only report strong negative correlations
                        findings.append({
                            "type": "correlation",
                            "description": f"Strong negative correlation of {corr_val:.2f} between {col1} and {col2}",
                            "importance": "high" if corr_val < -0.8 else "medium"
                        })
        
        # Check for skewed distributions
        numeric_cols = data.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            skewness = data[col].skew()
            if abs(skewness) > 1.5:
                direction = "positively" if skewness > 0 else "negatively"
                findings.append({
                    "type": "distribution",
                    "description": f"Column '{col}' is highly {direction} skewed (skewness: {skewness:.2f})",
                    "importance": "medium"
                })
        
        # Check for high cardinality in categorical columns
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            unique_ratio = data[col].nunique() / len(data)
            if unique_ratio > 0.8 and data[col].nunique() > 10:
                findings.append({
                    "type": "cardinality",
                    "description": f"Column '{col}' has high cardinality with {data[col].nunique()} unique values ({unique_ratio:.1%} of rows)",
                    "importance": "low"
                })
        
        # Check for imbalance in binary columns
        for col in data.columns:
            if set(data[col].dropna().unique()) == {0, 1} or set(data[col].dropna().unique()) == {'0', '1'}:
                # Convert to numeric to handle string values
                values = pd.to_numeric(data[col], errors='coerce')
                ratio = values.mean()
                
                if ratio < 0.1 or ratio > 0.9:
                    minority_val = 1 if ratio < 0.5 else 0
                    minority_pct = (1 - ratio) if ratio > 0.5 else ratio
                    findings.append({
                        "type": "imbalance",
                        "description": f"Binary column '{col}' is highly imbalanced with class {minority_val} representing only {minority_pct:.1%} of data",
                        "importance": "medium"
                    })
        
        # Look for interesting time patterns if we have datetime columns
        datetime_cols = data.select_dtypes(include=['datetime64']).columns
        if len(datetime_cols) > 0:
            date_col = datetime_cols[0]  # Use the first datetime column
            
            # Check data completeness across time
            data['temp_date'] = data[date_col].dt.date
            daily_counts = data.groupby('temp_date').size()
            
            # Look for days with unusually low counts
            mean_count = daily_counts.mean()
            std_count = daily_counts.std()
            
            low_days = daily_counts[daily_counts < (mean_count - 2 * std_count)]
            if len(low_days) > 0:
                findings.append({
                    "type": "time_pattern",
                    "description": f"Found {len(low_days)} days with unusually low data volume (below {mean_count - 2 * std_count:.0f} records)",
                    "importance": "medium"
                })
            
            # Clean up temporary column
            data = data.drop(columns=['temp_date'])
        
        self.logger.info(f"Extracted {len(findings)} key findings")
        return findings
    
    def detect_anomalies(self, data):
        """Detect anomalies in the data using isolation forest."""
        self.logger.info("Detecting anomalies")
        
        anomalies = []
        
        # Only use numeric columns for anomaly detection
        numeric_data = data.select_dtypes(include=['number'])
        
        if numeric_data.shape[1] < 2:
            self.logger.info("Not enough numeric columns for anomaly detection")
            return anomalies
        
        # Select columns with less than 10% missing values
        valid_cols = numeric_data.columns[numeric_data.isnull().mean() < 0.1]
        
        if len(valid_cols) < 2:
            self.logger.info("Not enough valid numeric columns for anomaly detection")
            return anomalies
        
        try:
            # Prepare data for anomaly detection
            X = numeric_data[valid_cols].fillna(numeric_data[valid_cols].median())
            
            # Use Isolation Forest for anomaly detection
            isolation_forest = IsolationForest(contamination=0.05, random_state=42)
            anomaly_scores = isolation_forest.fit_predict(X)
            
            # Isolation Forest returns -1 for anomalies, 1 for normal points
            anomaly_indices = np.where(anomaly_scores == -1)[0]
            
            if len(anomaly_indices) > 0:
                # Calculate feature importance
                feature_importances = {}
                
                for col in valid_cols:
                    normal_values = data.loc[anomaly_scores == 1, col].dropna()
                    anomaly_values = data.loc[anomaly_indices, col].dropna()
                    
                    if len(normal_values) > 0 and len(anomaly_values) > 0:
                        # Compare distributions using KS test
                        try:
                            ks_stat, _ = stats.ks_2samp(normal_values, anomaly_values)
                            feature_importances[col] = ks_stat
                        except:
                            self.logger.warning(f"KS test failed for column {col}")
                
                # Sort features by importance
                sorted_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
                
                # Analyze the top anomalies
                top_anomalies = anomaly_indices[:min(10, len(anomaly_indices))]
                
                for idx in top_anomalies:
                    record = data.iloc[idx]
                    
                    # Construct anomaly description
                    unusual_features = []
                    for feature, importance in sorted_features[:3]:  # Top 3 contributing features
                        if importance > 0.3:  # Only include significant features
                            value = record[feature]
                            avg_value = data[feature].mean()
                            std_value = data[feature].std()
                            
                            if abs(value - avg_value) > 2 * std_value:
                                direction = "high" if value > avg_value else "low"
                                unusual_features.append({
                                    "feature": feature,
                                    "value": value,
                                    "avg_value": avg_value,
                                    "direction": direction,
                                    "z_score": (value - avg_value) / std_value
                                })
                    
                    if unusual_features:
                        anomalies.append({
                            "index": idx,
                            "unusual_features": unusual_features,
                            "importance": "high" if len(unusual_features) > 1 else "medium"
                        })
        
        except Exception as e:
            self.logger.error(f"Anomaly detection failed: {str(e)}")
        
        self.logger.info(f"Detected {len(anomalies)} anomalies")
        return anomalies
    
    def identify_segments(self, data, max_clusters=5):
        """Identify meaningful segments in the data using clustering."""
        self.logger.info("Identifying segments")
        
        segments = []
        
        # Only use numeric columns for clustering
        numeric_data = data.select_dtypes(include=['number'])
        
        if numeric_data.shape[1] < 2:
            self.logger.info("Not enough numeric columns for segment identification")
            return segments
        
        # Select columns with less than 10% missing values
        valid_cols = numeric_data.columns[numeric_data.isnull().mean() < 0.1]
        
        if len(valid_cols) < 2:
            self.logger.info("Not enough valid numeric columns for segment identification")
            return segments
        
        try:
            # Prepare data for clustering
            X = numeric_data[valid_cols].fillna(numeric_data[valid_cols].median())
            
            # Determine optimal number of clusters
            best_score = float('-inf')
            best_k = 2
            
            # Try different numbers of clusters
            for k in range(2, min(max_clusters + 1, len(X) // 100 + 2)):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(X)
                
                # Calculate silhouette score
                if len(set(cluster_labels)) > 1:  # Need at least 2 clusters for silhouette score
                    try:
                        silhouette = stats.silhouette_score(X, cluster_labels)
                        if silhouette > best_score:
                            best_score = silhouette
                            best_k = k
                    except:
                        self.logger.warning(f"Silhouette score calculation failed for k={k}")
            
            # Use the best number of clusters
            kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X)
            
            # Analyze each cluster
            for i in range(best_k):
                cluster_indices = np.where(cluster_labels == i)[0]
                cluster_size = len(cluster_indices)
                
                if cluster_size > 0:
                    # Calculate cluster statistics
                    cluster_data = data.iloc[cluster_indices]
                    
                    # Find distinctive features of this cluster
                    distinctive_features = []
                    
                    for col in valid_cols:
                        cluster_mean = cluster_data[col].mean()
                        overall_mean = data[col].mean()
                        overall_std = data[col].std()
                        
                        if overall_std > 0:
                            z_score = (cluster_mean - overall_mean) / overall_std
                            
                            if abs(z_score) > 1.0:  # Only include significant differences
                                direction = "higher" if z_score > 0 else "lower"
                                distinctive_features.append({
                                    "feature": col,
                                    "cluster_mean": cluster_mean,
                                    "overall_mean": overall_mean,
                                    "difference_percent": ((cluster_mean / overall_mean) - 1) * 100 if overall_mean != 0 else 0,
                                    "direction": direction,
                                    "importance": abs(z_score)
                                })
                    
                    # Sort features by importance
                    distinctive_features.sort(key=lambda x: x["importance"], reverse=True)
                    
                    if distinctive_features:
                        segments.append({
                            "cluster_id": i,
                            "size": cluster_size,
                            "percentage": (cluster_size / len(data)) * 100,
                            "distinctive_features": distinctive_features[:5],  # Top 5 most distinctive features
                            "importance": "high" if distinctive_features[0]["importance"] > 2 else "medium"
                        })
        
        except Exception as e:
            self.logger.error(f"Segment identification failed: {str(e)}")
        
        self.logger.info(f"Identified {len(segments)} segments")
        return segments
    
    def identify_trends(self, data):
        """Identify trends in time series data."""
        self.logger.info("Identifying trends")
        
        trends = []
        
        # Look for datetime columns
        datetime_cols = data.select_dtypes(include=['datetime64']).columns
        
        if len(datetime_cols) == 0:
            self.logger.info("No datetime columns found for trend analysis")
            return trends
        
        date_col = datetime_cols[0]  # Use the first datetime column
        
        # For each numeric column, analyze trend over time
        numeric_cols = data.select_dtypes(include=['number']).columns
        
        for col in numeric_cols:
            try:
                # Skip columns with too many missing values
                if data[col].isnull().sum() / len(data) > 0.3:
                    continue
                
                # Group by year-month and calculate mean
                data['year_month'] = data[date_col].dt.to_period('M')
                monthly_means = data.groupby('year_month')[col].mean()
                
                if len(monthly_means) < 3:
                    continue  # Need at least 3 months of data
                
                # Calculate trend using linear regression
                x = np.arange(len(monthly_means))
                y = monthly_means.values
                
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                
                # Calculate overall percent change
                if len(monthly_means) > 0 and monthly_means.iloc[0] != 0:
                    first_value = monthly_means.iloc[0]
                    last_value = monthly_means.iloc[-1]
                    percent_change = ((last_value / first_value) - 1) * 100
                else:
                    percent_change = 0
                
                # Check if trend is statistically significant
                if p_value < 0.05 and abs(r_value) > 0.3:
                    direction = "increasing" if slope > 0 else "decreasing"
                    trends.append({
                        "metric": col,
                        "direction": direction,
                        "strength": abs(r_value),
                        "percent_change": percent_change,
                        "p_value": p_value,
                        "months_analyzed": len(monthly_means),
                        "importance": "high" if abs(r_value) > 0.7 else "medium"
                    })
                
                # Clean up temporary column
                data = data.drop(columns=['year_month'])
            
            except Exception as e:
                self.logger.warning(f"Trend analysis failed for column {col}: {str(e)}")
        
        # Sort trends by strength
        trends.sort(key=lambda x: x["strength"], reverse=True)
        
        self.logger.info(f"Identified {len(trends)} significant trends")
        return trends
    
    def generate_recommendations(self, data, exploratory_results=None):
        """Generate actionable recommendations based on the analysis."""
        self.logger.info("Generating recommendations")
        
        recommendations = []
        
        # Recommendation for data quality
        missing_data = data.isnull().sum().sum()
        if missing_data > 0:
            missing_ratio = missing_data / (data.shape[0] * data.shape[1])
            if missing_ratio > 0.1:
                recommendations.append({
                    "type": "data_quality",
                    "description": f"Address missing data issues ({missing_ratio:.1%} of values are missing)",
                    "details": "Consider using more advanced imputation techniques for missing values."
                })
        
        # Recommendations based on correlations
        if exploratory_results and "correlation_analysis" in exploratory_results:
            corr_results = exploratory_results["correlation_analysis"]
            
            if "top_positive" in corr_results:
                # Look for potential multicollinearity
                multicollinear_pairs = []
                for (col1, col2), corr_val in corr_results["top_positive"].items():
                    if corr_val > 0.9:
                        multicollinear_pairs.append((col1, col2))
                
                if multicollinear_pairs:
                    recommendations.append({
                        "type": "feature_selection",
                        "description": f"Address multicollinearity among {len(multicollinear_pairs)} pairs of features",
                        "details": f"Consider removing one feature from each highly correlated pair to improve model stability."
                    })
        
        # Recommendations for imbalanced classes
        binary_cols = []
        for col in data.columns:
            unique_vals = set(data[col].dropna().unique())
            if unique_vals == {0, 1} or unique_vals == {'0', '1'} or unique_vals == {True, False}:
                binary_cols.append(col)
        
        for col in binary_cols:
            # Convert to numeric to handle string/boolean values
            values = pd.to_numeric(data[col], errors='coerce')
            ratio = values.mean()
            
            if ratio < 0.1 or ratio > 0.9:
                recommendations.append({
                    "type": "class_balance",
                    "description": f"Address imbalance in column '{col}' ({min(ratio, 1-ratio):.1%} minority class)",
                    "details": "Consider using techniques like SMOTE, class weights, or collecting more data for the minority class."
                })
        
        # Recommendations for potential feature engineering
        datetime_cols = data.select_dtypes(include=['datetime64']).columns
        if len(datetime_cols) > 0 and not any('dayofweek' in col for col in data.columns):
            recommendations.append({
                "type": "feature_engineering",
                "description": "Create cyclical time features from datetime columns",
                "details": "Extract day of week, hour of day, etc. and encode them as cyclical features using sine/cosine transformations."
            })
        
        # Recommendations for handling outliers
        numeric_cols = data.select_dtypes(include=['number']).columns
        outlier_cols = []
        
        for col in numeric_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            
            outlier_count = ((data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR))).sum()
            outlier_ratio = outlier_count / len(data)
            
            if outlier_ratio > 0.05:
                outlier_cols.append((col, outlier_ratio))
        
        if outlier_cols:
            recommendations.append({
                "type": "outlier_handling",
                "description": f"Address outliers in {len(outlier_cols)} columns",
                "details": "Consider transforming skewed variables using log or Box-Cox transformations."
            })
        
        self.logger.info(f"Generated {len(recommendations)} recommendations")
        return recommendations
    
    def answer_question(self, question, data):
        """Answer a natural language question about insights from the data."""
        if data is None:
            return "No dataset is currently loaded. Please load a dataset first."
        
        # Process the data first to generate insights
        exploratory_results = {
            "correlation_analysis": self.analyze_correlations(data),
            "key_metrics": self.calculate_key_metrics(data)
        }
        
        insights = self.process(data, exploratory_results)
        
        # Extract keywords from the question
        question = question.lower()
        
        # Questions about patterns
        if "pattern" in question:
            if insights["trends"]:
                response = "Key patterns detected in the data:\n"
                for trend in insights["trends"]:
                    response += f"- {trend['metric']} is {trend['direction']} over time (r² = {trend['strength']**2:.2f})\n"
                return response
            else:
                return "No significant patterns were detected in the time series data."
        
        # Questions about anomalies
        elif "anomal" in question or "outlier" in question:
            if insights["anomalies"]:
                response = "Key anomalies detected in the data:\n"
                for i, anomaly in enumerate(insights["anomalies"][:3]):
                    response += f"- Record {anomaly['index']}: "
                    feature_desc = []
                    for feature in anomaly["unusual_features"]:
                        feature_desc.append(f"{feature['feature']} is unusually {feature['direction']} ({feature['z_score']:.1f} std. dev.)")
                    response += ", ".join(feature_desc) + "\n"
                
                if len(insights["anomalies"]) > 3:
                    response += f"- Plus {len(insights['anomalies']) - 3} more anomalies\n"
                
                return response
            else:
                return "No significant anomalies were detected in the data."
        
        # Questions about segments
        elif "segment" in question or "cluster" in question:
            if insights["segments"]:
                response = "Key segments identified in the data:\n"
                for segment in insights["segments"]:
                    top_features = segment["distinctive_features"][:2]
                    feature_desc = []
                    for feature in top_features:
                        feature_desc.append(f"{feature['feature']} is {feature['direction']} ({feature['difference_percent']:.1f}%)")
                    
                    response += f"- Segment {segment['cluster_id']}: {segment['percentage']:.1f}% of data, characterized by: {', '.join(feature_desc)}\n"
                
                return response
            else:
                return "No significant segments were identified in the data."
        
        # Questions about key findings
        elif "finding" in question or "insight" in question:
            if insights["key_findings"]:
                response = "Key findings from the data analysis:\n"
                for finding in insights["key_findings"]:
                    response += f"- {finding['description']}\n"
                
                return response
            else:
                return "No significant findings were identified in the analysis."
        
        # Questions about recommendations
        elif "recommend" in question or "suggest" in question:
            if insights["recommendations"]:
                response = "Recommendations based on data analysis:\n"
                for recommendation in insights["recommendations"]:
                    response += f"- {recommendation['description']}\n"
                
                return response
            else:
                return "No specific recommendations were generated from the analysis."
        
        # Default response
        else:
            # Provide a summary of the most important insights
            response = "Summary of key insights from the data:\n"
            
            # Include top findings
            if insights["key_findings"]:
                high_importance = [f for f in insights["key_findings"] if f.get("importance") == "high"]
                if high_importance:
                    response += "Important findings:\n"
                    for finding in high_importance[:2]:
                        response += f"- {finding['description']}\n"
            
            # Include top trends
            if insights["trends"]:
                top_trends = sorted(insights["trends"], key=lambda x: x["strength"], reverse=True)[:2]
                if top_trends:
                    response += "\nSignificant trends:\n"
                    for trend in top_trends:
                        response += f"- {trend['metric']} is {trend['direction']} over time ({trend['percent_change']:.1f}% change)\n"
            
            # Include top recommendations
            if insights["recommendations"]:
                response += "\nKey recommendations:\n"
                for recommendation in insights["recommendations"][:2]:
                    response += f"- {recommendation['description']}\n"
            
            return response
    
    def analyze_correlations(self, data):
        """Utility method to analyze correlations for answering questions."""
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
        
        return correlation_results
    
    def calculate_key_metrics(self, data):
        """Utility method to calculate key metrics for answering questions."""
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
        
        return key_metrics

if __name__ == "__main__":
    # Example usage
    agent = InsightAgent()
    
    # Create a sample dataframe
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=365)
    data = pd.DataFrame({
        'date': dates,
        'value': np.random.normal(100, 20, 365) + np.arange(365) * 0.1,  # Increasing trend
        'category': np.random.choice(['A', 'B', 'C'], 365)
    })
    
    # Process the data
    insights = agent.process(data)
    print("Insights:", insights.keys()) 