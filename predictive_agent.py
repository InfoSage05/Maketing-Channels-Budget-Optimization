import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class PredictiveAgent:
    """AI agent responsible for predictive modeling and forecasting."""
    
    def __init__(self, log_level=logging.INFO):
        """Initialize the predictive agent."""
        # Set up logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('PredictiveAgent')
        self.logger.info("Predictive agent initialized")
        
        # Store the trained models
        self.models = {}
    
    def process(self, data, target_column=None, model_type=None):
        """
        Process data for predictions.
        
        Parameters:
        - data: pandas DataFrame
        - target_column: column to predict (if None, try to infer)
        - model_type: 'regression', 'classification', or None to auto-detect
        """
        if data is None or data.empty:
            self.logger.error("No data provided for predictive modeling")
            return None
        
        self.logger.info(f"Starting predictive modeling on dataset with shape {data.shape}")
        
        # Auto-detect target column if not specified
        if target_column is None:
            target_column = self._infer_target_column(data)
            if target_column:
                self.logger.info(f"Inferred target column: {target_column}")
            else:
                self.logger.error("Could not infer target column. Please specify it explicitly.")
                return None
        
        # Check if target column exists in the data
        if target_column not in data.columns:
            self.logger.error(f"Target column '{target_column}' not found in the data")
            return None
        
        # Auto-detect model type if not specified
        if model_type is None:
            model_type = self._infer_model_type(data[target_column])
            self.logger.info(f"Inferred model type: {model_type}")
        
        # Prepare data for modeling
        X, y, preprocessing = self._prepare_data(data, target_column)
        
        if X is None or y is None:
            self.logger.error("Data preparation failed")
            return None
        
        # Split data for training and evaluation
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            self.logger.info(f"Split data into train ({len(X_train)} samples) and test ({len(X_test)} samples) sets")
        except Exception as e:
            self.logger.error(f"Error splitting data: {str(e)}")
            return None
        
        # Train models
        if model_type == 'regression':
            model_results = self._train_regression_models(X_train, X_test, y_train, y_test, preprocessing)
        elif model_type == 'classification':
            model_results = self._train_classification_models(X_train, X_test, y_train, y_test, preprocessing)
        else:
            self.logger.error(f"Unsupported model type: {model_type}")
            return None
        
        # Store the best model
        if model_results['best_model']:
            model_id = f"{target_column}_{model_type}_{len(self.models)}"
            self.models[model_id] = {
                'pipeline': model_results['best_model'],
                'feature_names': X.columns.tolist(),
                'target_column': target_column,
                'model_type': model_type,
                'metrics': model_results['best_metrics']
            }
            self.logger.info(f"Stored best model as '{model_id}'")
            model_results['model_id'] = model_id
        
        self.logger.info("Predictive modeling completed")
        return model_results
    
    def _infer_target_column(self, data):
        """Attempt to infer which column is the target for prediction."""
        # Look for columns with common target names
        potential_targets = [
            col for col in data.columns if any(keyword in col.lower() 
                for keyword in ['target', 'label', 'output', 'result', 'outcome', 'response', 
                               'dependent', 'predict', 'forecast', 'y'])
        ]
        
        if potential_targets:
            return potential_targets[0]
        
        # Try to find binary columns that might be targets
        binary_cols = []
        for col in data.columns:
            unique_vals = set(data[col].dropna().unique())
            if len(unique_vals) == 2 and all(val in [0, 1, '0', '1', True, False] for val in unique_vals):
                binary_cols.append(col)
        
        if binary_cols:
            return binary_cols[0]
        
        # If nothing else works, guess the last column
        return data.columns[-1]
    
    def _infer_model_type(self, target_series):
        """Infer whether to use regression or classification based on the target variable."""
        # Check data type
        if pd.api.types.is_numeric_dtype(target_series):
            # Check number of unique values
            unique_ratio = target_series.nunique() / len(target_series)
            
            # If few unique values or all values are integers in a small range, likely classification
            if unique_ratio < 0.05 or (
                all(float(x).is_integer() for x in target_series.dropna()) and
                target_series.max() - target_series.min() < 10
            ):
                return 'classification'
            else:
                return 'regression'
        else:
            # Non-numeric targets are handled as classification
            return 'classification'
    
    def _prepare_data(self, data, target_column):
        """Prepare data for modeling, including feature engineering and preprocessing."""
        self.logger.info("Preparing data for modeling")
        
        # Make a copy to avoid modifying the original data
        df = data.copy()
        
        # Extract target variable
        y = df[target_column].copy()
        X = df.drop(columns=[target_column])
        
        # Remove columns that are unlikely to be useful for prediction
        cols_to_drop = []
        
        # Drop columns with too many unique values (like IDs)
        for col in X.columns:
            if X[col].nunique() > 0.9 * len(X):
                cols_to_drop.append(col)
                self.logger.info(f"Dropping high cardinality column: {col}")
        
        # Drop columns with too many missing values
        for col in X.columns:
            if X[col].isnull().mean() > 0.5:
                cols_to_drop.append(col)
                self.logger.info(f"Dropping column with >50% missing values: {col}")
        
        X = X.drop(columns=cols_to_drop)
        
        # Identify column types for preprocessing
        numeric_features = X.select_dtypes(include=['number']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Handle datetime columns by extracting useful features
        datetime_features = X.select_dtypes(include=['datetime64']).columns.tolist()
        for col in datetime_features:
            X[f"{col}_year"] = X[col].dt.year
            X[f"{col}_month"] = X[col].dt.month
            X[f"{col}_day"] = X[col].dt.day
            X[f"{col}_dayofweek"] = X[col].dt.dayofweek
            
            numeric_features.extend([f"{col}_year", f"{col}_month", f"{col}_day", f"{col}_dayofweek"])
            
            # Drop the original datetime column
            X = X.drop(columns=[col])
        
        self.logger.info(f"Prepared {len(numeric_features)} numeric features and {len(categorical_features)} categorical features")
        
        # Create preprocessing pipeline
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
        
        return X, y, preprocessor
    
    def _train_regression_models(self, X_train, X_test, y_train, y_test, preprocessor):
        """Train and evaluate regression models."""
        self.logger.info("Training regression models")
        
        # Define models to try
        models = {
            'linear_regression': Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', LinearRegression())
            ]),
            'random_forest': Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
            ])
        }
        
        results = {'model_comparison': {}}
        best_score = float('-inf')
        best_model = None
        best_metrics = {}
        
        # Train and evaluate each model
        for name, model in models.items():
            try:
                self.logger.info(f"Training {name}...")
                model.fit(X_train, y_train)
                
                # Cross-validation score
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
                cv_r2 = cv_scores.mean()
                
                # Predict on test set
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                metrics = {
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2,
                    'cv_r2': cv_r2
                }
                
                results['model_comparison'][name] = metrics
                
                # Track the best model
                if r2 > best_score:
                    best_score = r2
                    best_model = model
                    best_metrics = metrics
                
                self.logger.info(f"Completed {name}, R²: {r2:.4f}")
                
            except Exception as e:
                self.logger.error(f"Error training {name}: {str(e)}")
        
        results['best_model'] = best_model
        results['best_metrics'] = best_metrics
        
        # Feature importance if the best model is a random forest
        if best_model and 'random_forest' in results['model_comparison']:
            try:
                # Get feature names from preprocessor
                feature_names = []
                
                # Get numerical feature names directly
                feature_names.extend(preprocessor.transformers_[0][2])
                
                # Get one-hot encoded categorical feature names
                if len(preprocessor.transformers_) > 1:
                    encoder = preprocessor.transformers_[1][1].named_steps['onehot']
                    cat_features = preprocessor.transformers_[1][2]
                    
                    if hasattr(encoder, 'get_feature_names_out'):
                        cat_feature_names = encoder.get_feature_names_out(cat_features)
                        feature_names.extend(cat_feature_names)
                    elif hasattr(encoder, 'get_feature_names'):
                        cat_feature_names = encoder.get_feature_names(cat_features)
                        feature_names.extend(cat_feature_names)
                
                # Extract feature importances from the random forest model
                rf_model = models['random_forest'].named_steps['regressor']
                importances = rf_model.feature_importances_
                
                # If we have feature names, use them
                if len(feature_names) == len(importances):
                    feature_importance = dict(zip(feature_names, importances))
                    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
                    results['feature_importance'] = {feat: imp for feat, imp in top_features}
            except Exception as e:
                self.logger.warning(f"Could not extract feature importance: {str(e)}")
        
        return results
    
    def _train_classification_models(self, X_train, X_test, y_train, y_test, preprocessor):
        """Train and evaluate classification models."""
        self.logger.info("Training classification models")
        
        # Convert target to numeric for compatibility with all classifiers
        if not pd.api.types.is_numeric_dtype(y_train):
            from sklearn.preprocessing import LabelEncoder
            label_encoder = LabelEncoder()
            y_train = label_encoder.fit_transform(y_train)
            y_test = label_encoder.transform(y_test)
        
        # Define models to try
        models = {
            'logistic_regression': Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', LogisticRegression(max_iter=1000, random_state=42))
            ]),
            'random_forest': Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
            ])
        }
        
        results = {'model_comparison': {}}
        best_score = float('-inf')
        best_model = None
        best_metrics = {}
        
        # Check if it's a binary classification problem
        is_binary = len(np.unique(y_train)) == 2
        
        # Train and evaluate each model
        for name, model in models.items():
            try:
                self.logger.info(f"Training {name}...")
                model.fit(X_train, y_train)
                
                # Cross-validation score
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_weighted')
                cv_f1 = cv_scores.mean()
                
                # Predict on test set
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                metrics = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'cv_f1': cv_f1
                }
                
                # Add AUC for binary classification
                if is_binary:
                    try:
                        y_prob = model.predict_proba(X_test)[:, 1]
                        auc = roc_auc_score(y_test, y_prob)
                        metrics['auc'] = auc
                    except:
                        self.logger.warning(f"Could not calculate AUC for {name}")
                
                results['model_comparison'][name] = metrics
                
                # Track the best model (using F1 score)
                if f1 > best_score:
                    best_score = f1
                    best_model = model
                    best_metrics = metrics
                
                self.logger.info(f"Completed {name}, F1: {f1:.4f}")
                
            except Exception as e:
                self.logger.error(f"Error training {name}: {str(e)}")
        
        results['best_model'] = best_model
        results['best_metrics'] = best_metrics
        
        # Feature importance if the best model is a random forest
        if best_model and 'random_forest' in results['model_comparison']:
            try:
                # Get feature names from preprocessor
                feature_names = []
                
                # Get numerical feature names directly
                feature_names.extend(preprocessor.transformers_[0][2])
                
                # Get one-hot encoded categorical feature names
                if len(preprocessor.transformers_) > 1:
                    encoder = preprocessor.transformers_[1][1].named_steps['onehot']
                    cat_features = preprocessor.transformers_[1][2]
                    
                    if hasattr(encoder, 'get_feature_names_out'):
                        cat_feature_names = encoder.get_feature_names_out(cat_features)
                        feature_names.extend(cat_feature_names)
                    elif hasattr(encoder, 'get_feature_names'):
                        cat_feature_names = encoder.get_feature_names(cat_features)
                        feature_names.extend(cat_feature_names)
                
                # Extract feature importances from the random forest model
                rf_model = models['random_forest'].named_steps['classifier']
                importances = rf_model.feature_importances_
                
                # If we have feature names, use them
                if len(feature_names) == len(importances):
                    feature_importance = dict(zip(feature_names, importances))
                    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
                    results['feature_importance'] = {feat: imp for feat, imp in top_features}
            except Exception as e:
                self.logger.warning(f"Could not extract feature importance: {str(e)}")
        
        return results
    
    def predict(self, model_id, new_data):
        """Make predictions using a trained model."""
        if model_id not in self.models:
            self.logger.error(f"Model '{model_id}' not found")
            return None
        
        model_info = self.models[model_id]
        pipeline = model_info['pipeline']
        feature_names = model_info['feature_names']
        
        # Ensure new_data has the expected features
        missing_features = set(feature_names) - set(new_data.columns)
        extra_features = set(new_data.columns) - set(feature_names)
        
        if missing_features:
            self.logger.warning(f"New data is missing features: {missing_features}")
        
        # Select only the features used during training
        common_features = list(set(new_data.columns) & set(feature_names))
        X = new_data[common_features]
        
        # Add missing columns with NaN values (preprocessing will handle these)
        for feature in missing_features:
            X[feature] = np.nan
        
        # Ensure columns are in the same order as during training
        X = X[feature_names]
        
        # Make predictions
        try:
            predictions = pipeline.predict(X)
            return predictions
        except Exception as e:
            self.logger.error(f"Error making predictions: {str(e)}")
            return None
    
    def answer_question(self, question, data):
        """Answer a natural language question about predictive modeling."""
        if data is None:
            return "No dataset is currently loaded. Please load a dataset first."
        
        # Extract keywords from the question
        question = question.lower()
        
        # Questions about model performance
        if any(term in question for term in ['performance', 'accuracy', 'score', 'how good', 'how well']):
            # Check if we have a trained model
            if not self.models:
                # Train a model first
                target_column = self._infer_target_column(data)
                model_type = self._infer_model_type(data[target_column])
                model_results = self.process(data, target_column, model_type)
                
                if model_results and model_results['best_metrics']:
                    metrics = model_results['best_metrics']
                    if model_type == 'regression':
                        return f"The best regression model achieved R² = {metrics['r2']:.4f} on the test set, with RMSE = {metrics['rmse']:.4f}."
                    else:
                        return f"The best classification model achieved accuracy = {metrics['accuracy']:.4f} and F1 score = {metrics['f1']:.4f} on the test set."
                else:
                    return "Unable to train a predictive model successfully."
            else:
                # Report on the existing model
                model_id = list(self.models.keys())[0]
                model_info = self.models[model_id]
                metrics = model_info['metrics']
                
                if model_info['model_type'] == 'regression':
                    return f"The trained regression model for {model_info['target_column']} achieved R² = {metrics['r2']:.4f} on the test set, with RMSE = {metrics['rmse']:.4f}."
                else:
                    return f"The trained classification model for {model_info['target_column']} achieved accuracy = {metrics['accuracy']:.4f} and F1 score = {metrics['f1']:.4f} on the test set."
        
        # Questions about feature importance
        elif any(term in question for term in ['important feature', 'feature importance', 'which feature', 'what factor']):
            # Check if we have a trained random forest model with feature importance
            importance_available = False
            
            for model_id, model_info in self.models.items():
                if 'random_forest' in model_id or isinstance(model_info['pipeline'].steps[-1][1], RandomForestRegressor) or isinstance(model_info['pipeline'].steps[-1][1], RandomForestClassifier):
                    importance_available = True
                    break
            
            if not importance_available:
                # Train a random forest model
                target_column = self._infer_target_column(data)
                model_type = self._infer_model_type(data[target_column])
                model_results = self.process(data, target_column, model_type)
                
                if model_results and 'feature_importance' in model_results:
                    response = f"Top features for predicting {target_column}:\n"
                    for i, (feature, importance) in enumerate(model_results['feature_importance'].items(), 1):
                        response += f"{i}. {feature}: {importance:.4f}\n"
                    return response
                else:
                    return "Unable to determine feature importance. Try training a random forest model first."
            else:
                # Report on the existing model
                for model_id, model_info in self.models.items():
                    if 'random_forest' in model_id or isinstance(model_info['pipeline'].steps[-1][1], RandomForestRegressor) or isinstance(model_info['pipeline'].steps[-1][1], RandomForestClassifier):
                        pipeline = model_info['pipeline']
                        if hasattr(pipeline.steps[-1][1], 'feature_importances_'):
                            importances = pipeline.steps[-1][1].feature_importances_
                            
                            # Try to get feature names
                            feature_names = []
                            preprocessor = pipeline.steps[0][1]
                            
                            # Get numerical feature names directly
                            feature_names.extend(preprocessor.transformers_[0][2])
                            
                            # Get one-hot encoded categorical feature names
                            if len(preprocessor.transformers_) > 1:
                                encoder = preprocessor.transformers_[1][1].named_steps['onehot']
                                cat_features = preprocessor.transformers_[1][2]
                                
                                if hasattr(encoder, 'get_feature_names_out'):
                                    cat_feature_names = encoder.get_feature_names_out(cat_features)
                                    feature_names.extend(cat_feature_names)
                                elif hasattr(encoder, 'get_feature_names'):
                                    cat_feature_names = encoder.get_feature_names(cat_features)
                                    feature_names.extend(cat_feature_names)
                            
                            # Match feature names with importances
                            if len(feature_names) == len(importances):
                                feature_importance = dict(zip(feature_names, importances))
                                top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
                                
                                response = f"Top features for predicting {model_info['target_column']}:\n"
                                for i, (feature, importance) in enumerate(top_features, 1):
                                    response += f"{i}. {feature}: {importance:.4f}\n"
                                return response
                
                return "Feature importance information is not available for the current model."
        
        # Questions about forecasting or predicting
        elif any(term in question for term in ['forecast', 'predict', 'future value']):
            # Check if we have enough data for a forecast
            datetime_cols = data.select_dtypes(include=['datetime64']).columns
            
            if len(datetime_cols) == 0:
                return "No datetime columns found for forecasting. Please provide data with a timestamp column."
            
            date_col = datetime_cols[0]
            
            # Check if we already have a model
            if not self.models:
                return "No forecasting model is available. First train a model using a numeric target column."
            
            # Find a suitable numeric column to forecast
            numeric_cols = data.select_dtypes(include=['number']).columns
            
            if len(numeric_cols) == 0:
                return "No numeric columns found to forecast."
            
            forecast_col = numeric_cols[0]
            
            # Generate a simple message about forecasting
            return f"To forecast future values of {forecast_col}, I would need more information about the specific time period you want to predict. The data contains timestamps from {data[date_col].min()} to {data[date_col].max()}."
        
        # Default response for other questions
        else:
            return "I can help with questions about model performance, feature importance, or making predictions. Please specify what aspect of predictive modeling you're interested in."

if __name__ == "__main__":
    # Example usage
    agent = PredictiveAgent()
    
    # Create a sample regression dataset
    np.random.seed(42)
    X = np.random.rand(100, 3)
    y = 2 * X[:, 0] + 3 * X[:, 1] - 1.5 * X[:, 2] + np.random.normal(0, 0.1, 100)
    
    df = pd.DataFrame(X, columns=['feature1', 'feature2', 'feature3'])
    df['target'] = y
    
    # Process the data
    results = agent.process(df, 'target', 'regression')
    print("Regression model results:", results['model_comparison']) 