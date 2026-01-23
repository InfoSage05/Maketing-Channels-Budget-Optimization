import logging
import pandas as pd
from data_cleaning_agent import DataCleaningAgent
from exploratory_agent import ExploratoryAgent
from insight_agent import InsightAgent
from predictive_agent import PredictiveAgent
from visualization_agent import VisualizationAgent

class AgentController:
    """Main controller that coordinates all AI agents for data analysis."""
    
    def __init__(self, log_level=logging.INFO):
        """Initialize the agent controller with all specialized agents."""
        # Set up logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('AgentController')
        
        # Initialize agents
        self.data_cleaning_agent = DataCleaningAgent()
        self.exploratory_agent = ExploratoryAgent()
        self.insight_agent = InsightAgent()
        self.predictive_agent = PredictiveAgent()
        self.visualization_agent = VisualizationAgent()
        
        self.datasets = {}
        self.logger.info("Agent controller initialized with all specialized agents")
    
    def load_dataset(self, name, filepath, file_type='csv'):
        """Load a dataset from file into the system."""
        self.logger.info(f"Loading dataset '{name}' from {filepath}")
        
        try:
            if file_type.lower() == 'csv':
                data = pd.read_csv(filepath)
            elif file_type.lower() == 'excel' or file_type.lower() == 'xlsx':
                data = pd.read_excel(filepath)
            elif file_type.lower() == 'json':
                data = pd.read_json(filepath)
            else:
                self.logger.error(f"Unsupported file type: {file_type}")
                return False
            
            self.datasets[name] = data
            self.logger.info(f"Successfully loaded dataset '{name}' with shape {data.shape}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading dataset '{name}': {str(e)}")
            return False
    
    def process_dataset(self, dataset_name):
        """Process a dataset through the entire agent pipeline."""
        if dataset_name not in self.datasets:
            self.logger.error(f"Dataset '{dataset_name}' not found")
            return None
        
        data = self.datasets[dataset_name]
        self.logger.info(f"Starting processing pipeline for '{dataset_name}'")
        
        # Step 1: Clean the data
        clean_data = self.data_cleaning_agent.process(data)
        self.datasets[f"{dataset_name}_cleaned"] = clean_data
        
        # Step 2: Run exploratory analysis
        exploratory_results = self.exploratory_agent.process(clean_data)
        
        # Step 3: Generate insights
        insights = self.insight_agent.process(clean_data, exploratory_results)
        
        # Step 4: Run predictive models if requested
        predictions = self.predictive_agent.process(clean_data)
        
        # Step 5: Generate visualizations
        visualizations = self.visualization_agent.process(
            clean_data, 
            exploratory_results=exploratory_results,
            insights=insights,
            predictions=predictions
        )
        
        self.logger.info(f"Completed processing pipeline for '{dataset_name}'")
        
        return {
            "cleaned_data": clean_data,
            "exploratory_results": exploratory_results,
            "insights": insights,
            "predictions": predictions,
            "visualizations": visualizations
        }
    
    def ask_question(self, question, dataset_name=None):
        """Ask a natural language question about the data."""
        if dataset_name and dataset_name not in self.datasets:
            self.logger.error(f"Dataset '{dataset_name}' not found")
            return "Dataset not found"
        
        data = self.datasets[dataset_name] if dataset_name else None
        
        # Route question to the appropriate agent based on content
        if any(term in question.lower() for term in ['clean', 'missing', 'duplicate', 'error']):
            return self.data_cleaning_agent.answer_question(question, data)
        
        elif any(term in question.lower() for term in ['pattern', 'trend', 'anomaly', 'insight']):
            return self.insight_agent.answer_question(question, data)
        
        elif any(term in question.lower() for term in ['predict', 'forecast', 'model']):
            return self.predictive_agent.answer_question(question, data)
        
        elif any(term in question.lower() for term in ['plot', 'chart', 'graph', 'visual']):
            return self.visualization_agent.answer_question(question, data)
        
        else:
            # Default to exploratory agent for general questions
            return self.exploratory_agent.answer_question(question, data)

if __name__ == "__main__":
    # Example usage
    controller = AgentController()
    
    # Example of loading a dataset
    # controller.load_dataset("sales", "path/to/sales_data.csv")
    
    # Example of processing the dataset
    # results = controller.process_dataset("sales")
    
    # Example of asking a question
    # answer = controller.ask_question("What are the top 5 products by revenue?", "sales")
    # print(answer) 