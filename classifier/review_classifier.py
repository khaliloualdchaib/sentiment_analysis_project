from typing import Dict
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score
from classifier.sentiment_model import SentimentModel

class ReviewClassifier:
    """
    ReviewClassifier is a class designed to streamline the process of sentiment analysis on datasets 
    containing textual reviews. It supports dataset loading, multi-model sentiment classification, 
    and performance benchmarking for added analytical insights.
    """

    def __init__(self):
        """
        Initializes the ReviewClassifier instance.

        Attributes:
            models (Dict[str, SentimentModel]): A dictionary to store multiple sentiment models, 
                                                identified by their names.
        """
        self.models = {}

    def load_dataset(self, file_path: str) -> pd.DataFrame:
        """
        Loads a dataset of reviews from a specified CSV file.

        Parameters:
            file_path (str): Path to the CSV file containing the dataset. The file should have 
                             a delimiter of ";" and be encoded in 'Windows-1252'.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the dataset loaded from the CSV file.
        """
        return pd.read_csv(file_path, encoding='Windows-1252', delimiter=";")

    def add_model(self, model_path: str, pipeline_task: str, label_map: Dict[str, str]):
        """
        Adds a sentiment analysis model to the classifier.

        Parameters:
            model_path (str): Name or path of the pre-trained model to be used.
            pipeline_task (str): Task type for the Hugging Face pipeline (e.g., 'text-classification').
            label_map (Dict[str, str]): A mapping of the model's output labels to desired sentiment labels 
                                        (e.g., {"LABEL_0": "negative", "LABEL_1": "positive"}).
        """
        self.models[model_path] = SentimentModel(model_path, pipeline_task, label_map)

    def classify_reviews(self, reviews: pd.Series) -> pd.DataFrame:
        """
        Classifies a series of reviews using all registered sentiment models.

        Each review is processed through all the added models, and the results are aggregated 
        into a single DataFrame.

        Parameters:
            reviews (pd.Series): A pandas Series containing textual reviews.

        Returns:
            pd.DataFrame: A DataFrame where each row corresponds to a review and includes sentiment 
                          predictions from all models.
        """
        all_results = []

        for review in tqdm(reviews, desc="Classifying reviews"):
            review_results = {"review": review}

            for model_path, model in self.models.items():
                # Get sentiment from the model and store it in lowercase
                sentiment, score = model.classify_text(review)
                review_results[model_path] = sentiment
                review_results[f"{model_path} score"] = score
                
            all_results.append(review_results)

        return pd.DataFrame(all_results)

    def benchmark_models(self, predictions: pd.DataFrame, labels: pd.Series) -> Dict[str, Dict[str, float]]:
        """
        Benchmarks the performance of each sentiment model using evaluation metrics.

        This method calculates accuracy, precision, and recall for each model based on its predictions 
        compared to the actual labels.

        Parameters:
            predictions (pd.DataFrame): A DataFrame containing the predictions made by the models. 
                                        Columns should correspond to model names with their sentiments.
            labels (pd.Series): A pandas Series containing the ground truth sentiment labels.

        Returns:
            Dict[str, Dict[str, float]]: A dictionary where each key is a model name, and the value 
                                         is a dictionary of performance metrics (accuracy, precision, recall).
        """
        metrics = {}
        for model_path in self.models.keys():
            # Extract the predictions for the current model
            predicted_labels = predictions[model_path]
            # Calculate evaluation metrics
            accuracy = accuracy_score(labels, predicted_labels)
            precision = precision_score(labels, predicted_labels, pos_label='positive', zero_division=0)
            recall = recall_score(labels, predicted_labels, pos_label='positive', zero_division=0)

            metrics[model_path] = {
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall
            }
        return metrics
