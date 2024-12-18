from typing import List, Dict, Tuple
from transformers import pipeline, AutoTokenizer
import nltk

class SentimentModel:
    """
    SentimentModel is a class designed to encapsulate the logic required for sentiment analysis using 
    pre-trained models from the Hugging Face Transformers library. 

    This class supports advanced features like handling token limits through text chunking 
    and label mapping for customized sentiment outputs.
    """

    def __init__(self, model_path: str, pipeline_task: str, label_map: Dict[str, str]):
        """
        Initializes the SentimentModel.

        Parameters:
            model_path (str): The path of the pre-trained model to be used for sentiment analysis.
            pipeline_task (str): The task type for the Hugging Face pipeline.
            label_map (Dict[str, str]): A dictionary mapping model output labels to desired sentiment labels 
                                                  (e.g., {"LABEL_0": "negative", "LABEL_1": "positive"}).
        """
        self.model_path = model_path
        self.pipeline_task = pipeline_task
        self.label_map = label_map
        self.pipeline = pipeline(pipeline_task, model=model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.max_length = 512  # Default maximum token length for the tokenizer

    def classify_text(self, text: str) -> Tuple[str, float]:
        """
        Classifies the sentiment of the input text.

        The method processes the input text by splitting it into chunks if necessary, 
        performs classification using the pre-trained model, and aggregates the results 

        Parameters:
            text (str): The input text to be classified.

        Returns:
            Tuple[string, float]: The aggregated sentiment label for the input text and the confidence score.
        """
        # Split the text into chunks to fit within the tokenizer's token limit
        chunks = self.chunk_text(text)
        predictions = self.pipeline(chunks, padding=True)
        positive_scores = []
        negative_scores = []

        for pred in predictions:
            label = self.label_map[pred['label']]
            if label == "POSITIVE":
                positive_scores.append(pred['score'])
            elif label == "NEGATIVE":
                negative_scores.append(pred['score'])
        
        # Compute average scores for each label
        avg_positive_score = sum(positive_scores) / len(positive_scores) if positive_scores else 0
        avg_negative_score = sum(negative_scores) / len(negative_scores) if negative_scores else 0
        
        # Determine the final sentiment and its average score
        if avg_positive_score >= avg_negative_score:
            return "positive", avg_positive_score
        return "negative", avg_negative_score
    

    def chunk_text(self, text: str) -> List[str]:
        """
        Splits the input text into chunks to ensure compatibility with the tokenizer's 
        maximum token length. This method ensures that longer texts can be processed 
        by dividing them into manageable chunks without exceeding the token limit.

        Parameters:
            text (str): The input text to be split into chunks.

        Returns:
            List[str]: A list of text chunks, each fitting within the tokenizer's maximum token limit.
        """
        sentences = nltk.sent_tokenize(text)  # Break the text into sentences
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            # Tokenize the sentence without adding special tokens
            tokenized_sentence = self.tokenizer.encode(sentence, add_special_tokens=False)
            sentence_length = len(tokenized_sentence)

            # If adding the current sentence exceeds the token limit, finalize the current chunk
            if current_length + sentence_length > self.max_length - 2:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length

        # Add any remaining sentences as the last chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks
