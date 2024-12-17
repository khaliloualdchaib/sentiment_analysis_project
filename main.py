from classifier.review_classifier import ReviewClassifier
import nltk
import requests
import json

BASE_URL = "http://127.0.0.1:5000/api"

def classify_all():
    # Initialize review classifier
    classifier = ReviewClassifier()

    # Load dataset
    data_path = "data/IMDB-movie-reviews.csv"
    df = classifier.load_dataset(data_path)

    # Add models
    classifier.add_model(
        model_path="omidroshani/imdb-sentiment-analysis",
        pipeline_task="text-classification",
        label_map={"LABEL_0": "NEGATIVE", "LABEL_1": "POSITIVE"}
    )
    
    classifier.add_model(
        model_path="siebert/sentiment-roberta-large-english",
        pipeline_task="sentiment-analysis",
        label_map={"NEGATIVE": "NEGATIVE", "POSITIVE": "POSITIVE"}
    )

    # Classify reviews
    classified_df = classifier.classify_reviews(df["review"])
    classified_df.to_csv("classified_reviews.csv", index=False)

    # Benchmark models
    metrics = classifier.benchmark_models(classified_df, df['sentiment'])
    print(metrics)

def test_endpoints():
    """
    Test the /api/models and /api/predict endpoints with various test cases.
    """
    print("Testing GET /api/models...")
    response = requests.get(f"{BASE_URL}/models")
    if response.status_code == 200:
        print("Success! Available models:", response.json()["available_models"])
    else:
        print("Failed to retrieve models. Status Code:", response.status_code, "Response:", response.json())
    
    # Define test cases
    test_cases = [
        {"name": "Valid input - positive text", "text": "I love this movie!", "model_path": "omidroshani/imdb-sentiment-analysis"},
        {"name": "Valid input - negative text", "text": "This is the worst experience I've ever had.", "model_path": "siebert/sentiment-roberta-large-english"},
        {"name": "Valid input - neutral text", "text": "This movie was just okay.", "model_path": "siebert/sentiment-roberta-large-english"},
        {"name": "Invalid model path", "text": "Completely disappointed with the product.", "model_path": "invalid/model-path"},
        {"name": "Empty text", "text": "", "model_path": "siebert/sentiment-roberta-large-english"},
        {"name": "Missing text field", "model_path": "siebert/sentiment-roberta-large-english"},
        {"name": "Missing model_path field", "text": "I love this product!"},
        {"name": "Non-string text input", "text": 12345, "model_path": "siebert/sentiment-roberta-large-english"},
        {"name": "Invalid JSON payload", "payload": "invalid-json"}
    ]

    # Iterate over each test case
    for case in test_cases:
        print(f"\nTesting POST /api/predict: {case.get('name', 'Unnamed Test Case')}")

        # Build payload
        if "payload" in case:  # Special case for invalid JSON
            payload = case["payload"]
            headers = {"Content-Type": "application/json"}
            data = payload
        else:
            payload = {}
            if "text" in case:
                payload["text"] = case["text"]
            if "model_path" in case:
                payload["model_path"] = case["model_path"]
            headers = {"Content-Type": "application/json"}
            data = json.dumps(payload)

        try:
            response = requests.post(f"{BASE_URL}/predict", headers=headers, data=data)
            print(f"Request Payload: {data}")

            if response.status_code == 200:
                print("Success! Sentiment prediction:", response.json())
            elif response.status_code == 404:
                print("Error: Model not found. Response:", response.json())
            elif response.status_code == 400:
                print("Error: Invalid input. Response:", response.json())
            else:
                print("Unexpected error. Status Code:", response.status_code, "Response:", response.json())
        except Exception as e:
            print("Exception occurred while sending request:", str(e))


if __name__ == "__main__":
    # Ensure NLTK's tokenizer is available
    nltk.download('punkt')
    #classify_all()
    test_endpoints()