AVAILABLE_MODELS = {
    "omidroshani/imdb-sentiment-analysis": {
        "pipeline_task": "text-classification",
        "label_map": {"LABEL_0": "NEGATIVE", "LABEL_1": "POSITIVE"}
    },
    "siebert/sentiment-roberta-large-english": {
        "pipeline_task": "sentiment-analysis",
        "label_map": {"NEGATIVE": "NEGATIVE", "POSITIVE": "POSITIVE"}
    }
}
