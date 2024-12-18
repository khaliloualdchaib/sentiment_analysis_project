from flask import Flask, jsonify, request
from api.config import AVAILABLE_MODELS
from classifier.review_classifier import ReviewClassifier


class SentimentAPI:
    """
    A class-based Flask API for managing sentiment models and classifying text.
    """

    def __init__(self):
        self.app = Flask(__name__)
        self.review_classifier = ReviewClassifier()
        self._initialize_models(AVAILABLE_MODELS)
        self.define_routes()

        # Global error handler for unexpected exceptions
        @self.app.errorhandler(Exception)
        def handle_exception(e):
            """
            Handle uncaught exceptions and return a JSON response.
            """
            return jsonify({"error": "An unexpected error occurred", "message": str(e)}), 500

    def _initialize_models(self, models_config):
        """
        Load models into the ReviewClassifier from a configuration dictionary.
        """
        for model_path, model_info in models_config.items():
            self.review_classifier.add_model(
                model_path=model_path,
                pipeline_task=model_info["pipeline_task"],
                label_map=model_info["label_map"]
            )

    def define_routes(self):
        """
        Define all API routes.
        """

        @self.app.route("/api/models", methods=["GET"])
        def get_models():
            """
            Return a list of available models.
            """
            return jsonify({"available_models": list(self.review_classifier.models.keys())}), 200

        @self.app.route("/api/predict", methods=["POST"])
        def predict():
            """
            Classify a given text using the specified model.
            """
            try:
                # Check if the request has JSON data
                if not request.is_json:
                    return jsonify({"error": "Invalid input. Expected JSON request body."}), 400

                try:
                    data = request.get_json()
                except Exception as e:
                    return jsonify({"error": "Invalid JSON payload.", "details": str(e)}), 400

                # Validate required fields
                if "text" not in data or "model_path" not in data:
                    return jsonify({"error": "Invalid input. Provide 'text' and 'model_path'."}), 400

                text = data["text"]
                model_path = data["model_path"]

                # Ensure 'text' is a string
                if not isinstance(text, str):
                    return jsonify({"error": "Invalid input. 'text' must be a string."}), 400

                # Check if the text is not empty
                if len(text.strip()) == 0:
                    return jsonify({"error": "Text cannot be empty."}), 400

                # Check if model exists
                if model_path not in self.review_classifier.models:
                    return jsonify({"error": f"Model '{model_path}' not found."}), 404

                # Classify the text
                sentiment, score = self.review_classifier.models[model_path].classify_text(text)

                return jsonify({"model_path": model_path, "sentiment": sentiment.lower()}), 200

            except (KeyError, TypeError) as e:
                return jsonify({"error": "Invalid input. Provide 'text' and 'model_path'.", "details": str(e)}), 400

            except Exception as e:
                return jsonify({"error": "An unexpected error occurred", "message": str(e)}), 500

    def run(self, debug=True):
        """
        Run the Flask app.
        """
        self.app.run(debug=debug)
