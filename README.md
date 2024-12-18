# Sentiment Analysis Project

## Setup

Follow these steps to set up the project:

1. **Create and Activate a Virtual Environment**
   Run the following commands in your project directory:
   - On Windows:
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     python -m venv venv
     source venv/bin/activate
     ```

2. **Install Dependencies**
   Use the following command to install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Setup Script**
   Initialize the project by running:
   ```bash
   python setup.py
   ```

---

## `main.py` Overview

The `main.py` script contains two primary functions:

1. **`classify_all`**
   - This function generates a CSV file containing classifications made by the models. It benchmarks the performance of two models, comparing their accuracy, precision, and recall.

2. **`test_endpoints`**
   - This function sends requests to the API to verify its functionality and demonstrate how to use it.

### Using the API

To interact with the API, follow these steps:

1. **Start the Server**
   Open a separate terminal and run:
   ```bash
   python start_server.py
   ```

2. **Send Requests**
   - You can use the `test_endpoints` function in `main.py` or manually send custom requests to the server.

   The API provides the following endpoints:

   #### 1. `/api/models` (GET)
   - **Purpose**: Retrieve a list of all available models on the server.

   #### 2. `/api/predict` (POST)
   - **Purpose**: Get a sentiment prediction for a given text using a specified model.
   - **Request Payload** (JSON):
     ```json
     {
         "text": "This movie is amazing! I love it.",
         "model_path": "siebert/sentiment-roberta-large-english"
     }
     ```
   - **Response Example** (JSON):
     ```json
     {
         "model_path": "siebert/sentiment-roberta-large-english",
         "sentiment": "positive"
     }
     ```

---

## Creating the CSV File and Benchmarking Models

To generate a CSV file containing classifications from both models, along with their confidence scores for each review in the dataset, run the `classify_all` function in `main.py`.

### Benchmarking Results

Here are the benchmark results for the two models tested:

| Model                                      | Accuracy | Precision   | Recall       |
|--------------------------------------------|----------|-------------|---------------|
| `omidroshani/imdb-sentiment-analysis`      | 0.95     | 0.9744      | 0.9048        |
| `siebert/sentiment-roberta-large-english`  | 0.95     | 0.9512      | 0.9286        |

