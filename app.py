from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModel, pipeline, AutoModelForSequenceClassification
from data_connector import preprocess_text 
from slack_sdk import WebClient 
import os
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

app = Flask(__name__)

# Configurations
EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
CLEANED_DATA_FILE = 'knowledge_base_with_embeddings.csv'
EMBEDDINGS_FILE = 'kb_embeddings.npy'
RESOLUTIONS_FILE = 'ticket_resolutions.csv'
SLACK_CHANNEL_ID = "Past The Channel ID Here"
LOG_FILE_PATH = os.path.join(os.getcwd(), 'usage_log.jsonl')
FINE_TUNED_ISSUE_DIR = 'fine_tuned_model_issue'
FINE_TUNED_TEAM_DIR = 'fine_tuned_model_team'
SEVERITY_LABELS = ['Critical', 'High', 'Medium', 'Low']

# Global data stores
KB_ARTICLES = None
KB_EMBEDDINGS = None
KB_RESOLUTIONS = None
RECOMMENDATION_MODEL = None
RECOMMENDATION_TOKENIZER = None
SEVERITY_CLASSIFIER = None
ISSUE_MODEL = None
ISSUE_TOKENIZER = None
TEAM_MODEL = None
TEAM_TOKENIZER = None


def load_resources():
    """Loads all KB data, embeddings, and the classification models."""
    global KB_ARTICLES, KB_EMBEDDINGS, RECOMMENDATION_MODEL, RECOMMENDATION_TOKENIZER
    global SEVERITY_CLASSIFIER, ISSUE_MODEL, ISSUE_TOKENIZER, TEAM_MODEL, TEAM_TOKENIZER
    global KB_RESOLUTIONS
    models_loaded_successfully = True
    # 1. Load Knowledge Base Articles and Embeddings
    print("--- Loading KB Resources ---")
    try:
        KB_ARTICLES = pd.read_csv(CLEANED_DATA_FILE, sep='|', dtype={'Ticket ID': str})
        KB_EMBEDDINGS = np.load(EMBEDDINGS_FILE)
        print(f"Loaded {len(KB_ARTICLES)} KB articles and embeddings.")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load KB data or embeddings: {e}")
        models_loaded_successfully = False
    # 2. Load KB Resolutions
    try:
        resolutions_df = pd.read_csv(RESOLUTIONS_FILE, dtype={'Ticket ID': str})
        KB_RESOLUTIONS = resolutions_df.set_index('Ticket ID')['Resolution Steps'].to_dict()
        print(f"Loaded {len(KB_RESOLUTIONS)} resolution steps from {RESOLUTIONS_FILE}.")
    except Exception as e:
        print(f"ERROR: Failed to load ticket resolutions: {e}")
        KB_RESOLUTIONS = {}
    # 3. Load Recommendation Model
    print("Loading Recommendation Model...")
    try:
        RECOMMENDATION_TOKENIZER = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
        RECOMMENDATION_MODEL = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME)
        RECOMMENDATION_MODEL.eval()
        print("Recommendation Model loaded.")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load recommendation model: {e}")
        models_loaded_successfully = False
    # 4. Load Classification Models
    print("Loading Classification Models...")
    try:
        if os.path.exists(FINE_TUNED_TEAM_DIR):
            TEAM_MODEL = AutoModelForSequenceClassification.from_pretrained(FINE_TUNED_TEAM_DIR)
            TEAM_TOKENIZER = AutoTokenizer.from_pretrained(FINE_TUNED_TEAM_DIR)
            print(f"Team Model loaded from {FINE_TUNED_TEAM_DIR}.")
        else:
            print(f"WARNING: Team model directory not found at {FINE_TUNED_TEAM_DIR}. Team prediction will fail.")
            models_loaded_successfully = False
        if os.path.exists(FINE_TUNED_ISSUE_DIR):
            ISSUE_MODEL = AutoModelForSequenceClassification.from_pretrained(FINE_TUNED_ISSUE_DIR)
            ISSUE_TOKENIZER = AutoTokenizer.from_pretrained(FINE_TUNED_ISSUE_DIR)
            print(f"Issue Model loaded from {FINE_TUNED_ISSUE_DIR}.")
        else:
            ISSUE_MODEL = TEAM_MODEL
            ISSUE_TOKENIZER = TEAM_TOKENIZER
            print("WARNING: Dedicated Issue model not found. Using Team Model as a placeholder for structure.")
        SEVERITY_CLASSIFIER = pipeline(
            "text-classification",
            model="bhadresh-savani/distilbert-base-uncased-emotion",
            tokenizer="bhadresh-savani/distilbert-base-uncased-emotion"
        )
        print("Severity Classifier (Mock) loaded.")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load classification models: {e}")
        models_loaded_successfully = False
    if not (RECOMMENDATION_MODEL and KB_ARTICLES is not None):
        models_loaded_successfully = False
    return models_loaded_successfully

# Helper function to generate query embedding

def get_query_embedding(text):
    """Generates a normalized embedding for a query text."""
    if RECOMMENDATION_TOKENIZER is None or RECOMMENDATION_MODEL is None:
        return None
    preprocessed_text = preprocess_text(text)
    inputs = RECOMMENDATION_TOKENIZER(preprocessed_text, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = RECOMMENDATION_MODEL(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1)
    embedding = F.normalize(embedding, p=2, dim=1).squeeze().numpy().astype(np.float64)
    return embedding

# Helper function to predict category using a fine-tuned model

def predict_category(text, model, tokenizer):
    """Makes a prediction using a loaded fine-tuned model."""
    if model is None or tokenizer is None:
        return {'label': 'N/A (Model Missing)', 'score': 0.0}
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
    predicted_index = torch.argmax(logits, dim=-1).item()
    try:
        predicted_label = model.config.id2label[predicted_index]
        predicted_score = probabilities[0][predicted_index].item()
    except:
        return {'label': 'N/A (Config Error)', 'score': 0.0}
    return {'label': predicted_label, 'score': float(round(predicted_score, 4))}

# Helper function for Severity Mock

def predict_severity_mock(text, classifier):
    """Mocks severity prediction based on generic sentiment/emotion."""
    if classifier is None:
        return {'label': 'N/A (Classifier Missing)', 'score': 0.0}
    try:
        result = classifier(text)[0]
        emotion = result['label'].lower()
        score = result['score']
        if emotion in ['anger', 'sadness', 'fear']:
            severity = 'Critical'
        elif emotion in ['surprise']:
            severity = 'High'
        elif emotion in ['joy']:
            severity = 'Low'
        else:
            severity = 'Medium'
        return {'label': severity, 'score': float(round(score, 4))}
    except Exception as e:
        print(f"Severity classification failed: {e}")
        return {'label': 'N/A (Processing Error)', 'score': 0.0}

# --- Logging and Slack Integration ---

def log_recommendation_usage(ticket_text, ticket_id, suggestions, severity_prediction, issue_prediction, team_prediction):
    """Logs the interaction for reporting."""
    gap_flag = 1 if not suggestions else 0
    top_suggestion = suggestions[0] if suggestions else {}
    log_entry = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "ticket_id": ticket_id,
        "input_text": ticket_text,
        "gap_flag": gap_flag,
        "severity_prediction": severity_prediction['label'],
        "issue_prediction": issue_prediction['label'],
        "team_prediction": team_prediction['label'],
        "top_kb_id": top_suggestion.get('article_id', 'N/A'),
        "top_kb_similarity": top_suggestion.get('similarity_score', 0.0)
    }
    try:
        with open(LOG_FILE_PATH, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    except Exception as e:
        print(f"ERROR: Failed to write to log file: {e}")

def send_slack_recommendation(channel_id, ticket_id, suggestions, severity, issue, team):
    """Sends a Slack notification about the recommendation."""
    if severity not in ['Critical', 'High']:
        return False

        # --- New Route: Fetch Recent Slack Activity ---
@app.route("/slack_activity", methods=["GET"])
def get_slack_activity():
    """Fetch the latest messages from the configured Slack channel."""
    try:
        client = WebClient(token=os.environ.get("SLACK_BOT_TOKEN", "MOCK_TOKEN"))
        channel_id = SLACK_CHANNEL_ID

        # Mock mode if token is missing
        if client.token == "MOCK_TOKEN":
            print("⚠️ MOCK MODE: Returning simulated Slack messages.")
            return jsonify({
                "messages": [
                    {"user": "Bot", "text": "Mock Slack message #1", "time": time.strftime("%H:%M:%S")},
                    {"user": "Bot", "text": "Mock Slack message #2", "time": time.strftime("%H:%M:%S")}
                ]
            })

        # Fetch latest 10 messages from Slack
        response = client.conversations_history(channel=channel_id, limit=5)
        messages = []
        for msg in response.get("messages", []):
            user_id = msg.get("user", "Unknown")
            text = msg.get("text", "")
            ts = msg.get("ts", "")
            readable_time = time.strftime("%H:%M:%S", time.localtime(float(ts.split('.')[0]))) if ts else "N/A"
            messages.append({"user": user_id, "text": text, "time": readable_time})

        return jsonify({"messages": messages})

    except Exception as e:
        print(f"ERROR fetching Slack messages: {e}")
        return jsonify({"messages": [], "error": str(e)}), 500

    try:
        client = WebClient(token=os.environ.get("SLACK_BOT_TOKEN", "MOCK_TOKEN"))
        top_suggestion = suggestions[0] if suggestions else None
        blocks = [
            {"type": "section", "text": {"type": "mrkdwn", "text": f"*{ticket_id}: New Recommendation Generated*"}},
            {"type": "divider"},
            {"type": "section", "fields": [
                {"type": "mrkdwn", "text": f"*Severity:*\n{severity}"},
                {"type": "mrkdwn", "text": f"*Issue Type:*\n{issue}"},
                {"type": "mrkdwn", "text": f"*Assigned Team:*\n{team}"}
            ]}
        ]
        if top_suggestion:
            blocks.extend([
                {"type": "section", "text": {"type": "mrkdwn", "text": f":sparkles: *Top KB Suggestion ({top_suggestion['similarity_score']*100:.1f}%)*: {top_suggestion['title']}"}},
                {"type": "context", "elements": [
                    {"type": "mrkdwn", "text": f"ID: {top_suggestion['article_id']} | Summary: {top_suggestion['summary']}"}
                ]}
            ])
        else:
            blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": ":warning: *Content Gap Detected*: No relevant KB article found (Similarity < 50%)."}})
        if client.token == "MOCK_TOKEN":
            print(f"SLACK MOCK: Sending message to {channel_id} for ticket {ticket_id}")
        else:
            client.chat_postMessage(channel=channel_id, blocks=blocks, text=f"Recommendation for Ticket {ticket_id}")
        return True
    except Exception as e:
        print(f"ERROR: Failed to send Slack notification: {e}")
        return False

# --- Flask Route ---

@app.route('/recommend', methods=['POST'])
def recommend():
    """Endpoint to receive ticket text and return recommendations and predictions."""
    data = request.get_json()
    ticket_text = data.get('ticket_text', '')
    ticket_id = data.get('ticket_id', f"TKT-{int(time.time() * 100) % 10000}")
    if not ticket_text or KB_EMBEDDINGS is None:
        return jsonify({"error": "Invalid input or resources not loaded. Check server logs."}), 500
    query_embedding = get_query_embedding(ticket_text)
    if query_embedding is None:
        return jsonify({"error": "Failed to generate query embedding."}), 500
    classification_text = str(ticket_text)
    severity_prediction = predict_severity_mock(classification_text, SEVERITY_CLASSIFIER)
    issue_prediction = predict_category(classification_text, ISSUE_MODEL, ISSUE_TOKENIZER)
    team_prediction = predict_category(classification_text, TEAM_MODEL, TEAM_TOKENIZER)
    similarities = np.dot(KB_EMBEDDINGS, query_embedding)
    TOP_N = 5
    top_indices = np.argsort(similarities)[::-1][:TOP_N]
    suggestions = []
    for i in top_indices:
        similarity = similarities[i]
        if similarity > 0.5:
            article = KB_ARTICLES.iloc[i]
            article_id_raw = article.get('Ticket ID', i)
            article_id_formatted = str(article_id_raw)
            try:
                num_id = int(article_id_raw)
                article_id_formatted = f"TKT-{num_id:03d}"
            except ValueError:
                pass
            resolution_steps = KB_RESOLUTIONS.get(
                article_id_formatted,
                "Resolution steps unavailable."
            )
            suggestions.append({
                "article_id": article_id_formatted,
                "title": article.get('Subject', 'No Subject'),
                "similarity_score": float(round(similarity, 4)),
                "summary": article.get('Full_Ticket_Text', 'N/A')[:100] + '...',
                "resolution_steps": resolution_steps
            })
    log_recommendation_usage(
        ticket_text,
        ticket_id,
        suggestions,
        severity_prediction=severity_prediction,
        issue_prediction=issue_prediction,
        team_prediction=team_prediction
    )
    slack_success = send_slack_recommendation(
        SLACK_CHANNEL_ID,
        ticket_id,
        suggestions,
        severity_prediction['label'],
        issue_prediction['label'],
        team_prediction['label']
    )
    return jsonify({
        "suggestions": suggestions,
        "severity_prediction": severity_prediction,
        "issue_prediction": issue_prediction,
        "team_prediction": team_prediction,
        "slack_sent": slack_success
    })

if __name__ == '__main__':
    if load_resources():
        app.run(host='127.0.0.1', port=5000, debug=True, use_reloader=False)
    else:
        print("Application failed to start due to critical resource loading errors.")