from slack_sdk import WebClient
import os

token = os.environ.get("SLACK_BOT_TOKEN")
client = WebClient(token=token)
channel_id = "C09RL83SCTG"  # Replace with your real one

try:
    resp = client.chat_postMessage(channel=channel_id, text="✅ Test message from AI Ticket Classifier")
    print("Success:", resp["ok"])
except Exception as e:
    print("Slack Error:", e)
