from collections import defaultdict
import requests
from flask import Flask, render_template, jsonify, request
from langchain_core.messages import HumanMessage, AIMessage
import time
from Sales_Consulting_Chatbot import *
from session_control import *
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Flask App Initialization
app = Flask(__name__)
application = app

# Chatwoot API Configuration
BASE_URL = "https://app.chatwoot.com"  # Replace with your instance URL
API_TOKEN = os.getenv("CHATWOOT_API_TOKEN")
ACCOUNT_ID = os.getenv("ACCOUNT_ID")
INBOX_ID = os.getenv("INBOX_ID")
AGENT_ID = os.getenv("AGENT_ID")  # ID of the consultant or agent

# Global variables
processed_messages = defaultdict(lambda: None)  # Store the last message timestamp per conversation_id
MESSAGE_PROCESSING_TIMEOUT = 5  # Ignore duplicate messages within this time frame (in seconds)

# Route to render the homepage
@app.route('/')
def hello_world():
    return render_template('index.html')

# Webhook to handle incoming messages from Chatwoot
@app.route("/api/webhook", methods=["POST"])
def webhook():
    try:
        # Parse the incoming JSON payload
        body = request.json
        print("Received payload:", body)

        # Log the event
        event_type = body.get("event")
        print(f"Received event: {event_type}")

        if event_type == "message_created":
            # User sent a message, connect to Weaviate and reset idle timer
            connect_weaviate()
            reset_idle_timer()
        elif event_type == "conversation_resolved":
            # Conversation ended, close Weaviate client
            print("Conversation ended. Closing Weaviate connection...")
            close_weaviate()

        # Extract the event type
        event_type = body.get("event")
        print("Event type:", event_type)

        # Handle only 'message_created' events
        if event_type == "message_created":
            # Extract relevant data from the incoming payload
            conversation = body.get("conversation", {})
            messages = conversation.get("messages", [])

            if messages:
                conversation_id = messages[0].get("conversation_id")
                content = messages[0].get("content")
                sender_type = messages[0].get("sender_type")
                assignee_id = conversation.get("assignee_id")  # Check the current assignee status
            else:
                conversation_id = None
                content = None
                sender_type = None
                assignee_id = None

            print("Message content:", content)
            print("Conversation ID:", conversation_id)
            print("Sender Type:", sender_type)
            print("Assignee ID:", assignee_id)

            # Ignore messages not sent by the user
            if sender_type != "Contact":
                print("Message is not from a user. Ignoring...")
                return jsonify({"status": "ignored", "message": "Message is not from a user."}), 200

            # Validate content and conversation_id
            if not content or not conversation_id:
                print("Missing content or conversation_id.")
                return jsonify({"status": "ignored", "message": "Missing content or conversation_id."}), 200

            # If the conversation is assigned, stop chatbot processing
            if assignee_id is not None:
                print(f"Conversation {conversation_id} is assigned to agent {assignee_id}. Stopping chatbot response.")
                return jsonify({"status": "ignored", "message": "Chatbot disabled for assigned conversation."}), 200

            # Check if user explicitly requests a consultant
            if content.lower() in ["talk to consultant", "need human help", "consultant please"]:
                print("User requested a consultant.")

                # Assign the conversation to a consultant
                assign_to_consultant(conversation_id, AGENT_ID)

                return jsonify({"status": "success", "message": "Conversation assigned to a consultant!"}), 200

            # Process the user's message and generate a response (for unassigned conversations)
            chat_history.append(HumanMessage(content=content))
            result = qa.invoke({"input": content, "chat_history": chat_history})
            chat_history.append(AIMessage(content=result["answer"]))

            # Send the bot's response
            send_message_to_chatwoot(conversation_id, result["answer"])

            return jsonify({"status": "success", "message": "Chatbot response processed."}), 200

        else:
            print("Unhandled event type:", event_type)
            return jsonify({"status": "ignored", "message": f"Unhandled event: {event_type}"}), 200

    except Exception as e:
        print(f"Unexpected error in webhook: {e}")
        return jsonify({"status": "error", "details": str(e)}), 500


def send_message_to_chatwoot(conversation_id, message_content):
    """
    Sends a message to Chatwoot as a bot response.
    """
    url = f"{BASE_URL}/api/v1/accounts/{ACCOUNT_ID}/conversations/{conversation_id}/messages"
    headers = {"api_access_token": API_TOKEN}
    data = {"content": message_content}
    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200:
        print("Message sent successfully to Chatwoot.")
    else:
        print(f"Failed to send message: {response.status_code}, {response.text}")



def set_unassigned(conversation_id):
    """
    Sets a conversation to Unassigned status (assignee = null).
    """
    url = f"{BASE_URL}/api/v1/accounts/{ACCOUNT_ID}/conversations/{conversation_id}/assignments"
    headers = {"api_access_token": API_TOKEN}
    data = {"assignee_id": None}  # Set assignee to null
    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200:
        print(f"Conversation {conversation_id} successfully set to Unassigned status.")
    else:
        print(f"Failed to set conversation to Unassigned status: {response.status_code}, {response.text}")


def assign_to_consultant(conversation_id, consultant_id):
    """
    Assigns a conversation to a specific consultant in Chatwoot.
    """
    url = f"{BASE_URL}/api/v1/accounts/{ACCOUNT_ID}/conversations/{conversation_id}/assignments"
    headers = {"api_access_token": API_TOKEN}
    data = {"assignee_id": consultant_id}  # Include the assignee_id in the body
    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200:
        print(f"Conversation {conversation_id} successfully assigned to consultant {consultant_id}.")
    else:
        print(f"Failed to assign consultant: {response.status_code}, {response.text}")



# Fetch all agents
def get_agents():
    url = f"{BASE_URL}/api/v1/accounts/{ACCOUNT_ID}/agents"
    headers = {
        "api_access_token": API_TOKEN
    }

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        print(response.json())
        agents = response.json()["data"]
        for agent in agents:
            print(f"Agent Name: {agent['name']}, Agent ID: {agent['id']}")
        return agents
    else:
        print(f"Failed to fetch agents: {response.status_code}, {response.text}")
        return None

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True, port=5000)
