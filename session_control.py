from threading import Timer
from dotenv import load_dotenv

# Global variables
# weaviate_client = None
# global weaviate_client, is_connected, idle_timer, idle_timeout
# weaviate_client = None
idle_timer = None
idle_timeout = 300  # 5 minutes
is_connected = False

import os
import weaviate
from weaviate.classes.init import Auth

weaviate_url = os.getenv("WEAVIATE_URL")
weaviate_api_key = os.getenv("WEAVIATE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Function to initialize Weaviate client
def connect_weaviate():
    global weaviate_client, is_connected
    if not is_connected:
        weaviate_client = weaviate.connect_to_weaviate_cloud(
            cluster_url=weaviate_url,
            auth_credentials=Auth.api_key(weaviate_api_key),
            headers={"X-OpenAI-Api-Key": openai_api_key}
        )
        is_connected = True
        print(weaviate_client)
        print("Connected to Weaviate")

# Function to close Weaviate client
def close_weaviate():
    global weaviate_client, is_connected
    if is_connected and weaviate_client is not None:
        weaviate_client.close()
        weaviate_client = None
        is_connected = False
        print("Closed Weaviate connection")

# Function to reset the idle timer
def reset_idle_timer():
    global idle_timer
    if idle_timer is not None:
        idle_timer.cancel()  # Cancel the previous timer
    idle_timer = Timer(idle_timeout, handle_user_idle)
    idle_timer.start()
    print("Idle timer reset")

# Function to handle user idle timeout
def handle_user_idle():
    print("User is idle. Closing Weaviate connection...")
    close_weaviate()

