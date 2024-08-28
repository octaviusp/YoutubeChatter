import json
import os
import shutil
import sys
from typing import Optional, Tuple

from groq import Groq
import pytube
import requests
from colorama import Fore, Style, init
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory

from audio_extract import extract_audio

# Initialize colorama for cross-platform color support
init()

# Load environment variables
load_dotenv()

# Constants
DEBUG_MODE = False
TRANSCRIPTION_DIR = 'transcriptions'

# Check for API keys
if not any(os.getenv(key) for key in ["GROQ_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"]):
    print(Fore.RED + "❌ Error: No API keys found in environment variables." + Style.RESET_ALL)
    print(Fore.YELLOW + "Please add GROQ_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY as environment variables and try again." + Style.RESET_ALL)
    sys.exit(1)

# Check for ffmpeg installation
if shutil.which("ffmpeg") is None:
    print(Fore.RED + "❌ Error: ffmpeg is not installed or not in PATH." + Style.RESET_ALL)
    print(Fore.YELLOW + "Please install ffmpeg and ensure it's in your system PATH." + Style.RESET_ALL)
    print(Fore.CYAN + "Installation instructions:")
    print("- On macOS: brew install ffmpeg")
    print("- On Ubuntu/Debian: sudo apt-get install ffmpeg")
    print("- On Windows: Download from https://ffmpeg.org/download.html and add to PATH" + Style.RESET_ALL)
    sys.exit(1)

# Set up memory
memory = ConversationBufferMemory(return_messages=True)

def initialize_llm(model_choice: str):
    model_map = {
        '1': (ChatGroq, "GROQ_API_KEY", "llama-3.1-70b-versatile"),
        '2': (ChatOpenAI, "OPENAI_API_KEY", "gpt-4o"),
        '3': (ChatAnthropic, "ANTHROPIC_API_KEY", "claude-3-5-sonnet-20240620")
    }
    
    model_class, api_key_var, model_name = model_map.get(model_choice, (None, None, None))
    
    if model_class is None or not os.getenv(api_key_var):
        print(Fore.RED + f"❌ Error: {api_key_var} is required for the selected model." + Style.RESET_ALL)
        sys.exit(1)

    if model_class == ChatAnthropic:
        return model_class(
            model=model_name,
            anthropic_api_key=os.getenv(api_key_var),
            temperature=0.5,
            max_tokens=4096
        )
    else:
        return model_class(
            model=model_name,
            temperature=0.5,
            max_tokens=4096,
            max_retries=2,
            api_key=os.getenv(api_key_var)
        )

def print_welcome():
    welcome_message = r"""

    \033[1;32m
    ██╗   ██╗ ██████╗ ██╗   ██╗████████╗██╗   ██╗██████╗ ███████╗ ██████╗██╗  ██╗ █████╗ ████████╗
    ╚██╗ ██╔╝██╔═══██╗██║   ██║╚══██╔══╝██║   ██║██╔══██╗██╔════╝██╔════╝██║  ██║██╔══██╗╚══██╔══╝
     ╚████╔╝ ██║   ██║██║   ██║   ██║   ██║   ██║██████╔╝█████╗  ██║     ███████║███████║   ██║   
      ╚██╔╝  ██║   ██║██║   ██║   ██║   ██║   ██║██╔══██╗██╔══╝  ██║     ██╔══██║██╔══██║   ██║   
       ██║   ╚██████╔╝╚██████╔╝   ██║   ╚██████╔╝██████╔╝███████╗╚██████╗██║  ██║██║  ██║   ██║   
       ╚═╝    ╚═════╝  ╚═════╝    ╚═╝    ╚═════╝ ╚═════╝ ╚══════╝ ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   
    \033[0m
    """
    print(Fore.CYAN + Style.BRIGHT + welcome_message + Style.RESET_ALL)
    print(Fore.YELLOW + "Welcome to YouTube Chatter! Let's discuss your favorite videos." + Style.RESET_ALL)
    print(Fore.CYAN + "\nBefore we begin, please ensure you have the following:" + Style.RESET_ALL)
    print(Fore.GREEN + "✅ GROQ_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY set as environment variables")
    print("✅ ffmpeg installed and in your system PATH")
    print("✅ Python packages: pytube, langchain_groq, langchain_openai, langchain_anthropic, python-dotenv, colorama, requests" + Style.RESET_ALL)
    print(Fore.YELLOW + "\nTo use this app:")
    print("1. Choose to chat about a new YouTube video or a previously transcribed one.")
    print("2. For new videos, provide a valid YouTube URL.")
    print("3. The app will extract audio, transcribe it, and allow you to chat about the content.")
    print("4. You can save transcriptions for future use to save time.")
    print("5. Ask questions about the video content, and the AI will respond based on the transcription.")
    print("6. Type 'exit' or press Ctrl+C at any time to end the conversation." + Style.RESET_ALL)
    print(Fore.CYAN + "\nLet's get started!" + Style.RESET_ALL)

def transcribe_audio(audio_bytes):
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        transcription = client.audio.transcriptions.create(
            file=("audio.mp3", audio_bytes),
            model="whisper-large-v3",
            response_format="json"
        )
        if DEBUG_MODE:
            print(Fore.YELLOW + f"Debug: Transcription object: {transcription}" + Style.RESET_ALL)
        return transcription.text
    except requests.exceptions.RequestException as e:
        handle_transcription_error(e)
    except Exception as e:
        print(Fore.RED + f"❌ An unexpected error occurred during transcription: {str(e)}" + Style.RESET_ALL)
    return None

def handle_transcription_error(e):
    if e.response is not None:
        status_code = e.response.status_code
        error_messages = {
            413: ("❌ Error: The audio file is too large for transcription.",
                  "ℹ️  Groq's Whisper API has a file size limit of 25 MB.",
                  "   Try reducing the file size using ffmpeg before uploading:",
                  "   ffmpeg -i input.mp3 -ar 16000 -ac 1 -b:a 64k output.mp3"),
            429: ("❌ Error: Too many requests. Please try again later.",),
            500: ("❌ Error: Server error. Please try again later or contact support.",),
        }
        messages = error_messages.get(status_code, (f"❌ Error: An unexpected error occurred (Status code: {status_code})",))
        for message in messages:
            print(Fore.RED + message + Style.RESET_ALL)
    else:
        print(Fore.RED + f"❌ Error: An unexpected error occurred: {str(e)}" + Style.RESET_ALL)

def chat_with_ai(transcription, model_choice: str):
    model_classes = {
        '1': ChatGroq,
        '2': ChatOpenAI,
        '3': ChatAnthropic
    }
    model_names = {
        '1': "Groq",
        '2': "GPT",
        '3': "Anthropic"
    }
    model_class = model_classes.get(model_choice)
    model_name = model_names.get(model_choice, "AI")

    if not model_class:
        print(Fore.RED + "❌ Invalid model choice. Please select a valid model." + Style.RESET_ALL)
        return

    llm = initialize_llm(model_choice)

    print(Fore.GREEN + f"\n🤖 {model_name}: Hello! I've analyzed the video transcription. What would you like to know about the video?" + Style.RESET_ALL)

    while True:
        try:
            user_input = input(Fore.BLUE + "👤 You: " + Style.RESET_ALL)
            if user_input.lower() in ['exit', 'quit', 'q']:
                print(Fore.YELLOW + "\nConversation ended by user. Thank you for using YouTube Chatter!" + Style.RESET_ALL)
                break

            # Add the transcription to the input for context
            full_input = f"Based on this video transcription: {transcription}\n\nUser question: {user_input}"

            # Get the response from the chat model
            response: AIMessage = llm.invoke([HumanMessage(content=full_input)])
                
            print(Fore.GREEN + f"\n🤖 {model_name}: {response.content}" + Style.RESET_ALL)

            # Add the interaction to memory
            memory.chat_memory.add_user_message(user_input)
            memory.chat_memory.add_ai_message(response)

        except KeyboardInterrupt:
            print(Fore.YELLOW + "\nConversation ended by user. Thank you for using YouTube Chatter!" + Style.RESET_ALL)
            break
        except Exception as e:
            print(Fore.RED + f"\n❌ An error occurred during the conversation: {e}" + Style.RESET_ALL)
            print(Fore.YELLOW + "Let's try to continue our conversation." + Style.RESET_ALL)

def get_video_id(url: str) -> Optional[str]:
    try:
        yt = pytube.YouTube(url)
        return yt.video_id
    except Exception as e:
        print(Fore.RED + f"❌ Error extracting video ID: {e}" + Style.RESET_ALL)
        return None

def get_video_title(url: str) -> str:
    try:
        yt = pytube.YouTube(url)
        return yt.title
    except Exception as e:
        print(Fore.RED + f"❌ Error extracting video title: {e}" + Style.RESET_ALL)
        return "Unknown Title"

def get_or_create_transcription(youtube_url: str, save_transcription: bool = True) -> Tuple[Optional[str], Optional[str]]:
    video_id = get_video_id(youtube_url)
    if not video_id:
        return None, None

    video_title = get_video_title(youtube_url)
    transcription_file = os.path.join(TRANSCRIPTION_DIR, f"{video_id}.json")
    
    if os.path.exists(transcription_file):
        return load_cached_transcription(transcription_file, video_title)
    
    return create_new_transcription(youtube_url, video_id, video_title, save_transcription)

def load_cached_transcription(transcription_file: str, video_title: str) -> Tuple[Optional[str], Optional[str]]:
    print(Fore.CYAN + "📁 Using cached transcription..." + Style.RESET_ALL)
    try:
        with open(transcription_file, 'r') as f:
            data = json.load(f)
            return data.get('transcription'), data.get('title', video_title)
    except json.JSONDecodeError as e:
        print(Fore.RED + f"❌ Error reading cached transcription: {e}" + Style.RESET_ALL)
    return None, None

def create_new_transcription(youtube_url: str, video_id: str, video_title: str, save_transcription: bool) -> Tuple[Optional[str], Optional[str]]:
    try:
        print(Fore.CYAN + "🎵 Extracting audio from video..." + Style.RESET_ALL)
        audio_bytes = extract_audio(youtube_url)
    except Exception as e:
        print(Fore.RED + f"❌ Error extracting audio: {e}" + Style.RESET_ALL)
        return None, None
    
    try:
        print(Fore.CYAN + "🗣️ Transcribing audio..." + Style.RESET_ALL)
        transcription = transcribe_audio(audio_bytes)
    except Exception as e:
        print(Fore.RED + f"❌ Error transcribing audio: {e}" + Style.RESET_ALL)
        return None, None
    
    if transcription:
        if save_transcription:
            save_transcription_to_file(video_id, transcription, video_title)
        return transcription, video_title
    else:
        print(Fore.RED + "❌ Transcription failed." + Style.RESET_ALL)
        return None, None

def save_transcription_to_file(video_id: str, transcription: str, video_title: str):
    try:
        os.makedirs(TRANSCRIPTION_DIR, exist_ok=True)
        transcription_file = os.path.join(TRANSCRIPTION_DIR, f"{video_id}.json")
        with open(transcription_file, 'w') as f:
            json.dump({'transcription': transcription, 'title': video_title}, f)
        print(Fore.GREEN + "✅ Transcription saved for future use." + Style.RESET_ALL)
    except Exception as e:
        print(Fore.RED + f"❌ Error saving transcription: {e}" + Style.RESET_ALL)

def list_saved_transcriptions():
    transcriptions = []
    if not os.path.exists(TRANSCRIPTION_DIR):
        return transcriptions
    for file in os.listdir(TRANSCRIPTION_DIR):
        if file.endswith('.json'):
            try:
                with open(os.path.join(TRANSCRIPTION_DIR, file), 'r') as f:
                    data = json.load(f)
                    transcriptions.append((file[:-5], data.get('title', 'Unknown Title')))
            except json.JSONDecodeError as e:
                print(Fore.RED + f"❌ Error reading {file}: {e}. Skipping." + Style.RESET_ALL)
    return transcriptions

def print_menu():
    print(Fore.CYAN + "\n--- YouTube Chatter Menu ---" + Style.RESET_ALL)
    print(Fore.YELLOW + "1. Chat about a new YouTube video")
    print("2. Chat about a previously transcribed video")
    print("3. Exit" + Style.RESET_ALL)

def handle_new_video():
    youtube_url = input(Fore.YELLOW + "🔗 Enter the YouTube video URL (or type 'exit' to go back): " + Style.RESET_ALL)
    if youtube_url.lower() in ['exit', 'quit', 'q']:
        return

    if not youtube_url.startswith("http"):
        print(Fore.RED + "❌ Invalid URL. Please enter a valid YouTube URL." + Style.RESET_ALL)
        return

    save_option = input(Fore.YELLOW + "Do you want to save the transcription for future use? (y/n or type 'exit' to go back): " + Style.RESET_ALL).lower()
    if save_option in ['exit', 'quit', 'q']:
        return

    save_transcription = save_option == 'y'

    try:
        transcription, video_title = get_or_create_transcription(youtube_url, save_transcription)
        
        if transcription:
            print(Fore.GREEN + f"\n✅ Transcription ready for video: {video_title}" + Style.RESET_ALL)
            print(Fore.YELLOW + "ℹ️  You can end the conversation at any time by pressing Ctrl+C or typing 'exit'." + Style.RESET_ALL)
            model_choice = input(Fore.YELLOW + "Select the model to use (1: Groq, 2: OpenAI, 3: Anthropic): " + Style.RESET_ALL)
            chat_with_ai(transcription, model_choice)
        else:
            print(Fore.RED + "❌ Unable to start chat due to transcription failure." + Style.RESET_ALL)
    except Exception as e:
        print(Fore.RED + f"❌ An error occurred while processing the video: {e}" + Style.RESET_ALL)
def handle_saved_video():
    saved_transcriptions = list_saved_transcriptions()
    if not saved_transcriptions:
        print(Fore.RED + "❌ No saved transcriptions found." + Style.RESET_ALL)
        return
    
    print(Fore.CYAN + "\nSaved Transcriptions:" + Style.RESET_ALL)
    for i, (video_id, title) in enumerate(saved_transcriptions, 1):
        print(f"{i}. {title}")
    
    while True:
        selection = input(Fore.YELLOW + f"Select a video (1-{len(saved_transcriptions)}) or type 'exit' to go back: " + Style.RESET_ALL)
        if selection.lower() in ['exit', 'quit', 'q']:
            return

        try:
            index = int(selection) - 1
            if 0 <= index < len(saved_transcriptions):
                video_id, title = saved_transcriptions[index]
                with open(os.path.join(TRANSCRIPTION_DIR, f"{video_id}.json"), 'r') as f:
                    data = json.load(f)
                    transcription = data.get('transcription')
                    if not transcription:
                        raise ValueError("Transcription not found in the file")
                
                print(Fore.GREEN + f"\n✅ Loaded transcription for: {title}" + Style.RESET_ALL)
                print(Fore.YELLOW + "ℹ️  You can end the conversation at any time by pressing Ctrl+C or typing 'exit'." + Style.RESET_ALL)
                model_choice = input(Fore.YELLOW + "Select the model to use (1: Groq, 2: OpenAI, 3: Anthropic): " + Style.RESET_ALL)
                try:
                    chat_with_ai(transcription, model_choice)
                except ValueError as ve:
                    print(Fore.RED + f"❌ Error loading transcription: {str(ve)}. Please try again." + Style.RESET_ALL)
                return
            else:
                raise ValueError("Invalid selection")
        except (ValueError, IndexError, FileNotFoundError, json.JSONDecodeError) as e:
            print(Fore.RED + f"❌ Error loading transcription: {str(e)}. Please try again." + Style.RESET_ALL)

def main():
    print_welcome()
    while True:
        print_menu()
        choice = input(Fore.BLUE + "Enter your choice (1-3): " + Style.RESET_ALL)
        
        if choice == '1':
            handle_new_video()
        elif choice == '2':
            handle_saved_video()
        elif choice == '3':
            print(Fore.YELLOW + "Thank you for using YouTube Chatter. Goodbye!" + Style.RESET_ALL)
            break
        else:
            print(Fore.RED + "❌ That is not a valid option. Please try again." + Style.RESET_ALL)

if __name__ == "__main__":
    main()
