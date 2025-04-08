import toml
import sounddevice as sd
import numpy as np
from pynput import keyboard
import pyperclip
import whisper
import threading
import queue
import time
import argparse
import logging
import warnings
import requests
import io
from scipy.io.wavfile import write as write_wav
import tkinter as tk
from tkinter import ttk, scrolledtext
import sys
import os
import subprocess
import sqlite3
import uuid
import datetime
import platform
import getpass
import traceback
import json

# --- Model Choices and Descriptions ---
MODEL_CHOICES = [
    'tiny.en', 'tiny',
    'base.en', 'base',
    'small.en', 'small',
    'medium.en', 'medium',
    'large-v1', 'large-v2', 'large-v3', 'large'
]

MODEL_HELP = f"""
Choose the Whisper model to use when running in local mode. Models range from 'tiny' (fastest, lowest accuracy) 
to 'large' (slowest, highest accuracy). The '.en' suffix indicates English-only models.
Choices: {MODEL_CHOICES}
(Default: value from config.toml)
"""

# Global debug flag
DEBUG_ENABLED = False

def debug_print(*args, **kwargs):
    """Print debug messages only if debug flag is enabled"""
    if DEBUG_ENABLED:
        print(*args, **kwargs)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Whisper Transcription Tool')
parser.add_argument('--local', action='store_true', help='Run Whisper model locally instead of sending to server')
parser.add_argument('--gui', action='store_true', help='Show GUI interface instead of using hotkeys')
parser.add_argument('--server-url', type=str, default='http://localhost:5001/transcribe', help='URL of the transcription server')
parser.add_argument('--model', type=str, default='base.en', help='Whisper model to use in local mode')
parser.add_argument('--no-sound', action='store_true', help='Disable all sound feedback')
parser.add_argument('--debug', action='store_true', help='Enable debug logging')
args = parser.parse_args()

# Set global debug flag based on command line argument
DEBUG_ENABLED = args.debug

# Configure logging
log_level = logging.DEBUG if args.debug else logging.CRITICAL
log_format = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(level=log_level, format=log_format)

# Load configuration
logging.info("Loading configuration from config.toml...")
with open("config.toml", "r") as f:
    config = toml.load(f)

# --- Model Loading (Conditional) ---
model = None
model_name = None
if args.local:
    # Determine the final model name (CLI overrides config)
    model_name_config = config["settings"]["model"]
    logging.info(f"Model specified in config.toml: {model_name_config}")
    model_name = model_name_config
    if args.model:
        model_name = args.model
        logging.info(f"Overriding model with command-line argument: {model_name}")
    else:
        logging.info(f"Using model from config.toml: {model_name}")

    # Load Whisper model, suppressing the FP16 warning
    logging.info(f"Loading Whisper model '{model_name}'... (This may take a moment)")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        warnings.filterwarnings("ignore", message=".*FP16 is not supported on CPU.*")
        model = whisper.load_model(model_name)
    logging.info("Whisper model loaded.")
elif args.model:
    # Warn if -m is specified without --local
    print("Warning: --model argument ignored because --local flag is not set. Using server mode.")
    logging.warning("--model argument provided but --local is not set. Defaulting to server mode.")

# Print mode
if args.local:
    print(f"\n--- Running in LOCAL mode (Model: {model_name}) ---")
else:
    print(f"\n--- Running in CLIENT mode (Server: {args.server_url}) ---")

# Print configured hotkeys using print()
print("\nConfigured Hotkeys:")
for name, hotkey in config['hotkeys'].items():
    print(f"  - {name.replace('_', ' ').title()}: {hotkey}")
print("\nPress Ctrl+C to exit.")

# Global variables
recording_lock = threading.Lock()
recording_thread = None
stop_event = threading.Event()
command_queue = queue.Queue(maxsize=10)
current_pressed_keys = set()
key_lock = threading.Lock()
audio_data_queue = queue.Queue()

# --- State flags managed by control thread ---
_is_recording_internal = False
_hold_key_active_internal = False

# Sound feedback functions
def play_sound(sound_name):
    """Play a sound file if sound is enabled and the file exists."""
    if args.no_sound:
        return
    
    sound_path = os.path.join('sounds', sound_name)
    if os.path.exists(sound_path):
        try:
            # Use aplay for Linux, afplay for Mac, or start for Windows
            if sys.platform == 'linux':
                subprocess.Popen(['aplay', sound_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            elif sys.platform == 'darwin':  # macOS
                subprocess.Popen(['afplay', sound_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:  # Windows
                subprocess.Popen(['start', sound_path], shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            logging.warning(f"Could not play sound {sound_name}: {e}")

def generate_tone(frequency, duration, volume=0.5):
    """Generate a simple sine wave tone."""
    try:
        sample_rate = 44100
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        tone = np.sin(frequency * t * 2 * np.pi)
        audio = tone * (2**15 - 1) * volume
        return audio.astype(np.int16)
    except Exception as e:
        logging.warning(f"Error generating tone: {e}")
        return None

def play_tone(frequency, duration=0.1, volume=0.5):
    """Play a tone with the given frequency and duration."""
    try:
        audio = generate_tone(frequency, duration, volume)
        if audio is not None:
            play_obj = sa.play_buffer(audio, 1, 2, 44100)
            play_obj.wait_done()
    except Exception as e:
        logging.warning(f"Could not play audio feedback: {e}")
        # Continue execution even if audio fails

# Constants for different tones
START_TONE = 880  # A5
STOP_TONE = 440   # A4
DONE_TONE = 660   # E5

def parse_hotkey(hotkey_string):
    """Parses a hotkey string like 'ctrl+shift+r' into pynput components."""
    parts = hotkey_string.lower().split('+')
    keys = set()
    for part in parts:
        part = part.strip()
        # Handle ctrl, shift, alt for both left and right keys
        if part == 'ctrl':
            keys.add(keyboard.Key.ctrl_l)
            keys.add(keyboard.Key.ctrl_r)
        elif part == 'shift':
            keys.add(keyboard.Key.shift_l)
            keys.add(keyboard.Key.shift_r)
        elif part == 'alt':
            keys.add(keyboard.Key.alt_l) # Use alt_l for generic alt
            keys.add(keyboard.Key.alt_r) # Also add alt_r if needed
        elif len(part) == 1:  # Regular character key
            keys.add(keyboard.KeyCode.from_char(part))
        else:  # Special key (e.g., 'space', 'f1')
            try:
                # Map common alternative names if necessary
                key_map = {
                    'esc': keyboard.Key.esc,
                    # Add other mappings if needed
                }
                key_to_add = key_map.get(part, keyboard.Key[part])
                keys.add(key_to_add)
            except KeyError:
                logging.warning(f"Unrecognized key name '{part}' in hotkey '{hotkey_string}'")
                print(f"WARNING: Unrecognized key '{part}' in hotkey '{hotkey_string}'")
    return keys

# Pre-parse hotkeys from config
hotkey_sets = {}
for name, value in config['hotkeys'].items():
    hotkey_sets[name] = parse_hotkey(value)
    logging.debug(f"Parsed hotkey '{name}': {value} -> {hotkey_sets[name]}")

def get_modifier_char_sets(key_set):
    """Separates modifier keys from character keys in a key set.
    Character keys are stored as lowercase strings or Key objects for specials.
    """
    modifiers = set()
    chars = set()
    for key in key_set:
        if isinstance(key, keyboard.Key):
            # Separate specific modifier keys
            if key in {keyboard.Key.ctrl_l, keyboard.Key.ctrl_r,
                      keyboard.Key.shift_l, keyboard.Key.shift_r,
                      keyboard.Key.alt_l, keyboard.Key.alt_r}:
                modifiers.add(key)
            else: # Special keys like space, enter, f1, etc.
                chars.add(key) # Store the Key object
        elif isinstance(key, keyboard.KeyCode):
            # Store the lowercase character string representation or the KeyCode itself
            if hasattr(key, 'char') and key.char:
                chars.add(key.char.lower())
            else:
                chars.add(key)
    return modifiers, chars

# Pre-process hotkeys for faster checking
hotkey_parsed = {}
for name, key_set in hotkey_sets.items():
    hotkey_parsed[name] = get_modifier_char_sets(key_set)
    logging.debug(f"Processed hotkey '{name}': modifiers={hotkey_parsed[name][0]}, chars={hotkey_parsed[name][1]}")

def check_hotkey(target_name):
    """Checks if the currently pressed keys match the target hotkey.
    Uses the global current_pressed_keys set.
    """
    target_mods_set, target_chars_set = hotkey_parsed[target_name]

    # Determine currently active modifier types and pressed character/other keys
    pressed_chars = set()
    pressed_ctrl = False
    pressed_shift = False
    pressed_alt = False
    local_current_pressed_keys = current_pressed_keys.copy()

    for key in local_current_pressed_keys:
        if isinstance(key, keyboard.KeyCode):
            if hasattr(key, 'char') and key.char:
                pressed_chars.add(key.char.lower())
            else:
                pressed_chars.add(key)
        elif isinstance(key, keyboard.Key):
            if key in {keyboard.Key.ctrl_l, keyboard.Key.ctrl_r}:
                pressed_ctrl = True
            elif key in {keyboard.Key.shift_l, keyboard.Key.shift_r}:
                pressed_shift = True
            elif key in {keyboard.Key.alt_l, keyboard.Key.alt_r}:
                pressed_alt = True
            else:
                pressed_chars.add(key)

    required_ctrl = any(k in {keyboard.Key.ctrl_l, keyboard.Key.ctrl_r} for k in target_mods_set)
    required_shift = any(k in {keyboard.Key.shift_l, keyboard.Key.shift_r} for k in target_mods_set)
    required_alt = any(k in {keyboard.Key.alt_l, keyboard.Key.alt_r} for k in target_mods_set)

    if pressed_chars != target_chars_set:
        return False
    if pressed_ctrl != required_ctrl:
        return False
    if pressed_shift != required_shift:
        return False
    if pressed_alt != required_alt:
        return False

    return True

# Function to start recording (now only contains the audio loop)
def start_recording():
    global audio_data_queue, stop_event, _is_recording_internal
    play_sound('start.wav')
    audio_data_queue = queue.Queue()
    samplerate = 16000
    chunk_duration_ms = 100
    chunk_size_frames = int(samplerate * chunk_duration_ms / 1000)
    stream = None

    try:
        stream = sd.InputStream(
            channels=1,
            samplerate=samplerate,
            blocksize=chunk_size_frames,
            dtype=np.float32
        )
        stream.start()

        while not stop_event.is_set():
            try:
                indata, overflowed = stream.read(chunk_size_frames)
                if _is_recording_internal:
                    audio_data_queue.put(indata.copy(), block=False)
            except (queue.Full, sd.PortAudioError, Exception):
                break

    except Exception:
        pass
    finally:
        if stream is not None:
            try:
                if stream.active:
                    stream.stop()
                stream.close()
            except:
                pass

def get_system_info():
    """Get system and user information for metrics."""
    try:
        return {
            'hostname': platform.node(),
            'username': getpass.getuser(),
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor()
        }
    except Exception as e:
        logging.error(f"Error getting system info: {e}")
        return {}

def send_audio_to_server(audio_data_fp32, samplerate, server_url):
    """Sends audio data to the transcription server."""
    logging.info(f"Sending audio data to server: {server_url}")
    
    start_time = time.time()
    try:
        # Create WAV in memory
        wav_bytes = io.BytesIO()
        write_wav(wav_bytes, samplerate, audio_data_fp32)
        wav_bytes.seek(0) # Rewind buffer to the beginning
        
        # Get system info for metrics
        system_info = get_system_info()
        
        files = {'audio': ('audio.wav', wav_bytes, 'audio/wav')}
        data = {
            'system_info': json.dumps(system_info),  # Convert to JSON string
            'start_time': start_time,
            'mode': 'server'  # Indicate this is a server transcription request
        }
        
        # Set a long timeout (e.g., 5 minutes = 300 seconds)
        timeout_seconds = 300 
        
        response = requests.post(server_url, files=files, data=data, timeout=timeout_seconds)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        result_json = response.json()
        duration = time.time() - start_time
        
        if "transcription" in result_json:
            transcription = result_json["transcription"].strip()
            logging.info(f"Server transcription successful in {duration:.1f}s")
            return transcription
        elif "error" in result_json:
            error_msg = result_json["error"]
            logging.error(f"Server returned an error: {error_msg}")
            print(f"ERROR: Server transcription failed: {error_msg}")
            return None
        else:
            logging.error("Server response did not contain 'transcription' or 'error' key.")
            print("ERROR: Received unexpected response format from server.")
            return None

    except requests.exceptions.ConnectionError as e:
        logging.error(f"Connection Error: Could not connect to server at {server_url}. {e}")
        print(f"ERROR: Cannot connect to server at {server_url}. Is it running?")
        return None
    except requests.exceptions.Timeout:
        logging.error(f"Timeout Error: Request to server timed out after {timeout_seconds}s.")
        print(f"ERROR: Request to server timed out ({timeout_seconds}s).")
        return None
    except requests.exceptions.RequestException as e:
        logging.error(f"Request Error: An error occurred sending data to the server: {e}")
        print(f"ERROR: Failed to communicate with server: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error sending/receiving data: {e}")
        print(f"ERROR: An unexpected error occurred during server communication: {e}")
        return None
    finally:
        if 'wav_bytes' in locals():
            wav_bytes.close() # Clean up BytesIO buffer

def _process_stopped_audio():
    global audio_data_queue, model, args
    
    debug_print("DEBUG - _process_stopped_audio started.")
    logging.info("Processing recorded audio...")
    play_sound('stop.wav')

    audio_chunks = []
    while not audio_data_queue.empty():
        try:
            audio_chunks.append(audio_data_queue.get_nowait())
        except queue.Empty:
            break
    
    if not audio_chunks:
        logging.warning("No audio data captured.")
        print("=== TRANSCRIPTION SKIPPED - No audio data ===")
        play_sound('error.wav')
        return

    try:
        audio_np = np.concatenate(audio_chunks, axis=0)
        audio_fp32 = audio_np.flatten().astype(np.float32)
    except ValueError as e:
        logging.error(f"Error concatenating audio chunks: {e}")
        print(f"=== TRANSCRIPTION FAILED - Audio data processing error === ({e})")
        play_sound('error.wav')
        return
    except Exception as e:
        logging.error(f"Unexpected error processing audio chunks: {e}")
        print(f"=== TRANSCRIPTION FAILED - Unexpected audio processing error === ({e})")
        play_sound('error.wav')
        return

    transcription = None
    duration = 0
    start_time = time.time()
    
    if args.local:
        # --- LOCAL TRANSCRIPTION ---
        logging.info("Performing local transcription...")
        print("--- Transcribing locally... ---")
        if model is None:
            print("ERROR: Local model was not loaded successfully. Cannot transcribe locally.")
            logging.error("Attempted local transcription, but model is None.")
            play_sound('error.wav')
            return

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                warnings.filterwarnings("ignore", message=".*FP16 is not supported on CPU.*")
                result = model.transcribe(audio_fp32, fp16=False)
            transcription = result["text"].strip()
            duration = time.time() - start_time
            logging.info(f"Local transcription successful in {duration:.1f}s")
            print(f"--- Local transcription finished ({duration:.1f}s) ---")
        except Exception as e:
            logging.error(f"Error during local Whisper transcription: {e}")
            print(f"=== LOCAL TRANSCRIPTION FAILED - Whisper error: {e} ===")
            play_sound('error.wav')
            return
            
    else:
        # --- SERVER TRANSCRIPTION ---
        print("--- Sending audio to server... ---")
        transcription = send_audio_to_server(audio_fp32, 16000, args.server_url)
        duration = time.time() - start_time

    # --- Post-processing ---
    if transcription:
        if not transcription.strip():  # Check if transcription is empty after stripping whitespace
            print("=== TRANSCRIPTION SKIPPED - Empty result ===")
            logging.warning("Transcription result was empty after stripping whitespace.")
            play_sound('error.wav')
            return
            
        try:
            pyperclip.copy(transcription)
            print(f"Transcription: {transcription}")
            print(f"Copied to clipboard. (Processed in {duration:.1f}s)")
            logging.info(f"Transcription successful and copied to clipboard.")
            play_sound('done.wav')
        except Exception as e:
            logging.error(f"Could not copy transcription to clipboard: {e}")
            print(f"Transcription: {transcription}")
            print(f"(Could not copy to clipboard: {e})")
            play_sound('error.wav')
    elif transcription is None:
        print("=== TRANSCRIPTION FAILED (See errors above) ===")
        logging.warning("Transcription result was None.")
        play_sound('error.wav')
    else:
        print("=== TRANSCRIPTION SKIPPED - Empty result ===")
        logging.info("Transcription result was empty.")
        play_sound('error.wav')

    logging.info("Processing thread finished.")
    debug_print("DEBUG - _process_stopped_audio finished.")

# --- Keyboard Listener Callbacks (Lightweight) ---
def on_press(key):
    global current_pressed_keys, command_queue
    
    with key_lock:
        current_pressed_keys.add(key)
        keys = current_pressed_keys.copy()
    
    # Check hotkeys and queue commands
    try:
        if check_hotkey('start_recording_toggle'):
            command_queue.put('START_TOGGLE', block=False)
        elif check_hotkey('stop_recording_toggle'):
            command_queue.put('STOP_TOGGLE', block=False)
        elif check_hotkey('hold_to_record'):
            command_queue.put('START_HOLD', block=False)
    except queue.Full:
        pass  # Ignore if queue is full

def on_release(key):
    global current_pressed_keys, command_queue
    
    with key_lock:
        keys_before = current_pressed_keys.copy()
        try:
            current_pressed_keys.remove(key)
        except KeyError:
            pass
        keys_after = current_pressed_keys.copy()
    
    # Check state BEFORE release
    original_current_pressed_keys = current_pressed_keys
    current_pressed_keys = keys_before
    hold_active_before = check_hotkey('hold_to_record')
    current_pressed_keys = original_current_pressed_keys

    # Check state AFTER release
    current_pressed_keys = keys_after
    hold_active_after = check_hotkey('hold_to_record')
    current_pressed_keys = original_current_pressed_keys
    
    if hold_active_before and not hold_active_after:
        try:
            command_queue.put('STOP_HOLD', block=False)
        except queue.Full:
            pass

# --- Control Thread --- 
def control_thread_func():
    global command_queue, recording_thread, stop_event
    global _is_recording_internal, _hold_key_active_internal
    
    while True:
        try:
            command = command_queue.get(timeout=0.1)  # Add timeout to prevent blocking
        except queue.Empty:
            continue

        with recording_lock:
            if command == 'START_TOGGLE':
                if not _is_recording_internal:
                    print("\n=== RECORDING STARTED - Toggle ===")
                    _is_recording_internal = True
                    _hold_key_active_internal = False
                    stop_event.clear()
                    recording_thread = threading.Thread(target=start_recording)
                    recording_thread.start()
            
            elif command == 'STOP_TOGGLE':
                if _is_recording_internal and not _hold_key_active_internal:
                    print("\n=== RECORDING STOPPED - Toggle ===")
                    stop_event.set()
                    _is_recording_internal = False
                    processing_thread = threading.Thread(target=_process_stopped_audio)
                    processing_thread.start()

            elif command == 'START_HOLD':
                if not _is_recording_internal:
                    print("\n=== RECORDING STARTED - Hold ===")
                    _is_recording_internal = True
                    _hold_key_active_internal = True
                    stop_event.clear()
                    recording_thread = threading.Thread(target=start_recording)
                    recording_thread.start()
            
            elif command == 'STOP_HOLD':
                if _is_recording_internal and _hold_key_active_internal:
                    print("\n=== RECORDING STOPPED - Hold ===")
                    stop_event.set()
                    _is_recording_internal = False
                    _hold_key_active_internal = False
                    processing_thread = threading.Thread(target=_process_stopped_audio)
                    processing_thread.start()

            elif command == 'EXIT':
                if _is_recording_internal:
                    stop_event.set()
                    _is_recording_internal = False
                    _hold_key_active_internal = False
                    if recording_thread and recording_thread.is_alive():
                        recording_thread.join(timeout=1.0)
                break
            
            else:
                logging.warning(f"Control thread received unknown command: {command}")

    debug_print("DEBUG - Control thread finished.")

class WhisperGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Whisper Transcription Tool")
        self.root.geometry("600x400")
        self.root.minsize(500, 350)
        
        # Set theme and style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure styles
        style.configure('TFrame', background='#f0f0f0')
        style.configure('TLabel', background='#f0f0f0', font=('Segoe UI', 10))
        style.configure('TButton', font=('Segoe UI', 12, 'bold'), padding=15)
        style.configure('Status.TLabel', font=('Segoe UI', 10, 'bold'))
        style.configure('Mode.TLabel', font=('Segoe UI', 9), foreground='#666666')
        style.configure('Clipboard.TLabel', font=('Segoe UI', 12, 'bold'))
        style.configure('LogTitle.TLabel', font=('Segoe UI', 11, 'bold'))
        
        # Create custom button styles with different colors
        style.configure('Start.TButton', background='#4CAF50', foreground='white')
        style.configure('Stop.TButton', background='#f44336', foreground='white')
        style.configure('Clear.TButton', background='#2196F3', foreground='white')
        style.configure('Exit.TButton', background='#FF9800', foreground='white')
        
        # Create main frame with padding
        main_frame = ttk.Frame(root, padding="15")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights for resizing
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.columnconfigure(2, weight=1)
        main_frame.columnconfigure(3, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Status label
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(main_frame, textvariable=self.status_var, style='Status.TLabel')
        self.status_label.grid(row=0, column=0, columnspan=4, pady=(0, 10))
        
        # Clipboard status label
        self.clipboard_status_var = tk.StringVar(value="")
        self.clipboard_status = ttk.Label(
            main_frame, 
            textvariable=self.clipboard_status_var, 
            style='Clipboard.TLabel',
            foreground='green'
        )
        self.clipboard_status.grid(row=1, column=0, columnspan=4, pady=(0, 10))
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, columnspan=4, pady=(0, 10))
        
        # Left column frame for recording buttons
        left_frame = ttk.Frame(button_frame)
        left_frame.grid(row=0, column=0, padx=5)
        
        # Right column frame for utility buttons
        right_frame = ttk.Frame(button_frame)
        right_frame.grid(row=0, column=1, padx=5)
        
        # Recording buttons in left column
        self.record_button = ttk.Button(
            left_frame, 
            text="Start Recording", 
            command=self.start_recording,
            style='Start.TButton',
            width=20
        )
        self.record_button.grid(row=0, column=0, pady=5)
        
        self.stop_button = ttk.Button(
            left_frame, 
            text="Stop Recording", 
            command=self.stop_recording, 
            state=tk.DISABLED,
            style='Stop.TButton',
            width=20
        )
        self.stop_button.grid(row=1, column=0, pady=5)
        
        # Utility buttons in right column
        self.clear_button = ttk.Button(
            right_frame, 
            text="Clear Log", 
            command=self.clear_text,
            style='Clear.TButton',
            width=20
        )
        self.clear_button.grid(row=0, column=0, pady=5)
        
        self.exit_button = ttk.Button(
            right_frame,
            text="Exit",
            command=self.on_closing,
            style='Exit.TButton',
            width=20
        )
        self.exit_button.grid(row=1, column=0, pady=5)
        
        # Transcription log title
        log_title = ttk.Label(
            main_frame,
            text="Transcription Log",
            style='LogTitle.TLabel'
        )
        log_title.grid(row=3, column=0, columnspan=4, pady=(0, 5), sticky=tk.W)
        
        # Transcription text area
        self.text_area = scrolledtext.ScrolledText(
            main_frame, 
            wrap=tk.WORD, 
            width=50,
            height=8,
            font=('Segoe UI', 10),
            padx=10,
            pady=10
        )
        self.text_area.grid(row=4, column=0, columnspan=4, pady=(0, 10), sticky=(tk.W, tk.E, tk.N))
        
        # Mode label
        mode_text = f"Local ({model_name})" if args.local else "Server"
        mode_label = ttk.Label(
            main_frame, 
            text=f"Mode: {mode_text}",
            style='Mode.TLabel'
        )
        mode_label.grid(row=5, column=0, columnspan=4)
        
        # Bind window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Start the control thread
        self.control_thread = threading.Thread(target=self.control_thread_func, daemon=True)
        self.control_thread.start()
        
        # Set focus to the window
        self.root.focus_force()
    
    def start_recording(self):
        self.record_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_var.set("Recording...")
        self.clipboard_status_var.set("")
        play_sound('start.wav')
        command_queue.put('START_TOGGLE', block=False)
    
    def stop_recording(self):
        self.record_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_var.set("Processing...")
        play_sound('stop.wav')
        command_queue.put('STOP_TOGGLE', block=False)
    
    def clear_text(self):
        self.text_area.delete(1.0, tk.END)
    
    def append_text(self, text):
        self.text_area.insert(tk.END, text + "\n")
        self.text_area.see(tk.END)
    
    def control_thread_func(self):
        global _is_recording_internal, _hold_key_active_internal
        
        while True:
            try:
                command = command_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            
            if command == 'START_TOGGLE':
                if not _is_recording_internal:
                    _is_recording_internal = True
                    _hold_key_active_internal = False
                    stop_event.clear()
                    recording_thread = threading.Thread(target=start_recording)
                    recording_thread.start()
            
            elif command == 'STOP_TOGGLE':
                if _is_recording_internal and not _hold_key_active_internal:
                    stop_event.set()
                    _is_recording_internal = False
                    processing_thread = threading.Thread(target=self.process_audio)
                    processing_thread.start()
            
            elif command == 'EXIT':
                if _is_recording_internal:
                    stop_event.set()
                    _is_recording_internal = False
                break
    
    def process_audio(self):
        global audio_data_queue, model, args
        
        audio_chunks = []
        while not audio_data_queue.empty():
            try:
                audio_chunks.append(audio_data_queue.get_nowait())
            except queue.Empty:
                break
        
        if not audio_chunks:
            message = "=== TRANSCRIPTION SKIPPED - No audio data ==="
            print(message)
            self.append_text(message)
            play_sound('error.wav')
            return
        
        try:
            audio_np = np.concatenate(audio_chunks, axis=0)
            audio_fp32 = audio_np.flatten().astype(np.float32)
        except Exception as e:
            message = f"=== TRANSCRIPTION FAILED - Audio processing error: {e} ==="
            print(message)
            self.append_text(message)
            play_sound('error.wav')
            return
        
        start_time = time.time()
        if args.local:
            if model is None:
                message = "ERROR: Local model was not loaded successfully."
                print(message)
                self.append_text(message)
                play_sound('error.wav')
                return
            
            try:
                self.status_var.set("Transcribing locally...")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    warnings.filterwarnings("ignore", message=".*FP16 is not supported on CPU.*")
                    result = model.transcribe(audio_fp32, fp16=False)
                transcription = result["text"].strip()
                duration = time.time() - start_time
                
                if not transcription:  # Check if transcription is empty
                    message = "=== TRANSCRIPTION SKIPPED - Empty result ==="
                    print(message)
                    self.append_text(message)
                    play_sound('error.wav')
                    return
                    
                message = f"Transcription ({duration:.1f}s): {transcription}"
                print(message)
                self.append_text(message)
                try:
                    pyperclip.copy(transcription)
                    preview = transcription[:30] + "..." if len(transcription) > 30 else transcription
                    self.clipboard_status_var.set(f"✓ Copied to clipboard ({duration:.1f}s): {preview}")
                    play_sound('done.wav')
                except Exception as e:
                    self.clipboard_status_var.set("⚠ Failed to copy to clipboard")
                    play_sound('error.wav')
            except Exception as e:
                message = f"=== LOCAL TRANSCRIPTION FAILED - Whisper error: {e} ==="
                print(message)
                self.append_text(message)
                play_sound('error.wav')
        else:
            self.status_var.set("Sending to server...")
            transcription = send_audio_to_server(audio_fp32, 16000, args.server_url)
            duration = time.time() - start_time
            if transcription:
                if not transcription.strip():  # Check if transcription is empty
                    message = "=== TRANSCRIPTION SKIPPED - Empty result ==="
                    print(message)
                    self.append_text(message)
                    play_sound('error.wav')
                    return
                    
                message = f"Transcription ({duration:.1f}s): {transcription}"
                print(message)
                self.append_text(message)
                try:
                    pyperclip.copy(transcription)
                    preview = transcription[:30] + "..." if len(transcription) > 30 else transcription
                    self.clipboard_status_var.set(f"✓ Copied to clipboard ({duration:.1f}s): {preview}")
                    play_sound('done.wav')
                except Exception as e:
                    self.clipboard_status_var.set("⚠ Failed to copy to clipboard")
                    play_sound('error.wav')
            else:
                message = "=== TRANSCRIPTION FAILED - Server returned no transcription ==="
                print(message)
                self.append_text(message)
                play_sound('error.wav')
        
        self.status_var.set("Ready")
    
    def on_closing(self):
        command_queue.put('EXIT', block=False)
        self.root.destroy()
        sys.exit(0)

# --- Main Execution ---
if args.gui:
    root = tk.Tk()
    app = WhisperGUI(root)
    root.mainloop()
else:
    # Start the keyboard listener in a separate thread
    print("Starting keyboard listener...")
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.daemon = True
    listener.start()
    print("Listener started.")

    # Start the control thread
    control_thread = threading.Thread(target=control_thread_func)
    control_thread.daemon = True
    control_thread.start()

    # Keep the main thread alive, handle Ctrl+C
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nCtrl+C detected. Cleaning up...")
        command_queue.put('EXIT')
        time.sleep(0.5)
        if listener.is_alive():
            listener.stop()
        if control_thread.is_alive():
            control_thread.join(timeout=1.0)

    # If recording is active, wait for it to stop
    if recording_thread and recording_thread.is_alive():
        recording_thread.join(timeout=1.0)
        if recording_thread.is_alive():
            print("Warning: Recording thread did not exit cleanly")
    
    print("Cleanup complete. Exiting...") 