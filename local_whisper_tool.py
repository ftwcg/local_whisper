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

# --- Model Choices and Descriptions ---
MODEL_CHOICES = [
    'tiny.en', 'tiny',
    'base.en', 'base',
    'small.en', 'small',
    'medium.en', 'medium',
    'large-v1', 'large-v2', 'large-v3', 'large' # large-v versions might be specific
]

MODEL_HELP = f"""
Choose the Whisper model to use. Models range from 'tiny' (fastest, lowest accuracy, low resource usage) 
to 'large' (slowest, highest accuracy, high resource usage). 
The '.en' suffix indicates English-only models, which are often faster and more accurate for English speech.
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
parser = argparse.ArgumentParser(description='Local Whisper Transcription Tool')
parser.add_argument('-d', '--debug', action='store_true', help='Enable debug logging')
parser.add_argument(
    '-m', '--model',
    type=str,
    choices=MODEL_CHOICES,
    help=MODEL_HELP,
    metavar='MODEL_NAME' # Shows MODEL_NAME instead of the full choices list in brief help
)
args = parser.parse_args()

# Set global debug flag based on command line argument
DEBUG_ENABLED = args.debug

# Configure logging
# Default level is CRITICAL (effectively off) unless --debug is passed
log_level = logging.DEBUG if args.debug else logging.CRITICAL
log_format = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(level=log_level, format=log_format)

# Load configuration
logging.info("Loading configuration from config.toml...")
with open("config.toml", "r") as f:
    config = toml.load(f)
# Get default model from config
model_name_config = config["settings"]["model"]
logging.info(f"Model specified in config.toml: {model_name_config}")

# Determine the final model name (CLI overrides config)
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

# Print configured hotkeys using print()
print("\nConfigured Hotkeys:")
for name, hotkey in config['hotkeys'].items():
    print(f"  - {name.replace('_', ' ').title()}: {hotkey}")
print("\nPress Ctrl+C to exit.")

# Global variables
recording_lock = threading.Lock()
recording_thread = None
stop_event = threading.Event()
command_queue = queue.Queue(maxsize=10)  # Limit queue size to prevent buildup
current_pressed_keys = set()
key_lock = threading.Lock()  # Add lock for key state synchronization

# --- State flags managed by control thread ---
_is_recording_internal = False
_hold_key_active_internal = False

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

# Function to process audio (no changes needed here conceptually)
def _process_stopped_audio():
    global audio_data_queue, model
    
    debug_print("DEBUG - _process_stopped_audio started.")
    logging.info("Processing audio...")

    audio_chunks = []
    while not audio_data_queue.empty():
        try:
            audio_chunks.append(audio_data_queue.get_nowait())
        except queue.Empty:
            break
    
    if not audio_chunks:
        logging.warning("No audio data captured.")
        print("=== TRANSCRIPTION SKIPPED - No audio ===")
        return

    try:
        audio_np = np.concatenate(audio_chunks, axis=0)
        audio_fp32 = audio_np.flatten().astype(np.float32)
    except ValueError as e:
        logging.error(f"Error concatenating audio chunks: {e}")
        print("=== TRANSCRIPTION FAILED - Data error ===")
        return

    try:
        start_time = time.time()
        result = model.transcribe(audio_fp32, fp16=False)
        transcription = result["text"].strip()
        duration = time.time() - start_time
    except Exception as e:
        logging.error(f"Error during transcription: {e}")
        print("=== TRANSCRIPTION FAILED - Whisper error ===")
        return

    if transcription:
        pyperclip.copy(transcription)
        print(f"Transcription: {transcription}")
        print(f"Copied to clipboard - Transcribed in {duration:.1f}s")
    else:
        print("=== TRANSCRIPTION SKIPPED - Empty result ===")

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


# --- Main Execution --- 

# Start the keyboard listener in a separate thread
print("Starting keyboard listener...")
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.daemon = True # Allow main thread to exit even if listener is active
listener.start()
print("Listener started.")

# Start the control thread
control_thread = threading.Thread(target=control_thread_func)
control_thread.daemon = True # Allow main thread to exit
control_thread.start()

# Keep the main thread alive, handle Ctrl+C
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nCtrl+C detected. Cleaning up...")
    
    # Signal all threads to stop
    command_queue.put('EXIT')
    
    # Give threads a moment to clean up
    time.sleep(0.5)
    
    # Stop the keyboard listener
    if listener.is_alive():
        listener.stop()
    
    # Wait for control thread to finish (with timeout)
    if control_thread.is_alive():
        control_thread.join(timeout=1.0)
        if control_thread.is_alive():
            print("Warning: Control thread did not exit cleanly")
    
    # If recording is active, wait for it to stop
    if recording_thread and recording_thread.is_alive():
        recording_thread.join(timeout=1.0)
        if recording_thread.is_alive():
            print("Warning: Recording thread did not exit cleanly")
    
    print("Cleanup complete. Exiting...") 