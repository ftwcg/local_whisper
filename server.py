from flask import Flask, request, jsonify
import whisper
import tempfile
import os
import warnings
import sqlite3
import datetime
import traceback
import time
import json

app = Flask(__name__)

# Initialize metrics database
def init_metrics_db():
    """Initialize the metrics database if it doesn't exist."""
    db_path = 'server_metrics.db'
    if not os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute('''CREATE TABLE usage_metrics
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     timestamp TEXT,
                     hostname TEXT,
                     username TEXT,
                     system_info TEXT,
                     duration REAL,
                     char_count INTEGER,
                     success BOOLEAN,
                     mode TEXT,
                     model TEXT,
                     processing_time REAL,
                     error_type TEXT,
                     error_message TEXT,
                     error_traceback TEXT)''')
        conn.commit()
        conn.close()

init_metrics_db()

def track_metrics(system_info, duration, char_count, success, mode, model=None, processing_time=None, error_info=None):
    """Track usage metrics in the database.
    
    This function can be easily replaced with a different implementation
    (e.g., sending to a different database, API, etc.) without changing
    the rest of the code.
    """
    try:
        conn = sqlite3.connect('server_metrics.db')
        c = conn.cursor()
        
        c.execute('''INSERT INTO usage_metrics 
                    (timestamp, hostname, username, system_info, duration, char_count, 
                     success, mode, model, processing_time, error_type, error_message, error_traceback)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                 (datetime.datetime.now().isoformat(),
                  system_info.get('hostname', 'unknown') if system_info else 'unknown',
                  system_info.get('username', 'unknown') if system_info else 'unknown',
                  str(system_info) if system_info else '{}',
                  duration,
                  char_count,
                  success,
                  mode,
                  model,
                  processing_time,
                  error_info.get('type') if error_info else None,
                  error_info.get('message') if error_info else None,
                  error_info.get('traceback') if error_info else None))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error tracking usage metrics: {e}")

# Load the Whisper model (use base.en by default)
try:
    model_name = "base.en" 
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        warnings.filterwarnings("ignore", message=".*FP16 is not supported on CPU.*")
        model = whisper.load_model(model_name)
    print(f"Whisper model '{model_name}' loaded successfully.")
except Exception as e:
    print(f"Error loading Whisper model: {e}")
    # Exit if model loading fails? Or handle requests differently?
    # For now, let's let it run, but transcription will fail.
    model = None

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if model is None:
        return jsonify({"error": "Whisper model not loaded"}), 500

    if 'audio' not in request.files:
        return jsonify({"error": "No audio file part in the request"}), 400

    audio_file = request.files['audio']
    system_info = request.form.get('system_info', '{}')
    start_time = float(request.form.get('start_time', time.time()))
    mode = request.form.get('mode', 'unknown')

    if audio_file.filename == '':
        return jsonify({"error": "No selected audio file"}), 400

    try:
        # Save the uploaded audio file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            audio_file.save(temp_audio.name)
            temp_audio_path = temp_audio.name
        
        print(f"Temporary audio file saved at: {temp_audio_path}")

        # Transcribe the audio file with FP16 warning suppression
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            warnings.filterwarnings("ignore", message=".*FP16 is not supported on CPU.*")
            result = model.transcribe(temp_audio_path, fp16=False)
        transcription = result["text"]
        
        print(f"Transcription successful: {transcription}")

        # Clean up the temporary file
        os.remove(temp_audio_path)
        print(f"Temporary file removed: {temp_audio_path}")

        # Track successful transcription
        duration = time.time() - start_time
        track_metrics(
            system_info=json.loads(system_info) if system_info else None,
            duration=duration,
            char_count=len(transcription),
            success=True,
            mode=mode,
            model=model_name,
            processing_time=duration
        )

        return jsonify({"transcription": transcription})

    except Exception as e:
        print(f"Error during transcription: {e}")
        # Clean up the temporary file in case of error
        if 'temp_audio_path' in locals() and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
            print(f"Temporary file removed after error: {temp_audio_path}")
        
        # Track failed transcription
        duration = time.time() - start_time
        track_metrics(
            system_info=json.loads(system_info) if system_info else None,
            duration=duration,
            char_count=0,
            success=False,
            mode=mode,
            model=model_name,
            processing_time=duration,
            error_info={
                'type': type(e).__name__,
                'message': str(e),
                'traceback': traceback.format_exc()
            }
        )
        
        return jsonify({"error": f"Transcription failed: {str(e)}"}), 500

if __name__ == '__main__':
    # Run the app on localhost, port 5001
    app.run(host='localhost', port=5001, debug=True) # Debug=True is helpful for development 