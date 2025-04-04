# Local Whisper Transcription Tool

A desktop application that uses OpenAI's Whisper model locally to transcribe audio captured via hotkeys and copies the text to the clipboard.

## Features

-  **Local Transcription:** Uses Whisper model locally, no data sent to the cloud.
-  **Flexible Hotkeys:** Configure hotkeys for hold-to-record and toggle start/stop recording modes.
-  **Clipboard Integration:** Automatically copies transcribed text to the clipboard.

## Installation

1.  Clone this repository.
2.  Create a virtual environment and activate it.
3.  Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

## Configuration

Edit the `config.toml` file to configure hotkeys and the desired Whisper model.

## Usage

Run the application:

```bash
python local_whisper_tool.py
```

Use the configured hotkeys to start and stop recording. The transcribed text will be copied to the clipboard. # local_whisper
