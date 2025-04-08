import numpy as np
from scipy.io.wavfile import write
import os

def generate_tone(frequency, duration, volume=0.5, sample_rate=44100):
    """Generate a simple sine wave tone and save it as a WAV file."""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    tone = np.sin(frequency * t * 2 * np.pi)
    audio = tone * (2**15 - 1) * volume
    return audio.astype(np.int16)

def generate_error_sound():
    """Generate an error sound with three quick tones."""
    sample_rate = 44100
    tones = [
        (330, 0.05, 0.5),  # E4, 50ms, 50% volume
        (330, 0.05, 0.5),  # E4, 50ms, 50% volume
        (330, 0.05, 0.5)   # E4, 50ms, 50% volume
    ]
    
    # Generate each tone
    audio_segments = []
    for freq, dur, vol in tones:
        audio = generate_tone(freq, dur, vol, sample_rate)
        audio_segments.append(audio)
        # Add a longer silence between tones (50ms instead of 20ms)
        silence = np.zeros(int(0.05 * sample_rate), dtype=np.int16)
        audio_segments.append(silence)
    
    # Combine all segments
    return np.concatenate(audio_segments)

def main():
    # Create sounds directory if it doesn't exist
    if not os.path.exists('sounds'):
        os.makedirs('sounds')
    
    # Generate and save the tones
    tones = {
        'start.wav': (660, 0.1, 0.3),   # E5, 100ms, 30% volume
        'stop.wav': (440, 0.1, 0.5),    # A4, 100ms, 50% volume
        'done.wav': (550, 0.1, 0.5)     # C#5, 100ms, 50% volume
    }
    
    for filename, (frequency, duration, volume) in tones.items():
        audio = generate_tone(frequency, duration, volume)
        write(os.path.join('sounds', filename), 44100, audio)
        print(f"Generated {filename}")
    
    # Generate error sound
    error_audio = generate_error_sound()
    write(os.path.join('sounds', 'error.wav'), 44100, error_audio)
    print("Generated error.wav")

if __name__ == '__main__':
    main() 