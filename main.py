import sounddevice as sd
from scipy.io.wavfile import write
import ollama
from faster_whisper import WhisperModel
import numpy as np

# --- CONFIGURATION ---
FS = 44100  # Sample rate (CD quality)
FILENAME = "lecture_recording.wav"

def record_audio():
    print("\n🎤 Recording... (Press Ctrl+C to stop)")
    audio_data = []
    
    try:
        #use a stream so we can record for an indefinite amount of time
        with sd.InputStream(samplerate=FS, channels=1, dtype='int16') as stream:
            while True:
                chunk, overflowed = stream.read(1024)
                audio_data.append(chunk)
    except KeyboardInterrupt:
        print("\n Recording stopped.")
    
    # Combine chunks and save
    full_audio = np.concatenate(audio_data, axis=0)
    write(FILENAME, FS, full_audio)
    return FILENAME

def transcribe_and_summarize(audio_file):
    # 1. Transcribe (Local Whisper)
    print(" Transcribing...")
    model = WhisperModel("base", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(audio_file)
    transcript = " ".join([s.text for s in segments])
    
    # 2. Summarize (Local Ollama)
    print(" Creating notes...")
    prompt = f"Convert this lecture transcript into structured notes with LaTeX for formulas:\n\n{transcript}"
    #Use any model
    response = ollama.chat(model='gemma3:4b', messages=[
        {'role': 'user', 'content': prompt},
    ])
    
    return response['message']['content']

# --- RUN ---
file_path = record_audio()
notes = transcribe_and_summarize(file_path)

with open("Lecture_Notes.md", "w") as f:
    f.write(notes)
print("\n✅ Done! Check 'Lecture_Notes.md'")
