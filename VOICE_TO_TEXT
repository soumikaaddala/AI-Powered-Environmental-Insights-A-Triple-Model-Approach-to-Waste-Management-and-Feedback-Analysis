pip install torch torchaudio torchvision numpy librosa soundfile gtts pytesseract opencv-python

!pip install g2p_en

!apt-get install -y espeak

!pip install deep-phonemizer

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

import pandas as pd

file_path = "/content/drive/MyDrive/voicecode/TestReviews.csv"  # Update path if necessary

try:
df = pd.read_csv(file_path, encoding="utf-8")
print("✅ CSV loaded successfully!")
print("🔹 First 5 rows of the CSV:")
print(df.head())  # Show first 5 rows
print("🔹 Columns in CSV:", df.columns)
except Exception as e:
print(f"❌ Error reading CSV: {e}")

from gtts import gTTS
import IPython.display as display

Ensure the correct column name

text_column = df.columns[0]  # Use the first column

Iterate through rows and generate audio

for index, row in df.iterrows():
text = str(row[text_column]).strip()  # Convert to string and clean spaces
if text:
tts = gTTS(text=text, lang='en')

audio_filename = f"output_{index}.mp3"  
    tts.save(audio_filename)  

    print(f"Playing: {audio_filename}")  
    display.display(display.Audio(audio_filename, autoplay=True))

print("✅ All text-to-speech conversions are done!")

import os
import shutil

source_folder = "/content/"
destination_folder = "/content/destination_folder/"

Ensure the destination folder exists

os.makedirs(destination_folder, exist_ok=True)

Move all .mp3 files

for file in os.listdir(source_folder):
if file.endswith(".mp3"):
shutil.move(os.path.join(source_folder, file), destination_folder)

print("Files moved successfully!")

pip install deep_phonemizer

!pip install gradio pygame

gui

import os
import gradio as gr

Path where audio files are stored

destination_folder = "/content/destination_folder"  # Update if needed

Get all MP3 files in sorted order

audio_files = sorted([f for f in os.listdir(destination_folder) if f.endswith(".mp3")])
current_index = 0

def play_next():
global current_index
if current_index < len(audio_files):
audio_path = os.path.join(destination_folder, audio_files[current_index])
current_index += 1
return audio_path  # Gradio will play the file
else:
return None  # No more audio files

Create Gradio UI

with gr.Blocks() as app:
gr.Markdown("# 🎵 Audio Player - Click 'Next' to Play")
audio_output = gr.Audio(label="Audio Player")
next_button = gr.Button("Next")

next_button.click(play_next, outputs=audio_output)

Launch the Gradio UI

app.launch(share=True)
