import os
import whisper
print("WARNING: FILES IN THE FOLDER WILL BE CONVERTED TO MP3S, AND THE ORIGINAL WILL BE DELETED, MAKE A COPY")
dir = input("Path to dir containing video/audio files: ")
print("Converting Files to MP3s...")
mp3s = []
for file in os.listdir(dir):
   filename_without_extension = file.split(".")[:-1]
   os.system(f"ffmpeg -i {dir}/{file} {dir}/{filename_without_extension}.mp3")
   mp3s.append(f"{dir}/{filename_without_extension}.mp3")
   os.remove(file)
print("Done converting Files to MP3s")
print("Transcribing files with whisper-large-v3")
model = whisper.load_model("large-v3")
for file in mp3s:
    audio=whisper.load_audio(file)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)
print("Transcription done")
