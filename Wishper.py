import streamlit as st
import sounddevice as sd
import pyaudio
import wave
import whisper
import os

# Function to record audio from microphone
def record_audio(file_path, duration=5, samplerate=44100):
    st.info("Recording... Speak now!")
    try:
        # Record audio (Mono - 1 channel)
        recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype="int16")
        sd.wait()  # Wait for the recording to finish

        # Save the recorded audio as a WAV file
        with wave.open(file_path, "wb") as wf:
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(samplerate)
            wf.writeframes(recording.tobytes())

        st.success("Recording complete! Audio saved.")
    except Exception as e:
        st.error(f"Error during recording: {e}")
        return False
    return True

# Function to transcribe audio using Whisper
def transcribe_audio(file_path):
    model = whisper.load_model("base", device="cpu")
    result = model.transcribe(file_path, fp16=False)
    return result["language"], result["text"]

# Function to save transcription to a text file
def save_transcription(text, file_name="transcription.txt"):
    with open(file_name, "w", encoding="utf-8") as file:
        file.write(text)
    return file_name

# Streamlit App
def main():
    # Title and description
    st.title("Japanese Speech Recognition System Development with transcription")
    st.write("Record live audio or upload a file to transcribe Japanese speech into text.")

    # Tabs: Record Audio or Upload File
    tab1, tab2 = st.tabs(["🎙️ Record Audio", "📂 Upload File"])

    # Tab 1: Record Audio
    with tab1:
        duration = st.slider("Select recording duration (seconds):", min_value=3, max_value=30, value=5)
        record_button = st.button("Start Recording")

        if record_button:
            # Record and save the audio
            recorded_file = "recorded_audio.wav"
            success = record_audio(recorded_file, duration=duration)

            if success and os.path.exists(recorded_file):
                # Transcribe the recorded audio
                st.info("Transcribing audio...")
                detected_language, transcription_text = transcribe_audio(recorded_file)

                # Display the results
                st.write(f"**Detected Language:** {detected_language}")
                st.subheader("Transcription:")
                st.write(transcription_text)

                # Provide download option
                transcription_file = save_transcription(transcription_text)
                st.download_button(
                    label="Download Transcription",
                    data=open(transcription_file, "rb"),
                    file_name="transcription.txt",
                    mime="text/plain"
                )

                # Cleanup
                os.remove(recorded_file)

    # Tab 2: Upload Audio File
    with tab2:
        uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a"])

        if uploaded_file is not None:
            # Save the uploaded file
            uploaded_file_path = "uploaded_audio.wav"
            with open(uploaded_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success("File uploaded successfully!")

            # Transcribe the uploaded audio
            if st.button("Transcribe Uploaded File"):
                st.info("Transcribing audio...")
                detected_language, transcription_text = transcribe_audio(uploaded_file_path)

                # Display the results
                st.write(f"**Detected Language:** {detected_language}")
                st.subheader("Transcription:")
                st.write(transcription_text)

                # Provide download option
                transcription_file = save_transcription(transcription_text)
                st.download_button(
                    label="Download Transcription",
                    data=open(transcription_file, "rb"),
                    file_name="transcription.txt",
                    mime="text/plain"
                )

                # Cleanup
                os.remove(uploaded_file_path)

if __name__ == "__main__":
    main()
