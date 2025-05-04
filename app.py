# Import the modal library - Modal is a cloud platform that makes it easy to run Python code in the cloud
# It handles infrastructure, scaling, and deployment for you
import modal
from typing import Optional

# Define the CUDA version and related parameters
# CUDA is NVIDIA's parallel computing platform that allows using GPUs for faster processing
# These parameters help create the right environment for our AI model to run efficiently
cuda_version = "11.8.0"  # Downgraded from 12.4.0 for better compatibility
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

# Create a custom Docker image using Modal's Image API
# This image contains all the software dependencies our application needs
# Think of it like preparing a specialized computer with all the tools installed
image = (
    # Install system dependencies using apt-get (Ubuntu's package manager)
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install(
        "git",  # For version control and downloading code repositories
        "ffmpeg",  # For audio/video processing, encoding, and decoding
    )
    # Install PyTorch (machine learning framework) with CUDA support
    # These specific versions are chosen for compatibility with the GPU
    .pip_install(
        "torch==2.0.0",  # Deep learning framework - core library
        "torchaudio==2.0.0",  # PyTorch extension for audio processing
        "numpy<2.0",  # Numerical computing library, for array operations
        index_url="https://download.pytorch.org/whl/cu118",  # Special URL for CUDA 11.8 enabled versions
    )
    # Install Whisper and other required packages from standard PyPI
    .pip_install(
        "openai-whisper",  # Speech recognition model from OpenAI
        "ffmpeg-python",   # Python bindings for ffmpeg
        "requests",        # HTTP library for making API requests
        "fastapi[standard]",  # Web framework for building APIs
    )
)

# Define the Modal App with a name and the image we created above
app = modal.App("example-whisper", image=image)

# Create a persistent storage volume for caching
# This allows us to reuse downloaded models and processed files between runs
cache_vol = modal.Volume.from_name("whisper-cache", create_if_missing=True)

# Create a completely self-contained web endpoint function as shown in the documentation
@app.function(
    gpu="T4",  # Request a T4 GPU for acceleration
    volumes={"/cache": cache_vol},  # Mount the cache volume
    image=image
)
@modal.fastapi_endpoint(method="POST")
def transcribe_endpoint(item: dict):
    """
    Web endpoint for transcribing audio from a URL.
    
    This endpoint can be called via HTTP POST requests with JSON body:
    {
        "audio_url": "https://example.com/audio.wav"
    }
    
    Returns a JSON response with the transcription.
    """
    import os
    import whisper
    import requests
    import tempfile
    
    # Get audio URL from the request body
    audio_url = item.get("audio_url")
    if not audio_url:
        return {"error": "Missing audio_url parameter"}
    
    try:
        # Set cache directory to our mounted volume
        os.environ["XDG_CACHE_HOME"] = "/cache"
        
        # Load the medium Whisper model
        print("Loading Whisper model...")
        model = whisper.load_model("medium")
        
        print(f"Downloading audio from {audio_url}")
        # Download the audio file from the provided URL
        response = requests.get(audio_url)
        if response.status_code != 200:
            raise Exception(f"Failed to download audio file: {response.status_code}")
        
        # Save the audio file to a temporary location
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_file.write(response.content)
            temp_file_path = temp_file.name
        
        try:
            print(f"Transcribing audio file...")
            # Transcribe the audio file using the Whisper model
            result = model.transcribe(temp_file_path)
            
            # Return just the essential parts for the API response
            return {
                "text": result["text"],
                "segments": result.get("segments", []),
                "language": result.get("language", ""),
            }
        finally:
            # Clean up the temporary file
            os.unlink(temp_file_path)
            
    except Exception as e:
        return {"error": str(e)}

# Define a function for command-line usage
@app.function(
    gpu="T4",  # Request a T4 GPU for acceleration
    volumes={"/cache": cache_vol},  # Mount the cache volume
    image=image
)
def transcribe_cli(audio_url: str):
    """
    Transcribe audio from a URL using the OpenAI Whisper model.
    This function is for command-line usage.
    """
    import os
    import whisper
    import requests
    import tempfile
    
    # Set cache directory to our mounted volume
    os.environ["XDG_CACHE_HOME"] = "/cache"
    
    # Load the medium Whisper model
    print("Loading Whisper model...")
    model = whisper.load_model("medium")
    
    print(f"Downloading audio from {audio_url}")
    # Download the audio file from the provided URL
    response = requests.get(audio_url)
    if response.status_code != 200:
        raise Exception(f"Failed to download audio file: {response.status_code}")
    
    # Save the audio file to a temporary location
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_file.write(response.content)
        temp_file_path = temp_file.name
    
    try:
        print(f"Transcribing audio file...")
        # Transcribe the audio file using the Whisper model
        result = model.transcribe(temp_file_path)
        
        # Return the transcription results
        return result
    finally:
        # Clean up the temporary file
        os.unlink(temp_file_path)

# Define the main function that runs when executed directly
def main():
    # Get the URL from command-line arguments or use a default
    import sys
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        # Default URL to a sample audio file
        url = "https://pub-ebe9e51393584bf5b5bea84a67b343c2.r2.dev/examples_english_english.wav"
    
    # Print the result of transcribing the audio at the given URL
    print(transcribe_cli.remote(url))

# This block ensures the main function only runs when the script is executed directly
if __name__ == "__main__":
    main()