# Import necessary libraries used by the code

import torch  # PyTorch for tensor computations and model handling
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline  # Hugging Face Transformers for model and pipeline
import gradio as gr  # Gradio for creating web-based interfaces
from langdetect import detect_langs  # langdetect for language detection
import time

# Function to detect the language of the given text
def detect_language(text):
    return detect_langs(text)

#Convert the received json timestamps to .srt to be used in youtube, etc.
def convert_to_srt(timestamps):
    srt_content = ""
    for index, entry in enumerate(timestamps):
        start_time = entry["timestamp"][0]
        end_time = entry["timestamp"][1] 
        if end_time ==None: 
            end_time = start_time + 3
        text = entry["text"]
        #print("{}-{}".format(start_time, end_time))
        # Convert start and end times to SRT format
        start_time_srt = "{:02}:{:02}:{:02},{:03}".format(int(start_time // 3600), int((start_time % 3600) // 60), int(start_time % 60), int((start_time % 1) * 1000))
        end_time_srt = "{:02}:{:02}:{:02},{:03}".format(int(end_time // 3600), int((end_time % 3600) // 60), int(end_time % 60), int((end_time % 1) * 1000))

        # Append to SRT content
        srt_content += f"{index + 1}\n{start_time_srt} --> {end_time_srt}\n{text}\n\n"

    return srt_content


# Function to transcribe audio files
def transcribe(audio, selected_file, option):
    start = time.time()
    if selected_file != "None":
        audio = selected_file
    if audio is None:
        # Return a  message if no file is uploaded
        return "Please upload an audio file first.", None, None
    # Use the pipeline to transcribe the audio
    if option=="Yes":
        result = pipe(audio, return_timestamps=True) #If word level timestamps is needed replace True with "word"
        srt = convert_to_srt(result["chunks"])
    else:
        result = pipe(audio)
        srt="Not requested" 
    
    # Detect the language of the transcribed text
    #print(result)
    acc = detect_language(result["text"])
    if not "chunks"  in result:
        print("No chunks")
        chunks = '{"message": "Not requested"}'
    else:
        chunks = result['chunks']
    end = time.time()
    duration = "### Processing took: {} seconds".format(round(end-start,2))
    return result["text"], result, acc, chunks, srt, duration
    

# Function to update the file upload component
def update_file_upload(selected_file):
    if selected_file != "None":
        return gr.update(visible=False)
    else:
        return gr.update(visible=True)

def main(): #Basic Python part begins
    
    # Determine the device to use (GPU if available, otherwise CPU)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # Set the appropriate torch data type based on the device
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Model identifier for the pre-trained model
    model_id = "openai/whisper-large-v3-turbo"  # https://huggingface.co/openai/whisper-large-v3-turbo

    """ Load the pre-trained model with specified configurations
    Downloads the model from huggingface before loading into memory, 
    if run via dockerfile the model is downloaded during the build stage"""
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,  # The identifier of the pre-trained model to load
        torch_dtype=torch_dtype,  # Set the data type for the model's tensors (float16 if using GPU, otherwise float32, CPUs don't have support for float16)
        low_cpu_mem_usage=True,  # Optimize the model loading to use less CPU memory
        use_safetensors=True  # Use the safetensors format for loading the model, which is more secure and efficient
    ).to(device)  # Move the model to the selected device (GPU if available, otherwise CPU)

    # Load the processor associated with the model
    processor = AutoProcessor.from_pretrained(model_id)

    # Create a pipeline for automatic speech recognition
    #pipeline simplifies the process of using pre-trained models from huggingface
    global pipe
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model, #defined above
        tokenizer=processor.tokenizer, #defined above
        feature_extractor=processor.feature_extractor,
        chunk_length_s=20,  # Length of audio chunks  in seconds for processing
        batch_size=30,  # Batch size for inference, number of chunks processed parallel
        torch_dtype=torch_dtype, #defined above, float16 or float32
        device=device, #defined above, cuda:0 or cpu
    )

    # List of internal example files provided with the packages, feel free to alter or convert to dynamic
    example_files = [
    "None",
    "examples/English1.mp3",
    "examples/English-tooLong.mp3",
    "examples/Finnish-Olympic1952-100m-Final.mp3",
    "examples/German1.mp3",
    "examples/OSR_us_000_0010_8k.wav",
    "examples/Spanish1.mp3",
    "examples/Video-Teddy-Roosevelt.mp4"
]
    
    # Create a Gradio interface --> UI part starts
    with gr.Blocks() as demo:
        # Add a header and instructions
        gr.Markdown("# Audio/Video File Transcription\n### Upload a file or use a provided example to get its transcription. Supported formats include all common audio and video formats")
        # File upload component
        file_input = gr.File(label="Upload your own file")
        example_dropdown = gr.Dropdown(label="Or select a provided example file", choices=example_files, value="None") 
        # Radio button component
        gr.Markdown("### Choose if to create timestamps or not. If created, provided in json and srt formated")
        radio_options = gr.Radio(label="Create timestamps?", choices=["Yes", "No"], value="No")
        # Button to trigger the transcription
        transcribe_button = gr.Button("Transcribe")
        time_output = gr.Markdown()
        # Output components for displaying results
        with gr.Tab("Transcription"):
            text_output = gr.Textbox(label="Transcribed Text")
            jsontext_output = gr.JSON(label="Full text result as JSON")
            accuracy_output = gr.Textbox(label="Language and accuracy")
        with gr.Tab("Timed transcription"):
            with gr.Row():
                jsonstamps_output = gr.JSON(label="Timestamps as JSON")
                srt_output = gr.Textbox(label="Timestamps as srt format", lines=20)
        
        
        #Hide or show the file upload part depending on the example_file selection
        example_dropdown.change(fn=update_file_upload, inputs=example_dropdown, outputs=file_input)
        
        # Define the action when the button is clicked
        transcribe_button.click(fn=transcribe, inputs=[file_input, example_dropdown, radio_options], 
                                outputs=[text_output, jsontext_output,  accuracy_output, jsonstamps_output, srt_output, time_output])

    # Launch the Gradio app on the specified port and server name
    demo.launch(server_port=7860, server_name="0.0.0.0")
    # Below is used to launch this on Hippu and access via https://memorylab.fi/demot/asr/
    #demo.launch(server_port=8004, server_name="0.0.0.0", root_path="/demot/asr/")

    
    

if __name__ == "__main__":
    main()
