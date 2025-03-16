from transformers import LlavaProcessor, LlavaForConditionalGeneration, TextIteratorStreamer
import cv2
from PIL import Image
import torch
# Select the best available device
device = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Using device: {device}")

import time
from threading import Thread
import os


model_id = "llava-hf/llava-interleave-qwen-0.5b-hf"

processor = LlavaProcessor.from_pretrained(model_id)
model = LlavaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16)
model.to(device)

def replace_video_with_images(text, frames):
    return text.replace("<video>", "<image>" * frames)


def sample_frames(video_file, num_frames):
    video = cv2.VideoCapture(video_file)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = total_frames // num_frames
    frames = []
    for i in range(total_frames):
        ret, frame = video.read()
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not ret:
            continue
        if i % interval == 0:
            frames.append(pil_img)
    video.release()
    return frames

def process_image(image_file, text_message):
    image = Image.open(image_file).convert("RGB")
    prompt = f"<|im_start|>user <image>\n{text_message}<|im_end|><|im_start|>assistant"
    
    # Prepare the input for the model
    inputs = processor(prompt, [image], return_tensors="pt").to(device, torch.float16)
    
    # Set up the text streamer to get the model's response
    streamer = TextIteratorStreamer(processor, **{"max_new_tokens": 900, "skip_special_tokens": False})
    generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=900)
    
    generated_text = ""

    # Run the model generation in a separate thread for non-blocking operation
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # Stream the output text from the model and collect all generated text
    buffer = ""
    for new_text in streamer:
        buffer += new_text
    generated_text = buffer[len(f"<|im_start|>user <image>\n{text_message}<|im_end|><|im_start|>assistant"):]

    return generated_text

# Function to handle video processing and text generation
def process_video(video_file, text_message):
    # Sample 12 frames from the video
    image_list = sample_frames(video_file, 12)
    
    # Create the prompt with the correct number of image tokens
    image_tokens = "<image>" * len(image_list)
    prompt = f"<|im_start|>user {image_tokens}\n{text_message}<|im_end|><|im_start|>assistant"
    inputs = processor(prompt, image_list, return_tensors="pt").to("cuda", torch.float16)
    
    # Set up the text streamer to get the model's response
    streamer = TextIteratorStreamer(processor, **{"max_new_tokens": 200, "skip_special_tokens": False})
    generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=100)
    
    generated_text = ""

    # Run the model generation in a separate thread for non-blocking operation
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # Stream the output text from the model and collect all generated text
    buffer = ""
    for new_text in streamer:
        buffer += new_text
    generated_text = buffer[len(f"<|im_start|>user {image_tokens}\n{text_message}<|im_end|><|im_start|>assistant"):]
    # print(generated_text)
    return generated_text

# Main function to decide whether it's an image or a video and process accordingly
def process_input(file_path, text_message):
    generated_text = ""
    if os.path.splitext(file_path)[1].lower() in [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"]:
        # print(f"Processing image: {file_path}")
        generated_text = process_image(file_path, text_message)
        # print(f"Generated Text: {generated_text}")
    elif os.path.splitext(file_path)[1].lower() in [".avi", ".mp4", ".mov", ".mkv", ".flv", ".wmv", ".mjpeg"]:
        # print(f"Processing video: {file_path}")
        generated_text = process_video(file_path, text_message)
        # print(f"Generated Text: {generated_text}")
    else:
        print("Unsupported file format. Please provide an image or video.")
    return generated_text


# file_path = "/Masking.png"  
# text_message = "is there is any security violations, give the answer in meaningfull way , tampering/ masking the camera in the ATM is a process violation, the man was touching the camera"


# process_input(file_path, text_message)

