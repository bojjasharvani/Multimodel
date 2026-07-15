import gradio as gr
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load model
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

def generate_caption(image):
    inputs = processor(images=image, return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

demo = gr.Interface(
    fn=generate_caption,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=gr.Textbox(label="Generated Caption"),
    title="AI Image Caption Generator",
    description="Upload an image to generate an AI caption."
)

demo.launch(server_name="0.0.0.0", server_port=7860)