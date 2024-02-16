run_api = False
SSD_1B = False
import os

# Use GPU
gpu_info = os.popen("nvidia-smi").read()
if "failed" in gpu_info:
    print("Not connected to a GPU")
    is_gpu = False
else:
    print(gpu_info)
    is_gpu = True
print(is_gpu)


from IPython.display import clear_output


def check_enviroment():
    try:
        import torch

        print("Enviroment is already installed.")
    except ImportError:
        print("Enviroment not found. Installing...")
        # Install requirements from requirements.txt
        os.system("pip install -r requirements.txt")
        # Install gradio version 3.48.0
        os.system("pip install gradio==3.39.0")
        # Install python-dotenv
        os.system("pip install python-dotenv")
        # Clear the output
        clear_output()

        print("Enviroment installed successfully.")


# Call the function to check and install Packages if necessary
check_enviroment()


from IPython.display import clear_output
import os
import gradio as gr
import numpy as np
import PIL
import base64
import io
import torch
from diffusers import UNet2DConditionModel, DiffusionPipeline, LCMScheduler

# SDXL
from diffusers import UNet2DConditionModel, DiffusionPipeline, LCMScheduler

# Get the current directory
current_dir = os.getcwd()
model_path = os.path.join(current_dir)
# Set the cache path
cache_path = os.path.join(current_dir, "cache")
MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "1024"))
SECRET_TOKEN = os.getenv("SECRET_TOKEN", "default_secret")

# Uncomment the following line if you are using PyTorch 1.10 or later
# os.environ["TORCH_USE_CUDA_DSA"] = "1"

if is_gpu:
    # Uncomment the following line if you want to enable CUDA launch blocking
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
else:
    # Uncomment the following line if you want to use CPU instead of GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Get the current directory
current_dir = os.getcwd()
model_path = os.path.join(current_dir)

# Set the cache path
cache_path = os.path.join(current_dir, "cache")

if not SSD_1B:

    unet = UNet2DConditionModel.from_pretrained(
        "latent-consistency/lcm-sdxl",
        torch_dtype=torch.float16,
        variant="fp16",
        cache_dir=cache_path,
    )
    pipe = DiffusionPipeline.from_pretrained(
        # "stabilityai/stable-diffusion-xl-base-1.0",
        "stabilityai/sdxl-turbo",
        unet=unet,
        torch_dtype=torch.float16,
        variant="fp16",
        cache_dir=cache_path,
    )

    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    if torch.cuda.is_available():
        pipe.to("cuda")
else:
    # SSD-1B
    from diffusers import LCMScheduler, AutoPipelineForText2Image

    pipe = AutoPipelineForText2Image.from_pretrained(
        "segmind/SSD-1B",
        torch_dtype=torch.float16,
        variant="fp16",
        cache_dir=cache_path,
    )
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    if torch.cuda.is_available():
        pipe.to("cuda")

    # load and fuse
    pipe.load_lora_weights("latent-consistency/lcm-lora-ssd-1b")
    pipe.fuse_lora()


def generate(
    prompt: str,
    negative_prompt: str = "",
    seed: int = 0,
    width: int = 1024,
    height: int = 1024,
    guidance_scale: float = 0.0,
    num_inference_steps: int = 4,
    secret_token: str = "",
) -> PIL.Image.Image:
    if secret_token != SECRET_TOKEN:
        raise gr.Error(
            f"Invalid secret token. Please fork the original space if you want to use it for yourself."
        )

    generator = torch.Generator().manual_seed(seed)

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
        output_type="pil",
    ).images[0]
    return image


clear_output()

from IPython.display import display


def generate_image(prompt="A beautiful and sexy girl"):
    # Generate the image using the prompt
    generated_image = generate(
        prompt=prompt,
        negative_prompt="",
        seed=0,
        width=1024,
        height=1024,
        guidance_scale=0.0,
        num_inference_steps=4,
        secret_token="default_secret",  # Replace with your secret token
    )
    # Display the image in the Jupyter Notebook
    display(generated_image)


if not run_api:
    secret_token = gr.Text(
        label="Secret Token",
        max_lines=1,
        placeholder="Enter your secret token",
    )
    prompt = gr.Text(
        label="Prompt",
        show_label=False,
        max_lines=1,
        placeholder="Enter your prompt",
        container=False,
    )
    result = gr.Image(label="Result", show_label=False)
    negative_prompt = gr.Text(
        label="Negative prompt",
        max_lines=1,
        placeholder="Enter a negative prompt",
        visible=True,
    )
    seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0)

    width = gr.Slider(
        label="Width",
        minimum=256,
        maximum=MAX_IMAGE_SIZE,
        step=32,
        value=1024,
    )
    height = gr.Slider(
        label="Height",
        minimum=256,
        maximum=MAX_IMAGE_SIZE,
        step=32,
        value=1024,
    )
    guidance_scale = gr.Slider(
        label="Guidance scale", minimum=0, maximum=2, step=0.1, value=0.0
    )
    num_inference_steps = gr.Slider(
        label="Number of inference steps", minimum=1, maximum=8, step=1, value=4
    )
    inputs = [
        prompt,
        negative_prompt,
        seed,
        width,
        height,
        guidance_scale,
        num_inference_steps,
        secret_token,
    ]
    iface = gr.Interface(
        fn=generate,
        inputs=inputs,
        outputs=result,
        title="Image Generator",
        description="Generate images based on prompts.",
    )

    iface.launch()


if run_api:
    with gr.Blocks() as demo:
        gr.HTML(
            """
        <div style="z-index: 100; position: fixed; top: 0px; right: 0px; left: 0px; bottom: 0px; width: 100%; height: 100%; background: white; display: flex; align-items: center; justify-content: center; color: black;">
            <div style="text-align: center; color: black;">
                <p style="color: black;">This space is a REST API to programmatically generate images using LCM LoRA SSD-1B.</p>
                <p style="color: black;">It is not meant to be directly used through a user interface, but using code and an access key.</p>
            </div>
        </div>"""
        )
        secret_token = gr.Text(
            label="Secret Token",
            max_lines=1,
            placeholder="Enter your secret token",
        )
        prompt = gr.Text(
            label="Prompt",
            show_label=False,
            max_lines=1,
            placeholder="Enter your prompt",
            container=False,
        )
        result = gr.Image(label="Result", show_label=False)
        negative_prompt = gr.Text(
            label="Negative prompt",
            max_lines=1,
            placeholder="Enter a negative prompt",
            visible=True,
        )
        seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0)

        width = gr.Slider(
            label="Width",
            minimum=256,
            maximum=MAX_IMAGE_SIZE,
            step=32,
            value=1024,
        )
        height = gr.Slider(
            label="Height",
            minimum=256,
            maximum=MAX_IMAGE_SIZE,
            step=32,
            value=1024,
        )
        guidance_scale = gr.Slider(
            label="Guidance scale", minimum=0, maximum=2, step=0.1, value=0.0
        )
        num_inference_steps = gr.Slider(
            label="Number of inference steps", minimum=1, maximum=8, step=1, value=4
        )

        inputs = [
            prompt,
            negative_prompt,
            seed,
            width,
            height,
            guidance_scale,
            num_inference_steps,
            secret_token,
        ]
        prompt.submit(
            fn=generate,
            inputs=inputs,
            outputs=result,
            api_name="run",
        )

    # demo.queue(max_size=32).launch()
    # Launch the Gradio app with multiple workers and debug mode enabled
    # demo.queue(max_size=32).launch(debug=True)# For Standard
    demo.queue(max_size=32).launch(server_name="0.0.0.0", server_port=7860)  # Docker
