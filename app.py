
#!/usr/bin/env python

import os
import gradio as gr
import numpy as np
import PIL
import base64
import io
import torch

# SSD-1B
#from diffusers import LCMScheduler, AutoPipelineForText2Image

# SDXL
from diffusers import UNet2DConditionModel, DiffusionPipeline, LCMScheduler

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = int(os.getenv('MAX_IMAGE_SIZE', '1024'))
SECRET_TOKEN = os.getenv('SECRET_TOKEN', 'default_secret')

#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():

    #pipe = AutoPipelineForText2Image.from_pretrained("segmind/SSD-1B", torch_dtype=torch.float16, variant="fp16")
    #pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    #pipe.to("cuda")

    # load and fuse
    #pipe.load_lora_weights("latent-consistency/lcm-lora-ssd-1b")
    #pipe.fuse_lora()

    unet = UNet2DConditionModel.from_pretrained("latent-consistency/lcm-sdxl", torch_dtype=torch.float16, variant="fp16")
    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", unet=unet, torch_dtype=torch.float16, variant="fp16")

    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.to('cuda')
    
else:
    pipe = None

def generate(prompt: str,
             negative_prompt: str = '',
             seed: int = 0,
             width: int = 1024,
             height: int = 1024,
             guidance_scale: float = 0.0,
             num_inference_steps: int = 4,
             secret_token: str = '') -> PIL.Image.Image:
    if secret_token != SECRET_TOKEN:
        raise gr.Error(
            f'Invalid secret token. Please fork the original space if you want to use it for yourself.')
        
    generator = torch.Generator().manual_seed(seed)

    image = pipe(prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
                output_type='pil').images[0]
    
    return image

with gr.Blocks() as demo:
    gr.HTML("""
    <div style="z-index: 100; position: fixed; top: 0px; right: 0px; left: 0px; bottom: 0px; width: 100%; height: 100%; background: white; display: flex; align-items: center; justify-content: center; color: black;">
        <div style="text-align: center; color: black;">
            <p style="color: black;">This space is a REST API to programmatically generate images using LCM LoRA SSD-1B.</p>
            <p style="color: black;">It is not meant to be directly used through a user interface, but using code and an access key.</p>
        </div>
    </div>""")
    secret_token = gr.Text(
        label='Secret Token',
        max_lines=1,
        placeholder='Enter your secret token',
    )
    prompt = gr.Text(
        label='Prompt',
        show_label=False,
        max_lines=1,
        placeholder='Enter your prompt',
        container=False,
    )
    result = gr.Image(label='Result', show_label=False)
    negative_prompt = gr.Text(
        label='Negative prompt',
        max_lines=1,
        placeholder='Enter a negative prompt',
        visible=True,
    )
    seed = gr.Slider(label='Seed',
                    minimum=0,
                    maximum=MAX_SEED,
                    step=1,
                    value=0)

    width = gr.Slider(
        label='Width',
        minimum=256,
        maximum=MAX_IMAGE_SIZE,
        step=32,
        value=1024,
    )
    height = gr.Slider(
        label='Height',
        minimum=256,
        maximum=MAX_IMAGE_SIZE,
        step=32,
        value=1024,
    )
    guidance_scale = gr.Slider(
        label='Guidance scale',
        minimum=0,
        maximum=2,
        step=0.1,
        value=0.0)
    num_inference_steps = gr.Slider(
        label='Number of inference steps',
        minimum=1,
        maximum=8,
        step=1,
        value=4)

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
        api_name='run',
    )

demo.queue(max_size=32).launch()