import PIL
import requests
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
import os

prompt = "turn him into cyborg"
diffusion_steps = 999

exp_name = prompt.replace(" ", "-")
out_path = os.path.join("work/examples/out/instruct-pix2pix", f"{exp_name}-{diffusion_steps}")
os.makedirs(out_path, exist_ok=True)

model_id = "timbrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
pipe.to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

url = "https://raw.githubusercontent.com/timothybrooks/instruct-pix2pix/main/imgs/example.jpg"


def download_image(url):
    image = PIL.Image.open(requests.get(url, stream=True).raw)
    image.save(os.path.join(out_path, "before.png"))
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image


image = download_image(url)

images = pipe(prompt, image=image, num_inference_steps=diffusion_steps, image_guidance_scale=1).images
images[0].save(os.path.join(out_path, "after.png"))
