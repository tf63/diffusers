from diffusers import DiffusionPipeline, AutoPipelineForText2Image
import torch
import os
import random
import numpy as np


def fix_seed(seed=0):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True


fix_seed(0)

model_path = "runwayml/stable-diffusion-v1-5"
pipeline = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16).to("cuda")
# pipeline = AutoPipelineForText2Image.from_pretrained(model_path, torch_dtype=torch.float16).to("cuda")

prompt = "a photo of real cat"

exp_name = prompt.replace(' ', '-')
print(exp_name)

out_path = f"work/examples/out/{exp_name}"
os.makedirs(out_path, exist_ok=True)

num_steps = 999

for i in range(4):
    # save image
    images = pipeline(prompt, guidance_scale=3.0, num_inference_steps=num_steps).images
    images[0].save(f"{out_path}/{i}_{num_steps}_s3.png")

    # nothing: s7.6
