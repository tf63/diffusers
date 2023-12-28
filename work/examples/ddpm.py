from diffusers import DDPMPipeline

# load model and scheduler
pipe = DDPMPipeline.from_pretrained("google/ddpm-cat-256").to("cuda")

# run pipeline in inference (sample random noise and denoise)
images = pipe(batch_size=8, num_inference_steps=1000).images

for i, image in enumerate(images):
    # save image
    image.save(f"work/examples/out/ddpm_generated_image{i}.png")
