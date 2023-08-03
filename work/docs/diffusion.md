### stable diffusion のデモ

**サンプリングのデモ**

- サンプリングしてみる

```
    from diffusers import DiffusionPipeline
    import torch

    pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
    pipeline.to("cuda")
    image = pipeline("a photo of tennis players").images[0]
    image.save("a.png")
```
