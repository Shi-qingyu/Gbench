from diffusers import DiffusionPipeline
from diffusers.models.unets.unet_3d_condition import UNet3DConditionModel
from diffusers.utils import export_to_video

from diffusers.utils import export_to_video
from pathlib import Path
from tqdm import tqdm

import torch

unet = UNet3DConditionModel.from_pretrained("adamdad/videocrafterv2_diffusers", torch_dtype=torch.float16).to("cuda:3")
pipe = DiffusionPipeline.from_pretrained("cerspense/zeroscope_v2_576w", unet=unet, torch_dtype=torch.float16).to("cuda:3")

prompt_root = Path("prompts/my_prompts_per_dimension")
save_root = Path("sampled_videos/videocrafterv2")
save_root.mkdir(exist_ok=True)

for txt_file in prompt_root.iterdir():
    dimension = txt_file.stem
    dimension_root = save_root.joinpath(dimension)
    dimension_root.mkdir(exist_ok=True)

    with open(txt_file.as_posix(), "r") as file:
        prompts = file.read().splitlines()
    
    for prompt in tqdm(prompts):
        for i in range(5):
            generator = torch.Generator("cuda").manual_seed(i)
            image = pipe(
                prompt,
                num_inference_steps=40, 
                height=320, 
                width=576, 
                num_frames=24,
                generator=generator,
            ).frames[0]
            save_name = prompt + "-" + str(i) + ".mp4"
            save_path = dimension_root.joinpath(save_name)
            export_to_video(image, save_path, fps=8)