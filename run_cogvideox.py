from diffusers.pipelines.cogvideo.pipeline_cogvideox import CogVideoXPipeline
from diffusers.utils import export_to_video

from pathlib import Path
from tqdm import tqdm

import torch


pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-5b", torch_dtype=torch.bfloat16).to("cuda:1")
pipe.vae.enable_tiling()

prompt_root = Path("prompts/my_prompts_per_dimension")
save_root = Path("sampled_videos/cogvideox-5b")
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
                num_inference_steps=30, 
                generator=generator,
            ).frames[0]
            save_name = prompt + "-" + str(i) + ".mp4"
            save_path = dimension_root.joinpath(save_name)
            export_to_video(image, save_path, fps=8)