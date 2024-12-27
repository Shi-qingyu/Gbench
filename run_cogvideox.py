from diffusers.pipelines.cogvideo.pipeline_cogvideox import CogVideoXPipeline
from diffusers.utils import export_to_video

from accelerate import PartialState  # Can also be Accelerator or AcceleratorState

from pathlib import Path
from tqdm import tqdm

import torch


pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-5b", torch_dtype=torch.bfloat16)
pipe.vae.enable_tiling()
distributed_state = PartialState()
pipe.to(distributed_state.device)

prompt_root = Path("prompts/my_prompts_per_dimension")
save_root = Path("sampled_videos/cogvideox-5b_")
save_root.mkdir(exist_ok=True)

for txt_file in prompt_root.iterdir():
    dimension = txt_file.stem
    dimension_root = save_root.joinpath(dimension)
    dimension_root.mkdir(exist_ok=True)

    with open(txt_file.as_posix(), "r") as file:
        prompts = file.read().splitlines()
    
    for i in range(0, len(prompts), 2):
        _prompts = prompts[i: i+2]
        with distributed_state.split_between_processes(_prompts, apply_padding=True) as prompt:
            prompt = prompt[0]
            for i in range(5):
                save_name = prompt + "-" + str(i) + ".mp4"
                save_path = dimension_root.joinpath(save_name)
                if save_path.is_file():
                    break
                else:
                    generator = torch.Generator(distributed_state.device).manual_seed(i)
                    image = pipe(
                        prompt,
                        num_inference_steps=30, 
                        generator=generator,
                    ).frames[0]
                    save_name = prompt + "-" + str(i) + ".mp4"
                    save_path = dimension_root.joinpath(save_name)
                    export_to_video(image, save_path, fps=8)