from diffusers import DiffusionPipeline, TextToVideoSDPipeline
from diffusers.utils import export_to_video

from accelerate import PartialState  # Can also be Accelerator or AcceleratorState

from pathlib import Path
from tqdm import tqdm

import torch


pipe = DiffusionPipeline.from_pretrained(
    "cerspense/zeroscope_v2_576w",
    torch_dtype=torch.float16
)
distributed_state = PartialState()
pipe.to(distributed_state.device)

prompt_root = Path("FVD/prompts")
save_root = Path("sampled_videos/zeroscope")
save_root.mkdir(exist_ok=True)

for txt_file in prompt_root.iterdir():
    dimension = txt_file.stem
    dimension_root = save_root.joinpath(dimension)
    dimension_root.mkdir(exist_ok=True)

    with open(txt_file.as_posix(), "r") as file:
        prompt_list = file.read().splitlines()

    with distributed_state.split_between_processes(prompt_list, apply_padding=True) as prompts:
        for prompt in tqdm(prompts):
            for i in range(5):
                generator = torch.Generator(distributed_state.device).manual_seed(i)
                image = pipe(
                    prompt,
                    num_inference_steps=30, 
                    height=320, 
                    width=576, 
                    num_frames=24,
                    generator=generator,
                ).frames[0]
                save_name = prompt + "-" + str(i) + ".mp4"
                save_path = dimension_root.joinpath(save_name)
                export_to_video(image, save_path, fps=8)