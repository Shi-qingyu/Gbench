from diffusers.pipelines.cogvideo.pipeline_cogvideox_image2video import CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video

from accelerate import PartialState  # 也可以使用 Accelerator 或 AcceleratorState

from pathlib import Path
from tqdm import tqdm
from PIL import Image
import json

import torch


def main():
    pipe = CogVideoXImageToVideoPipeline.from_pretrained(
        "THUDM/CogVideoX-5b-I2V",
        torch_dtype=torch.bfloat16
    )
    pipe.vae.enable_tiling()

    distributed_state = PartialState()
    
    pipe.to(distributed_state.device)
    

    prompt_root = Path("prompts/image_prompts")
    save_root = Path("sampled_videos/cogvideox-5b-i2v")
    save_root.mkdir(parents=True, exist_ok=True)
    
    for subdir in prompt_root.iterdir():
        if not subdir.is_dir():
            continue
            
        dimension = subdir.stem
        dimension_root = save_root.joinpath(dimension)
        dimension_root.mkdir(parents=True, exist_ok=True)

        json_name = dimension.lower() + ".json"
        json_file = prompt_root.joinpath(json_name)
        if not json_file.is_file():
            continue

        with open(json_file, "r") as f:
            prompts = json.load(f)
    
        imagedir = subdir.joinpath("16-9")
        if not imagedir.exists() or not imagedir.is_dir():
            imagedir = subdir

        image_list = [image for image in imagedir.iterdir() if image.is_file()]
    
        if not image_list:
            print(f"目录 {imagedir} 中没有图片文件，跳过。")
            continue
            
        with distributed_state.split_between_processes(image_list, apply_padding=True) as images:
            for image in tqdm(images, desc=f"Processing {dimension}"):
                image_path = image
                if not image_path.is_file():
                    continue
                
                prompt = prompts[image_path.name]
    
                save_name = f"{prompt}-0.mp4"
                save_path = dimension_root.joinpath(save_name)
    
                if save_path.is_file():
                    continue
    
                try:
                    image = Image.open(image_path.as_posix()).convert("RGB")
                    generator = torch.Generator().manual_seed(0)
                    
                    output = pipe(
                        image=image,
                        prompt=prompt,
                        num_inference_steps=30,
                        generator=generator,
                    )
                    
                    frame = output.frames[0]
                    
                    export_to_video(frame, save_path, fps=8)
                
                except Exception as e:
                    print(f"处理 {image_path} 时出错: {e}")
                    continue
                
                finally:
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
