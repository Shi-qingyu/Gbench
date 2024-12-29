from diffusers.pipelines.cogvideo.pipeline_cogvideox_image2video import CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video

from accelerate import PartialState  # 也可以使用 Accelerator 或 AcceleratorState

from pathlib import Path
from tqdm import tqdm
from PIL import Image

import torch
from torch.utils.data.dataloader import DataLoader


def main():
    pipe = CogVideoXImageToVideoPipeline.from_pretrained(
        "THUDM/CogVideoX-5b-I2V",
        torch_dtype=torch.bfloat16
    )
    pipe.vae.enable_tiling()

    distributed_state = PartialState()
    
    pipe.to(distributed_state.device)
    

    prompt_root = Path("I2V")
    save_root = Path("sampled_videos/cogvideox-5b-i2v")
    save_root.mkdir(parents=True, exist_ok=True)
    
    for subdir in prompt_root.iterdir():
        if not subdir.is_dir():
            continue  # 跳过非目录文件
        
        dimension = subdir.stem
        dimension_root = save_root.joinpath(dimension)
        dimension_root.mkdir(parents=True, exist_ok=True)
    
        imagedir = subdir.joinpath("16-9")
        if not imagedir.exists() or not imagedir.is_dir():
            print(f"目录 {imagedir} 不存在或不是目录，跳过。")
            continue
    
        image_list = [image for image in imagedir.iterdir() if image.is_file()]
    
        if not image_list:
            print(f"目录 {imagedir} 中没有图片文件，跳过。")
            continue
    
        # 使用 split_between_processes 分割图片列表
        with distributed_state.split_between_processes(image_list, apply_padding=True) as images:
    
            for image in tqdm(images, desc=f"Processing {dimension}"):
                image_path = image  # 提取 Path 对象
                if not image_path.is_file():
                    continue  # 跳过非文件（可能是填充图片）
                
                prompt = image_path.stem
    
                save_name = f"{prompt}-0.mp4"
                save_path = dimension_root.joinpath(save_name)
    
                if save_path.is_file():
                    continue  # 文件已存在，跳过当前图片
    
                try:
                    # 打开并转换图片
                    image = Image.open(image_path.as_posix()).convert("RGB")
                    
                    # 创建生成器（可选，确保一致性）
                    generator = torch.Generator().manual_seed(0)
                    
                    # 进行推理
                    output = pipe(
                        image=image,
                        prompt=prompt,
                        num_inference_steps=30,
                        generator=generator,
                    )
                    
                    frame = output.frames[0]
                    
                    # 保存视频
                    export_to_video(frame, save_path, fps=8)
                
                except Exception as e:
                    print(f"处理 {image_path} 时出错: {e}")
                    continue
                
                finally:
                    # 清理显存
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
