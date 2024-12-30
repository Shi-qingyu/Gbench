import os
import json
import shutil
import requests
from pathlib import Path

ATHLETICS = ['HighJump', 'JavelinThrow', 'LongJump', 'Shotput', 'PoleVault', 'ThrowDiscus']
INSTRUMENTS = ['PlayingCello', 'PlayingDaf', 'PlayingDhol', 'PlayingFlute', 'PlayingGuitar', 'PlayingPiano', 'PlayingSitar', 'PlayingTabla', 'PlayingViolin']
WATER_SPORTS = ['Diving', 'CliffDiving', 'Rowing', 'Kayaking', 'Surfing']

API_KEY = 'IBgrF0SctWyiA5Il0xlldjLjXjkSwk8wVwCn3Y1CFDRne9OM3CGSz0qb'

def write_video_to_txt():
    root = Path("/data/datasets/VideoBoothDataset/webvid_parsing_videobooth_subset/")
    video_root = Path("/data/datasets/VideoBoothDataset/videos")

    video_paths = []
    for subdir in root.iterdir():
        if subdir.is_dir():
            cnt = 0
            for videoid in subdir.iterdir():
                videoid = videoid.name
                video_name = videoid + ".mp4"
                video_path = video_root.joinpath(video_name)
                video_paths.append(video_path.as_posix())
                cnt += 1
                if cnt >= 70:
                    break

    with open("animal_videos.txt", "w") as file:
        for video_path in video_paths:
            video_path = video_path + "\n"
            file.write(video_path)


def copy_video_to_dst(txt_file, dstdir):
    with open(txt_file, "r") as file:
        video_list = file.read().splitlines()
    
    for video in video_list:
        video_name = Path(video).name
        dst_path = Path(dstdir).joinpath(video_name)
        shutil.copyfile(video, dst_path.as_posix())


def prepare_ucf101(categories, UCF101="/data/datasets/UCF-101", dst="athletics"):
    dst = os.path.join("./FVD/real_videos", dst)
    for category in categories:
        src_root = Path(UCF101).joinpath(category)
        cnt = 0
        for video_path in src_root.iterdir():
            video_name = video_path.name
            dst_path = Path(dst).joinpath(video_name)
            shutil.copyfile(video_path.as_posix(), dst_path.as_posix())
            cnt += 1
            if cnt >= 100:
                break


def prepare_t2v_full_info():
    full_info = []
    root = Path("prompts/my_prompts_per_dimension")

    prompts_en = {}
    for txt_file in root.iterdir():
        dimension = txt_file.stem
        with open(txt_file.as_posix(), "r", encoding="utf-8") as file:
            all_prompt_en = file.read().splitlines()
        for prompt_en in all_prompt_en:
            info = prompts_en.get(prompt_en, {})

            info["dimension"] = info.get("dimension", [])
            info["dimension"].append(dimension)

            auxiliary_info = info.get("auxiliary_info", {})
            if dimension == "multiple_objects" or dimension == "object_class":
                object = prompt_en.replace("a ", "").replace("an ", "")
                auxiliary_info[dimension] = {"object": object}
            elif dimension == "color":
                color = prompt_en.split(" ")[1]
                auxiliary_info[dimension] = {"color": color}
            elif dimension == "appearance_style":
                appearance_style = prompt_en.split(", ")[-1]
                auxiliary_info[dimension] = {"appearance_style": appearance_style}
            elif dimension == "scene":
                auxiliary_info[dimension] = {"scene": {"scene": prompt_en}}
            elif dimension == "spatial_relationship":
                clean_prompt_en = prompt_en.replace("an ", "").replace("a ", "").split(",")[0]
                r_idx_start = clean_prompt_en.find(" on ")
                object_a = clean_prompt_en[:r_idx_start]
                r_idx_end = clean_prompt_en.find(" of ") + 4
                relationship = clean_prompt_en[r_idx_start + 1: r_idx_end - 1]
                object_b = clean_prompt_en[r_idx_end:]
                auxiliary_info[dimension] = {
                    dimension: {
                        "object_a": object_a,
                        "object_b": object_b,
                        "relationship": relationship,
                    }
                }
            elif dimension == "subject_consistency":
                info["dimension"].extend(["dynamic_degree", "motion_smoothness"])
            elif dimension == "overall_consistency":
                info["dimension"].extend(["aesthetic_quality", "imaging_quality"])
            

            if len(auxiliary_info) > 0:
                info["auxiliary_info"] = auxiliary_info
            prompts_en[prompt_en] = info
        
    for prompt_en, info in prompts_en.items():
        case = {
            "prompt_en": prompt_en,
            "dimension": info["dimension"]
        }
        if "auxiliary_info" in info:
            case["auxiliary_info"] = info["auxiliary_info"]

        full_info.append(case)

    print(len(full_info))
    with open("full_info.json", "w", encoding="utf-8") as file:
        json.dump(full_info, file, indent=4, ensure_ascii=False)


def prepare_i2v_full_info():
    root = Path("prompts/image_prompts")

    for category in root.iterdir():
        if category.is_dir():
            full_info = []
            image_dir = category.joinpath("1-1")
            if not image_dir.is_dir():
                image_dir = category
            for image in image_dir.iterdir():
                prompt = image.stem
                dimension = [
                    "subject_consistency",
                    "overall_consistency",
                    "dynamic_degree",
                    "motion_smoothness"
                ]
                case = {
                    "prompt_en": prompt,
                    "dimension": dimension,
                }
                full_info.append(case)
            save_path = category.joinpath("full_info.json")
            with open(save_path.as_posix(), "w") as file:
                json.dump(full_info, file, indent=4)

headers = {
    'Authorization': API_KEY
}

def search_images(query, num_images=10):
    url = f'https://api.pexels.com/v1/search?query={query}&per_page={num_images}'
    response = requests.get(url, headers=headers)
    data = response.json()
    images = data['photos']
    return images


def save_image(image_url, image_name):
    img_data = requests.get(image_url).content
    with open(image_name, 'wb') as img_file:
        img_file.write(img_data)
    print(f"Saved image as {image_name}")

def search_and_save_image(query, num_images, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    images = search_images(query, num_images)
    for i, img in enumerate(images):
        image_url = img['src']['original']
        image_name = os.path.join(save_dir, f"{i}.jpg")
        save_image(image_url, image_name)


from torchvision.transforms import Resize, RandomCrop, Compose
from PIL import Image

def process_image(imagedir):
    transforms = Compose([
        Resize(size=480),
        RandomCrop(size=(480, 720), pad_if_needed=True),
    ])
    imagedir = Path(imagedir)
    for image in imagedir.iterdir():
        image_path = image.as_posix()
        image = Image.open(image_path)
        image = transforms(image)
        image.save(image_path)
    

if __name__ == "__main__":
    full_info_file = "prompts/image_prompts/Weather/full_info.json"
    prompt_file = "prompts/image_prompts/weather.json"

    with open(full_info_file, "r") as f:
        full_info = json.load(f)
    
    with open(prompt_file, "r") as f:
        idx2prompts = json.load(f)
    
    for info in full_info:
        idx = info["prompt_en"]
        prompt = idx2prompts[idx+".jpg"]
        info["prompt_en"] = prompt

    os.remove(full_info_file)
    with open(full_info_file, "r") as f:
        json.dump(full_info, f)