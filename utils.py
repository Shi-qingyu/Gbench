import os
import shutil
from pathlib import Path

ATHLETICS = ['HighJump', 'JavelinThrow', 'LongJump', 'Shotput', 'PoleVault', 'ThrowDiscus']
INSTRUMENTS = ['PlayingCello', 'PlayingDaf', 'PlayingDhol', 'PlayingFlute', 'PlayingGuitar', 'PlayingPiano', 'PlayingSitar', 'PlayingTabla', 'PlayingViolin']
WATER_SPORTS = ['Diving', 'CliffDiving', 'Rowing', 'Kayaking', 'Surfing']

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


if __name__ == "__main__":
    prepare_ucf101(WATER_SPORTS, dst="water_sports")