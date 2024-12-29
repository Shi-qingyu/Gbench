import json
from pathlib import Path


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
    root = Path("I2V")

    for category in root.iterdir():
        full_info = []
        image_dir = category.joinpath("1-1")
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


if __name__ == "__main__":
    prepare_i2v_full_info()