import torch
from torchvision import transforms as T
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from imageio import mimsave
from os import getcwd, listdir, path, makedirs, system

system("clear || cls")

# Defaults
df_selected = path.join(getcwd(), "train_subset.csv")
case_number = 101
day_number = 20

# Configs
while True:
    phase = input("Enter the Phase (Train, Eval, Test): ").strip().lower()

    if phase in ["train", "eval", "test"]:
        df_selected = pd.read_csv(path.join(getcwd(), f"{phase}_subset.csv"))
        break


def cases_labels_show():
    print(f"\nIncluded Cases:\n{np.sort(df_selected.case.unique())}\n")


def days_labels_show():
    print(f"\n\nIncluded Days:\n{np.sort(df_selected.day[df_selected.case == case_number].unique())}\n")


found_flag = False

while True:
    if found_flag:
        break

    cases_labels_show()
    case_number = int(input("Enter the number of Case: ").strip())

    if case_number in df_selected.case.unique():
        while True:
            days_labels_show()
            day_number = int(input("Enter the number of Day: ").strip())

            if day_number in df_selected.day[df_selected.case == case_number].unique():
                found_flag = True
                break
            else:
                system("clear || cls")
                cases_labels_show()
                print(f"\nThe Entered Day ({day_number}) NOT found!\nTry another one...\n")
    else:
        system("clear || cls")
        print(f"\nThe Entered Case ({case_number}) NOT found!\nTry another one...\n")

case_folder_path = path.join(getcwd(),
                             f"uw-madison-gi-tract-image-segmentation\\train\\case{case_number}\\case{case_number}_day{day_number}\\scans")
image_names = listdir(case_folder_path)
output = []


# RLE Decoder
def rle_decode(img_size, segments):
    mask_ = torch.zeros(3, img_size[0] * img_size[1], dtype=torch.float32)

    for i, segment in enumerate(segments):
        if str(segment) == "nan":
            continue

        segment = segment.split()
        starts = np.array(segment[::2], dtype=np.int32) - 1
        ends = starts + np.array(segment[1::2], dtype=np.int32)

        for s, e in zip(starts, ends):
            mask_[i, s:e] = 1

    return mask_.reshape((3, img_size[0], img_size[1]))


# Transforms
train_transforms = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Lambda(lambda x: (x - x.min()) / (x.max() - x.min())),
    T.Lambda(lambda x: x.repeat(3, 1, 1))
])

target_transform = T.Compose([T.Resize((224, 224))])

# Live images show
for image_name in image_names:
    image_path = path.join(case_folder_path, image_name)

    components = image_path.split(path.sep)[-3::2]
    id_ = '_'.join([components[0], components[1]]).rsplit('_', 4)[0]

    slice_ = df_selected[df_selected.id == id_]

    img = Image.open(image_path)

    if len(slice_) > 0:
        mask = rle_decode(img.size[::-1], slice_.iloc[0][["large_bowel", "small_bowel", "stomach"]])
        img = train_transforms(img)
        mask = target_transform(mask)

        out = cv2.addWeighted(img.permute(1, 2, 0).numpy(), 1, mask.permute(1, 2, 0).numpy(), 0.5, 0)
    else:
        img = train_transforms(img)
        out = img.permute(1, 2, 0).numpy()

    output.append(out)
    cv2.imshow('', out)

    cv2.waitKey(100)

# Save the images as a gif file
print("\nGIF file has been saving...\n")

gif_dir = "../images/gifs"
makedirs(path.join(getcwd(), gif_dir), exist_ok=True)
gif_name = f"case{case_number}_day{day_number}.gif"
gif_saving_addr = path.join(getcwd(), gif_dir, gif_name)

mimsave(gif_saving_addr, [np.uint8(img * 255) for img in output], duration=0.6, loop=0)

print(f"\nThe GIF file has been saved successfully in {gif_saving_addr}\n")
