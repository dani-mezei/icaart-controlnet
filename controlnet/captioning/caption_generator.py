import torch
import numpy as np
import argparse
import os
import json
from PIL import Image
import pandas as pd
import cv2
from controlnet_diffusers.custom.utils import validate_dir, create_dir_if_not_exists
from controlnet_diffusers.captioning.blip2_captions_extractor import Blip2CaptionsExtractor


OUTPUT_FILE = "prompt.jsonl"


def parse_args(input_args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("--load_json",
                        type=str,
                        help="Path to the json file containing the arguments."
                        )
    parser.add_argument("--save_json",
                        type=str,
                        help="Path to save the arguments in a json file after parsing."
                        )

    parser.add_argument("--blip2_dir",
                        type=str,
                        help="Path to the blip2 model."
                        )
    parser.add_argument("--input_dir",
                        type=str,
                        help="Path to the input directory containing the images."
                        )
    parser.add_argument("--output_dir",
                        type=str,
                        help="Path to the output directory where the prompts will be saved."
                        )
    parser.add_argument("--csv_to_image_paths",
                        type=str,
                        nargs="*",
                        help="Paths to csv files with an image column containing the relative path to the images which need to be captioned."
                        )
    parser.add_argument("--question",
                        type=str,
                        default=None,
                        help=("Question which will be answered by the model to generate the prompt."
                              "Recommended format: Question: <Some question> Answer:"
                              )
                        )
    parser.add_argument("--max_new_tokens",
                        type=int,
                        default=50,
                        help="Maximum number of tokens to generate."
                        )
    parser.add_argument("--step_size",
                        type=int,
                        default=64,
                        help="Number of images to process in each iteration."
                        )
    parser.add_argument("--prompt_editor",
                        action="store_true",
                        help=("Flag to enable prompt editing using a GUI."
                              "Press ENTER to save and go to next prompt."
                              "Press BACKSPACE to delete the last character."
                              )
                        )
    parser.add_argument("--prompt_suffix",
                        type=str,
                        default="",
                        help="String to be added to the end of each generated prompt."
                        )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    # Load the arguments from the json file
    if args.load_json is not None:
        with open(args.load_json, "r") as f:
            t_args = argparse.Namespace()
            t_args.__dict__.update(json.load(f))
            args = parser.parse_args(namespace=t_args)

    # Save the arguments to a json file
    if args.save_json is not None:
        with open(args.save_json, "w") as f:
            json.dump(vars(args), f, indent=4)

    # Validate the input directory
    validate_dir(args.input_dir)
    validate_dir(args.blip2_dir)
    # Create the output directory if it does not exist
    create_dir_if_not_exists(args.output_dir)

    return args


def save_changed_prompts(changed_prompts, output_dir):
    output_file_path = os.path.join(output_dir, OUTPUT_FILE)
    # Update the current prompts with the new prompts
    with open(output_file_path, "r") as f:
        data = f.readlines()
        for i, line in enumerate(data):
            image_name = json.loads(line)["image"]
            if image_name in changed_prompts:
                data[i] = {
                    "mask": image_name,
                    "image": image_name,
                    "prompt": changed_prompts[image_name]
                }

    # Save the updated prompts to the jsonl file
    with open(output_file_path, "w") as f:
        for line in data:
            f.write(json.dumps(line) + "\n")


def save_prompts(image_names, prompts, output_dir):
    # Save the prompts to a jsonl file
    with open(os.path.join(output_dir, OUTPUT_FILE), "w") as f:
        for image_name, prompt in zip(image_names, prompts):
            data = {
                "mask": image_name,
                "image": image_name,
                "prompt": prompt
            }
            f.write(json.dumps(data) + "\n")


def load_image_names(image_paths):
    image_names = []
    for image_path in image_paths:
        df = pd.read_csv(image_path, sep=";")
        image_names.extend(df["image"].tolist())
    image_names = [os.path.basename(image_name) for image_name in image_names]
    return image_names


# Function to display image and prompt on a canvas
def display_canvas(image, prompt, canvas_width, canvas_height):
    # Create a white canvas
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

    # Get the dimensions of the image
    img_height, img_width = image.shape[:2]

    # Place the image on the left side of the canvas
    canvas[:img_height, :img_width] = image

    # Place the prompt text below the image
    y_offset = img_height + 30
    cv2.putText(canvas, prompt, (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

    return canvas


def show_window(image, prompt):
    # Create a window
    cv2.namedWindow("Image and Prompt Editor", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Image and Prompt Editor", 1200, 800)

    while True:
        canvas_width = cv2.getWindowImageRect("Image and Prompt Editor")[2]
        canvas_height = cv2.getWindowImageRect("Image and Prompt Editor")[3]

        display_img = display_canvas(
            image, prompt, canvas_width, canvas_height)
        cv2.imshow("Image and Prompt Editor", display_img)

        key = cv2.waitKey(1) & 0xFF

        if key == 13:     # ENTER key to save the prompt
            break
        elif 32 <= key <= 126:  # ASCII printable characters
            prompt += chr(key)
        elif key == 8:     # BACKSPACE key
            prompt = prompt[:-1]

    cv2.destroyAllWindows()
    return prompt


def main(args=None):
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    blip2 = Blip2CaptionsExtractor(args.blip2_dir, device=device)

    generated_prompts = []

    if args.csv_to_image_paths is None:
        image_names = os.listdir(args.input_dir)
        image_names.sort()
    else:
        image_names = load_image_names(args.csv_to_image_paths)

    # Remove duplicates
    image_names = list(set(image_names))

    # Process the images in batches
    for i in range(0, len(image_names), args.step_size):
        slice_end = min(i + args.step_size, len(image_names))

        image_paths = [os.path.join(args.input_dir, image_name)
                       for image_name in image_names[i:slice_end]]
        # Extract prompts for the images
        prompts = blip2.extract(
            image_paths, max_new_tokens=args.max_new_tokens, prompt=args.question)
        generated_prompts.extend(prompts)

    # Add the prompt suffix to the generated prompts
    generated_prompts = [
        prompt + args.prompt_suffix for prompt in generated_prompts]

    # Save the prompts
    save_prompts(image_names, generated_prompts, args.output_dir)

    # Display the image and prompt if the prompt editor is enabled
    if args.prompt_editor:
        changed_prompts = {}
        for image_name, prompt in zip(image_names, generated_prompts):
            image_path = os.path.join(args.input_dir, image_name)
            image = cv2.imread(image_path)
            new_prompt = show_window(image, prompt)

            if new_prompt != prompt:
                changed_prompts[image_name] = new_prompt

        # Save the changed prompts
        save_changed_prompts(changed_prompts, args.output_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args)
