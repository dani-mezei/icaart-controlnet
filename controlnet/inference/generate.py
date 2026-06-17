from diffusers import (
    ControlNetModel, 
    AutoencoderKL,
    UniPCMultistepScheduler,
    StableDiffusionControlNetPipeline, 
    StableDiffusionXLControlNetPipeline,
)
from diffusers.utils import load_image
import torch
import argparse
import json
import os
import pandas as pd
import numpy as np
from PIL import Image
from controlnet.custom.utils import validate_dir, create_dir_if_not_exists


# Array fo the latents
latent_repr = []
# Max number of saved latents
MAX_SAVED_LATENTS = 10
# Number of masks processed in paralel in the pipeline
BATCH_SIZE = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    parser.add_argument("--use_sdxl",
                        action="store_true",
                        help="Check this argument if StableDiffusionXLControlNetPipeline needs to be used."
                        )
    parser.add_argument("--vae_dir",
                        type=str,
                        help="Path to VAE, this is only used for StableDiffusionXLControlNetPipeline."
                        )

    parser.add_argument("--controlnet_dir",
                        type=str,
                        help="Path to a directory which contains the ControlNet model.",
                        )
    parser.add_argument("--stable_diffusion_dir",
                        type=str,
                        help="Path to a directory which contains the Stable Diffusion model.",
                        )
    parser.add_argument("--input_data_dir",
                        type=str,
                        help="Path to a directory which contains the input data. An `images` folder which contains the input masks"
                        "and a `prompts.jsonl` file which contains the prompts. "
                        "a `prompt.jsonl` file entry looks like: {'image': 'image_name.jpg', 'prompt': 'prompt text'}"
                        )
    parser.add_argument("--mask_dir_name",
                        type=str, 
                        help="Name of the directory in the input dir, which contain the masks."
                        )
    parser.add_argument("--prompt_file_name",
                        type=str,
                        help="Name of the prompt file with .jsonl at the end."
                        )
    parser.add_argument("--output_dir",
                        type=str,
                        default="./output",
                        help="Path to a directory where the output images will be saved. Default is `./output`."
                        )

    parser.add_argument("--height",
                        type=int,
                        default=512,
                        help="The height in pixels of the generated image. Default is 512."
                        )
    parser.add_argument("--width",
                        type=int,
                        default=512,
                        help="The width in pixels of the generated image. Default is 512."
                        )
    parser.add_argument("--num_inference_steps",
                        type=int,
                        default=20,
                        help="Number of inference steps. Default is 20."
                        )
    parser.add_argument("--num_images_per_prompt",
                        type=int,
                        default=1,
                        help="Number of images to generate per prompt. Default is 1."
                        )
    parser.add_argument("--output_type",
                        type=str,
                        default="pil",
                        choices=["pil", "np"],
                        help=" The output format of the generated image. Choose between PIL.Image or np.array."
                        "`pil` or `np`. Default is `pil`."
                        )
    parser.add_argument("--latent_denoising_steps",
                        action="store_true",
                        help="If set, the latents will be converted to rgb images at each denoising step."
                        "It will be saved in the output directory."
                        )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.load_json is not None:
        with open(args.load_json, "r") as f:
            t_args = argparse.Namespace()
            t_args.__dict__.update(json.load(f))
            args = parser.parse_args(namespace=t_args)

    if args.save_json is not None:
        with open(args.save_json, "w") as f:
            json.dump(vars(args), f, indent=4)

    # Validate the directory paths
    validate_dir(args.stable_diffusion_dir)
    validate_dir(args.input_data_dir)
    validate_dir(args.controlnet_dir)
    if args.vae_dir:
        validate_dir(args.vae_dir)

    if args.num_inference_steps < 1:
        raise ValueError("Number of inference steps must be greater than 0.")

    input_dir_content = os.listdir(args.input_data_dir)
    if args.mask_dir_name not in input_dir_content:
        raise FileNotFoundError(
            f"The input data directory must contain an `{args.mask_dir_name}` folder.")
    if args.prompt_file_name not in input_dir_content:
        raise FileNotFoundError(
            f"The input data directory must contain a `{args.prompt_file_name}` file.")
    
    # If we have multiple controlnets, the images folder should contain subfolders for each controlnet
    if type(args.controlnet_dir) == list:
        # Get the directories in which the controlnets are stored
        controlnet_parents = [os.path.basename(os.path.normpath(controlnet_dir)) for controlnet_dir in args.controlnet_dir]
        
        # Don't allow duplicate controlnet parents
        if len(controlnet_parents) != len(set(controlnet_parents)):
            raise ValueError("The controlnet directories should not have the same directory names.")

        input_dir_content = os.listdir(os.path.join(args.input_data_dir, "images"))

        # Don't allow input directory to have duplicate subfolders
        if len(input_dir_content) != len(set(input_dir_content)):
            raise ValueError("The input data directory should not have duplicate subfolders.")

        # The input_dir has to contain subfolders for each controlnet parent with same name
        if set(controlnet_parents) != set(input_dir_content):
            raise ValueError("For each controlnet, there should be a subfolder in the images folder containing the masks for it, having the same name as the controlnet folder.")

    args.output_dir = os.path.abspath(args.output_dir)
    create_dir_if_not_exists(args.output_dir)

    return args


def read_images(input_dir, mask_dir, multi_control=False):
    image_dir = os.path.join(input_dir, mask_dir)
    images = os.listdir(image_dir)
    images.sort()

    image_names = []
    if multi_control:
        image_list = []
        for control_dir in images:
            control_images = os.listdir(os.path.join(image_dir, control_dir))
            control_images.sort()
            control_image_list = []
            for image in control_images:
                image_names.append(image)
                control_image_list.append(load_image(os.path.join(image_dir, control_dir, image)))
            image_list.append(control_image_list)
        image_list = [[image_list[j][i] for j in range(len(image_list))] for i in range(len(image_list[0]))]
    else:
        image_list = []
        for image in images:
            image_list.append(load_image(os.path.join(image_dir, image)))
            image_names.append(image)

    print(f"Image list shape: {np.shape(image_list)}")

    return image_list, image_names


def read_prompts(input_dir, prompt_file):
    prompt_file = os.path.join(input_dir, prompt_file)
    prompts = pd.read_json(prompt_file, lines=True)
    prompts = prompts.sort_values(by="image")

    prompt_list = []
    for _, row in prompts.iterrows():
        prompt_list.append(row["prompt"])

    return prompt_list


# See an alternative solution here:
# https://huggingface.co/blog/TimothyAlexisVass/explaining-the-sdxl-latent-space
def latents_to_rgb(pipe, latents):
    image_array = StableDiffusionControlNetPipeline.decode_latents(
        pipe, latents)

    # Get the numpy arrays from the np 4D matrix, the first dimension is the batch size
    decoded_np_arrays = []
    for i in range(len(image_array)):
        decoded_np_arrays.append(image_array[i])

    # Scale the image to 0-255 from 0-1
    for i in range(len(decoded_np_arrays)):
        decoded_np_arrays[i] = (decoded_np_arrays[i] * 255).astype("uint8")

    # Convert the numpy arrays to PIL images
    image_array = [Image.fromarray(decoded_np_arrays[i])
                   for i in range(len(decoded_np_arrays))]

    return image_array


def decode_tensors(pipe, step, timestep, callback_kwargs):
    latents = callback_kwargs["latents"]

    image = latents_to_rgb(pipe, latents)
    global latent_repr
    latent_repr.append(image)

    return callback_kwargs


def main(args):
    if args.use_sdxl:
        data_type = torch.float16
    else:
        data_type = torch.float32

    if type(args.controlnet_dir) == list:
        multi_control = True
        # Sort the controlnet directories by their parent directory
        args.controlnet_dir.sort(key=lambda x: os.path.basename(os.path.normpath(x)))
        controlnet = [ControlNetModel.from_pretrained(
            controlnet_dir, torch_dtype=data_type).to(DEVICE) for controlnet_dir in args.controlnet_dir]
    else:
        multi_control = False
        controlnet = ControlNetModel.from_pretrained(
            args.controlnet_dir, torch_dtype=data_type).to(DEVICE)

    if args.use_sdxl:
        vae = AutoencoderKL.from_pretrained(args.vae_dir, torch_dtype=data_type)
        pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
            args.stable_diffusion_dir, controlnet=controlnet, vae=vae, torch_dtype=data_type).to(DEVICE)
    else:
        pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            args.stable_diffusion_dir, controlnet=controlnet, torch_dtype=data_type, safety_checker=None).to(DEVICE)
    
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    
    images, image_names = read_images(args.input_data_dir, args.mask_dir_name, multi_control)
    prompts = read_prompts(args.input_data_dir, args.prompt_file_name)

    # Print the number of images and prompts
    print(f"Number of images: {len(images)}")
    print(f"Number of prompts: {len(prompts)}")

    with torch.no_grad():
        def chunks(lst, n):
            for i in range(0, len(lst), n):
                yield lst[i:i+n]

        prompts = list(chunks(prompts, BATCH_SIZE))
        images = list(chunks(images, BATCH_SIZE))
        image_names = list(chunks(image_names,  BATCH_SIZE))

        for prompt_batch, image_batch, name_batch in zip(prompts, images, image_names):
            generated_images = pipeline(
                prompt=prompt_batch,
                image=image_batch,
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_inference_steps,
                num_images_per_prompt=args.num_images_per_prompt,
                callback_on_step_end=decode_tensors if args.latent_denoising_steps else None,
                callback_on_step_end_tensor_inputs=["latents"] if args.latent_denoising_steps else None,
            )

            for image, name in zip(generated_images.images, name_batch):
                # Add to file name to which prompt it belongs
                image.save(os.path.join(args.output_dir, name))


        # Save the latents as images
        if args.latent_denoising_steps:
            # Check for a the max number of saved latents
            global latent_repr
            num_latents = len(latent_repr)
            if num_latents > MAX_SAVED_LATENTS:
                print(
                    f"Saving only {MAX_SAVED_LATENTS} latents out of {num_latents}.")
                indices = [int(i) for i in range(
                    0, num_latents, num_latents // MAX_SAVED_LATENTS)]
                latent_repr = [latent_repr[i] for i in indices]

            # Create a directory
            latent_dir = os.path.join(args.output_dir, "latents")
            create_dir_if_not_exists(latent_dir)
            for i, latent in enumerate(latent_repr):
                for j, latent_image in enumerate(latent):
                    latent_image.save(os.path.join(
                        latent_dir, f"image_{j}_step_{i}.png"))


if __name__ == "__main__":
    args = parse_args()
    main(args)
