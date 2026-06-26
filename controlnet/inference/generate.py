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
import time
import pandas as pd
import numpy as np
from PIL import Image
from controlnet.custom.utils import validate_dir, create_dir_if_not_exists


# Array fo the latents
latent_repr = []
# Max number of saved latents
MAX_SAVED_LATENTS = 10
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
    parser.add_argument("--batch_size",
                        type=int,
                        default=1,
                        help="Number of masks/prompts processed together. Use 1 on shared GPUs."
                        )
    parser.add_argument("--dtype",
                        type=str,
                        default="auto",
                        choices=["auto", "fp32", "fp16", "bf16"],
                        help="Pipeline dtype. auto uses fp16 on CUDA and fp32 on CPU."
                        )
    parser.add_argument("--cuda_memory_fraction",
                        type=float,
                        default=None,
                        help="Optional per-process CUDA memory cap, e.g. 0.80 on a shared A100."
                        )
    parser.add_argument("--enable_xformers",
                        action="store_true",
                        help="Enable xFormers memory efficient attention if installed."
                        )
    parser.add_argument("--enable_attention_slicing",
                        action="store_true",
                        help="Reduce inference peak VRAM at the cost of speed."
                        )
    parser.add_argument("--enable_vae_slicing",
                        action="store_true",
                        help="Reduce VAE decode memory for multi-image batches."
                        )
    parser.add_argument("--channels_last",
                        action="store_true",
                        help="Use channels_last memory format for CUDA UNet/ControlNet modules."
                        )
    parser.add_argument("--skip_existing",
                        action="store_true",
                        help="Skip samples whose output file already exists."
                        )
    parser.add_argument("--start_index",
                        type=int,
                        default=0,
                        help="Start index, inclusive, after sorting inputs."
                        )
    parser.add_argument("--end_index",
                        type=int,
                        default=None,
                        help="End index, exclusive, after sorting inputs."
                        )
    parser.add_argument("--max_samples",
                        type=int,
                        default=None,
                        help="Maximum number of samples to process after applying start/end indices."
                        )
    parser.add_argument("--seed",
                        type=int,
                        default=None,
                        help="Base seed. Per-sample seed is seed + absolute sample index."
                        )
    parser.add_argument("--manifest_file",
                        type=str,
                        default="generation_manifest.jsonl",
                        help="JSONL manifest filename or path. Relative paths are placed inside output_dir."
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
    if args.batch_size < 1:
        raise ValueError("Batch size must be greater than 0.")
    if args.cuda_memory_fraction is not None and not 0 < args.cuda_memory_fraction <= 1:
        raise ValueError("`--cuda_memory_fraction` must be in the range (0, 1].")
    if args.start_index < 0:
        raise ValueError("Start index must be non-negative.")
    if args.end_index is not None and args.end_index < args.start_index:
        raise ValueError("End index must be greater than or equal to start index.")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("Max samples must be greater than 0.")

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
    if not os.path.isabs(args.manifest_file):
        args.manifest_file = os.path.join(args.output_dir, args.manifest_file)

    return args


def resolve_dtype(dtype):
    if dtype == "auto":
        return torch.float16 if torch.cuda.is_available() else torch.float32
    if dtype == "fp16":
        return torch.float16
    if dtype == "bf16":
        return torch.bfloat16
    return torch.float32


def configure_cuda_memory(args):
    if args.cuda_memory_fraction is None:
        return
    if not torch.cuda.is_available():
        print("Ignoring --cuda_memory_fraction because CUDA is not available.")
        return
    torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction)
    total_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
    usable_gib = total_gib * args.cuda_memory_fraction
    print(f"CUDA memory cap set to {usable_gib:.1f} GiB of {total_gib:.1f} GiB.")


def list_image_names(input_dir, mask_dir):
    image_dir = os.path.join(input_dir, mask_dir)
    images = [name for name in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, name))]
    return sorted(images)


def read_images(input_dir, mask_dir, image_names=None, multi_control=False):
    image_dir = os.path.join(input_dir, mask_dir)
    images = sorted(os.listdir(image_dir)) if image_names is None else image_names
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


def read_prompt_records(input_dir, prompt_file):
    prompt_file = os.path.join(input_dir, prompt_file)
    prompts = pd.read_json(prompt_file, lines=True)
    prompts = prompts.sort_values(by="image")
    return [{"image": row["image"], "prompt": row["prompt"]} for _, row in prompts.iterrows()]


def build_generation_samples(image_names, prompt_records, output_dir):
    prompts_by_image = {record["image"]: record["prompt"] for record in prompt_records}
    samples = []
    for index, image_name in enumerate(image_names):
        if image_name not in prompts_by_image:
            raise ValueError(f"Missing prompt for image {image_name}.")
        samples.append(
            {
                "index": index,
                "image": image_name,
                "prompt": prompts_by_image[image_name],
                "output_path": os.path.join(output_dir, image_name),
            }
        )
    return samples


def filter_samples(samples, start_index=0, end_index=None, max_samples=None):
    selected = samples[start_index:end_index]
    if max_samples is not None:
        selected = selected[:max_samples]
    return selected


def filter_existing_samples(samples, skip_existing=False):
    if not skip_existing:
        return samples
    return [sample for sample in samples if not os.path.exists(sample["output_path"])]


def manifest_row(sample, status, seed=None, elapsed_seconds=None, error=None):
    return {
        "image": sample["image"],
        "prompt": sample["prompt"],
        "output_path": sample["output_path"],
        "seed": seed,
        "status": status,
        "elapsed_seconds": elapsed_seconds,
        "error": error,
    }


def append_manifest(manifest_file, row):
    manifest_dir = os.path.dirname(os.path.abspath(manifest_file))
    create_dir_if_not_exists(manifest_dir)
    with open(manifest_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]


def load_sample_images(input_dir, mask_dir, samples, multi_control=False):
    if multi_control:
        raise NotImplementedError("Range-based lazy loading is not implemented for multiple ControlNets.")
    return [load_image(os.path.join(input_dir, mask_dir, sample["image"])) for sample in samples]


def sample_seed(base_seed, sample):
    return None if base_seed is None else base_seed + sample["index"]


def batch_generator(base_seed, batch):
    if base_seed is None:
        return None
    seeds = [sample_seed(base_seed, sample) for sample in batch]
    if len(set(seeds)) == 1:
        return torch.Generator(device=DEVICE).manual_seed(seeds[0])
    return [torch.Generator(device=DEVICE).manual_seed(seed) for seed in seeds]


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
    configure_cuda_memory(args)
    data_type = resolve_dtype(args.dtype)

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
            args.stable_diffusion_dir, controlnet=controlnet, torch_dtype=data_type, safety_checker=None, feature_extractor=None, requires_safety_checker=False,).to(DEVICE)
    
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    if args.enable_xformers:
        pipeline.enable_xformers_memory_efficient_attention()
    if args.enable_attention_slicing:
        pipeline.enable_attention_slicing()
    if args.enable_vae_slicing:
        pipeline.enable_vae_slicing()
    if args.channels_last and torch.cuda.is_available():
        if isinstance(controlnet, list):
            for controlnet_model in controlnet:
                controlnet_model.to(memory_format=torch.channels_last)
        else:
            controlnet.to(memory_format=torch.channels_last)
        pipeline.unet.to(memory_format=torch.channels_last)
    
    if multi_control:
        if args.start_index or args.end_index is not None or args.max_samples is not None or args.skip_existing:
            raise NotImplementedError("Range filtering and skip_existing are only implemented for a single ControlNet.")
        images, image_names = read_images(args.input_data_dir, args.mask_dir_name, multi_control=True)
        prompts = read_prompts(args.input_data_dir, args.prompt_file_name)
        samples = build_generation_samples(image_names, [{"image": name, "prompt": prompt} for name, prompt in zip(image_names, prompts)], args.output_dir)
    else:
        image_names = list_image_names(args.input_data_dir, args.mask_dir_name)
        prompt_records = read_prompt_records(args.input_data_dir, args.prompt_file_name)
        samples = build_generation_samples(image_names, prompt_records, args.output_dir)
        samples = filter_samples(samples, args.start_index, args.end_index, args.max_samples)
        samples = filter_existing_samples(samples, args.skip_existing)

    # Print the number of images and prompts
    print(f"Number of selected samples: {len(samples)}")

    with torch.inference_mode():
        sample_batches = list(chunks(samples, args.batch_size))
        if multi_control:
            prompts = list(chunks(prompts, args.batch_size))
            images = list(chunks(images, args.batch_size))
            image_names = list(chunks(image_names, args.batch_size))

            for prompt_batch, image_batch, name_batch, sample_batch in zip(prompts, images, image_names, sample_batches):
                start_time = time.perf_counter()
                try:
                    generated_images = pipeline(
                        prompt=prompt_batch,
                        image=image_batch,
                        height=args.height,
                        width=args.width,
                        num_inference_steps=args.num_inference_steps,
                        num_images_per_prompt=args.num_images_per_prompt,
                        generator=batch_generator(args.seed, sample_batch),
                        callback_on_step_end=decode_tensors if args.latent_denoising_steps else None,
                        callback_on_step_end_tensor_inputs=["latents"] if args.latent_denoising_steps else None,
                    )

                    elapsed_seconds = time.perf_counter() - start_time
                    for image, name, sample in zip(generated_images.images, name_batch, sample_batch):
                        image.save(os.path.join(args.output_dir, name))
                        append_manifest(
                            args.manifest_file,
                            manifest_row(sample, "success", seed=sample_seed(args.seed, sample), elapsed_seconds=elapsed_seconds),
                        )
                except Exception as ex:
                    elapsed_seconds = time.perf_counter() - start_time
                    for sample in sample_batch:
                        append_manifest(
                            args.manifest_file,
                            manifest_row(
                                sample,
                                "error",
                                seed=sample_seed(args.seed, sample),
                                elapsed_seconds=elapsed_seconds,
                                error=str(ex),
                            ),
                        )
                    raise
            return

        for sample_batch in sample_batches:
            start_time = time.perf_counter()
            try:
                prompt_batch = [sample["prompt"] for sample in sample_batch]
                image_batch = load_sample_images(args.input_data_dir, args.mask_dir_name, sample_batch, multi_control)
                generated_images = pipeline(
                    prompt=prompt_batch,
                    image=image_batch,
                    height=args.height,
                    width=args.width,
                    num_inference_steps=args.num_inference_steps,
                    num_images_per_prompt=args.num_images_per_prompt,
                    generator=batch_generator(args.seed, sample_batch),
                    callback_on_step_end=decode_tensors if args.latent_denoising_steps else None,
                    callback_on_step_end_tensor_inputs=["latents"] if args.latent_denoising_steps else None,
                )

                elapsed_seconds = time.perf_counter() - start_time
                for image, sample in zip(generated_images.images, sample_batch):
                    image.save(sample["output_path"])
                    append_manifest(
                        args.manifest_file,
                        manifest_row(sample, "success", seed=sample_seed(args.seed, sample), elapsed_seconds=elapsed_seconds),
                    )
            except Exception as ex:
                elapsed_seconds = time.perf_counter() - start_time
                for sample in sample_batch:
                    append_manifest(
                        args.manifest_file,
                        manifest_row(
                            sample,
                            "error",
                            seed=sample_seed(args.seed, sample),
                            elapsed_seconds=elapsed_seconds,
                            error=str(ex),
                        ),
                    )
                raise


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
