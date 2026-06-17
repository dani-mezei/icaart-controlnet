import argparse
import os
import json
from controlnet.custom.utils import validate_dir


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(
        description="Train the model with custom arguments.")

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
                        help="Use Stable Diffusion XL model instead of normal one."
                        )

    parser.add_argument("--multi_gpu",
                        action="store_true",
                        help="Use multiple GPUs for training."
                        )
    parser.add_argument("--num_gpus",
                        type=int,
                        default=2,
                        help="Number of gpus to use for training. Default is 2."
                        )

    parser.add_argument("--memory_intensity",
                        type=str,
                        default="medium",
                        help=("Memory intensity of the model. Choose from 'low', 'medium', 'high'. "
                              "'low' ~12GB GPU, 'medium' ~16GB GPU, and 'high' ~38GB GPU. "
                              "Default is 'medium'.")
                        )

    parser.add_argument("--model_dir",
                        type=str,
                        help="Path to pretrained model or model identifier from huggingface.co/models.",
                        )
    parser.add_argument("--controlnet_dir",
                        type=str,
                        default=None,
                        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
                        " If not specified controlnet weights are initialized from unet.",
                        )
    parser.add_argument("--vae_dir",
                        type=str,
                        default=None,
                        help="Path to an improved VAE to stabilize training. For more details check out: https://github.com/huggingface/diffusers/pull/4038.",
                        )
    parser.add_argument("--output_dir",
                        type=str,
                        help="Path to save the trained model."
                        )
    parser.add_argument("--dataset_dir",
                        type=str,
                        help="Path to the dataset directory."
                        )
    parser.add_argument("--dataset_name",
                        type=str,
                        default="data_pipeline",
                        help="The name of the dataset."
                        )

    parser.add_argument("--resolution",
                        type=int,
                        default=512,
                        help=("Resolution of the input images. All images will be resized to this resolution. "
                              "Must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder.")
                        )
    parser.add_argument("--learning_rate",
                        type=float,
                        default=5e-6,
                        help="Learning rate of the model. Default is 5e-6."
                        )
    parser.add_argument("--scale_lr",
                        action="store_true",
                        default=False,
                        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size."
                        )
    parser.add_argument("--lr_scheduler",
                        type=str,
                        default="constant",
                        help=(
                            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
                            ' "constant", "constant_with_warmup"]')
                        )
    parser.add_argument("--lr_warmup_steps",
                        type=int,
                        default=500,
                        help="Number of steps for the warmup in the lr scheduler."
                        )
    parser.add_argument("--lr_num_cycles",
                        type=int,
                        default=1,
                        help="Number of hard resets of the lr in cosine_with_restarts scheduler."
                        )
    parser.add_argument("--lr_power",
                        type=float,
                        default=1.0,
                        help="Power factor of the polynomial scheduler."
                        )

    parser.add_argument("--train_batch_size",
                        type=int,
                        default=4,
                        help="Batch size for training. Default is 4."
                        )
    parser.add_argument("--num_train_epochs",
                        type=int,
                        default=1,
                        help="Number of epochs to train the model. Default is 1."
                        )

    parser.add_argument("--validation_steps",
                        type=int,
                        default=100,
                        help="Number of steps to validate the model. Default is 100."
                        )
    parser.add_argument("--validation_data_dir",
                        type=str,
                        help=("Path to the validation data directory, which contains an `images` folder"
                              "containing the validation images and a `prompt.jsonl` file containing the validation prompts."
                              "a `prompt.jsonl` file entry looks like: {'image': 'image_name.jpg', 'prompt': 'prompt text'}")
                        )

    parser.add_argument("--gradient_accumulation_steps",
                        type=int,
                        default=1,
                        help="It is used in the low and medium memory intensity. Default is 1."
                        )

    parser.add_argument("--checkpointing_steps",
                        type=int,
                        default=500,
                        help="Number of steps to save the model. Default is 500."
                        )

    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        required=False,
                        help="A seed for reproducible training."
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

    # Validate the arguments that are required
    validate_dir(args.model_dir)
    validate_dir(args.output_dir)
    validate_dir(args.dataset_dir)

    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder."
        )

    if args.validation_data_dir is not None:
        # Make sure validation_data_dir is an absolute path
        validate_dir(args.validation_data_dir)

        # Check if the validation data directory contains the required files
        dir_content = os.listdir(args.validation_data_dir)
        if "images" not in dir_content:
            raise ValueError(
                "The validation data directory must contain an `images` folder."
            )
        if "prompt.jsonl" not in dir_content:
            raise ValueError(
                "The validation data directory must contain a `prompt.jsonl` file."
            )

    return args


def main(args):
    command = "accelerate launch"
    if args.multi_gpu:
        command += f" --multi_gpu"
        command += f" --num_processes={args.num_gpus} "

    if args.scale_lr:
        command += f" --scale_lr "

    # Get absolute path of the data pipeline
    abs_path_of_data_pipeline = os.path.join(
        os.path.dirname(__file__), "data_pipeline.py")
    if not os.path.exists(abs_path_of_data_pipeline):
        raise FileNotFoundError(
            f"The file {abs_path_of_data_pipeline} does not exist.")

    # Get absolute path of the custom data pipeline creator script
    abs_path_of_custom_data_pipeline = os.path.join(
        os.path.dirname(__file__), "create_custom_data_pipeline.py")
    if not os.path.exists(abs_path_of_custom_data_pipeline):
        raise FileNotFoundError(
            f"The file {abs_path_of_custom_data_pipeline} does not exist.")

    # Customizing the data pipeline
    os.system(
        f"python \"{abs_path_of_custom_data_pipeline}\" --data_dir \"{args.dataset_dir}\" --dataset_name \"{args.dataset_name}\"")

    if args.use_sdxl:
        abs_path_of_training_script = os.path.join(
            os.path.dirname(__file__), "train_controlnet_sdxl.py")
    else:
        abs_path_of_training_script = os.path.join(
            os.path.dirname(__file__), "train_controlnet.py")

    command += f" --mixed-precision=fp16 \"{abs_path_of_training_script}\" \
            --pretrained_model_name_or_path=\"{args.model_dir}\" \
            --output_dir=\"{args.output_dir}\" \
            --train_data_dir=\"{abs_path_of_data_pipeline}\" \
            --seed={args.seed} \
            --resolution={args.resolution} \
            --learning_rate={args.learning_rate} \
            --lr_scheduler={args.lr_scheduler} \
            --lr_warmup_steps={args.lr_warmup_steps} \
            --lr_num_cycles={args.lr_num_cycles} \
            --lr_power={args.lr_power} \
            --train_batch_size={args.train_batch_size} \
            --num_train_epochs={args.num_train_epochs} \
            --validation_steps={args.validation_steps} \
            --validation_image=\"{args.validation_data_dir}\" \
            --validation_prompt=\"{args.validation_data_dir}\" \
            --checkpointing_steps={args.checkpointing_steps} "

    if args.controlnet_dir:
        command += f"--controlnet_model_name_or_path=\"{args.controlnet_dir}\" "
    if args.use_sdxl:
        command += f"--pretrained_vae_model_name_or_path=\"{args.vae_dir}\" "


    if args.memory_intensity == "low":
        command += f"--gradient_accumulation_steps={args.gradient_accumulation_steps} \
                --gradient_checkpointing \
                --use_8bit_adam \
                --enable_xformers_memory_efficient_attention \
                --set_grads_to_none"

    elif args.memory_intensity == "medium":
        command += f"--gradient_accumulation_steps={args.gradient_accumulation_steps} \
                --gradient_checkpointing \
                --use_8bit_adam"

    elif args.memory_intensity != "high":
        raise ValueError(
            "Invalid memory intensity. Choose from 'low', 'medium', 'high'.")

    os.system(command)


if __name__ == "__main__":
    args = parse_args()
    main(args)
