import argparse
import os


def parse_args(input_args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_dir",
                        type=str,
                        help="The directory where the data is stored."
                        )
    parser.add_argument("--dataset_name",
                        type=str,
                        default="data_pipeline",
                        help="The name of the dataset."
                        )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.dataset_dir is None:
        raise ValueError("You must specify the data directory.")

    return args


def replace_value(content, variable_name, new_value, data_pipeline_path):
    # Go until I find the DATA_DIR variable
    data_dir_index = content.find(variable_name)
    print(f"{variable_name=} found at index {data_dir_index}")
    if data_dir_index == -1:
        raise ValueError(f"The {variable_name} variable was not found in the file.")

    # Find the value of the variable
    data_dir_value_index = content.find("=", data_dir_index)
    if data_dir_value_index == -1:
        raise ValueError(f"The {variable_name} variable was not found in the file.")

    # Find the end of the value
    data_dir_value_end_index = content.find("\n", data_dir_value_index)
    if data_dir_value_end_index == -1:
        raise ValueError(f"The {variable_name} variable was not found in the file.")

    # Replace the value of the DATA_DIR variable
    new_content = content[:data_dir_value_index + 1] + \
        f" \"{new_value}\"" + content[data_dir_value_end_index:]

    return new_content


def process_python_file(args):
    # Absolute path to the data_pipeline.py file
    data_pipeline_path = os.path.join(
        os.path.dirname(__file__), "data_pipeline.py")
    if not os.path.exists(data_pipeline_path):
        raise FileNotFoundError(
            f"The file {data_pipeline_path} does not exist.")

    with open(data_pipeline_path, "r") as file:
        content = file.read()

    content = replace_value(content, "DATA_DIR", args.dataset_dir, data_pipeline_path)
    content = replace_value(content, "DATASET_NAME", args.dataset_name , data_pipeline_path)

    with open(data_pipeline_path, "w") as file:
        file.write(content)
    


def main():
    args = parse_args()
    process_python_file(args=args)


if __name__ == "__main__":
    main()
