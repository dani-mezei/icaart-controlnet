import os


def validate_dir(dir_path):
    if dir_path is None:
        raise ValueError("You must specify the directory path.")
    
    if isinstance(dir_path, list):
        for path in dir_path:
            path = os.path.abspath(path)
            if not os.path.exists(path):
                raise FileNotFoundError(f"The directory {path} does not exist.")
    else:
        dir_path = os.path.abspath(dir_path)
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"The directory {dir_path} does not exist.")

def create_dir_if_not_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)