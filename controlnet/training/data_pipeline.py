import os

import datasets
import pandas as pd

_VERSION = datasets.Version("0.0.2")

_DESCRIPTION = "TODO"
_HOMEPAGE = "TODO"
_LICENSE = "TODO"
_CITATION = "TODO"

# Define a custom features dictionary, which will be used to encode the dataset
_FEATURES = datasets.Features(
    {
        "image": datasets.Image(),
        "mask": datasets.Image(),
        "prompt": datasets.Value("string"),
    },
)

_DEFAULT_CONFIG = datasets.BuilderConfig(name="default", version=_VERSION)

DATA_DIR = "***IT WILL BE FILLED DURING RUNTIME***"
DATASET_NAME = "***IT WILL BE FILLED DURING RUNTIME***"


class CustomDataset(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [_DEFAULT_CONFIG]
    DEFAULT_CONFIG_NAME = "default"
    _config_file = "prompt.jsonl"    # image captions
    _images_dir = "image"
    _masks_dir = "mask"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dataset_name = DATASET_NAME

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=_FEATURES,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """
        This method is called once and defines the splits of the dataset to produce
        Here we have only one split, the train split

        Args:
            dl_manager: DownloadManager that can be used to download and extract data from the URLs

        Returns:
            A list of `SplitGenerator`, defining the split of train data
        """
        metadata_path = os.path.join(DATA_DIR, self._config_file)
        images_dir = os.path.join(DATA_DIR, self._images_dir)
        masks_dir = os.path.join(DATA_DIR, self._masks_dir)

        print("\n=== Split generators ===")
        print(f"{metadata_path=}")
        print(f"{images_dir=}")
        print(f"{masks_dir=}\n")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "metadata_path": metadata_path,
                    "images_dir": images_dir,
                    "masks_dir": masks_dir,
                },
            ),
        ]

    def _generate_examples(self, metadata_path, images_dir, masks_dir):
        """
        This method is called to produce the examples of the dataset
        Yields examples, one by one (image filename and a dictionary of features)

        Args:
            metadata_path: path to the metadata file
            images_dir: path to the images directory
            masks_dir: path to the conditioning images directory

        Yields:
            A dictionary of features, where the key is the feature name and the value is the feature value
        """

        metadata = pd.read_json(metadata_path, lines=True)

        for _, row in metadata.iterrows():
            prompt = row["prompt"]

            try:
                image_path = row["image"]
                image_path = os.path.join(images_dir, image_path)
                image = open(image_path, "rb").read()

                mask_path = row["mask"]
                mask_path = os.path.join(
                    masks_dir, row["mask"]
                )
                mask = open(mask_path, "rb").read()

                yield row["image"], {
                    "prompt": prompt,
                    "image": {
                        "path": image_path,
                        "bytes": image,
                    },
                    "mask": {
                        "path": mask_path,
                        "bytes": mask,
                    },
                }
            except Exception as ex:
                print(ex)
