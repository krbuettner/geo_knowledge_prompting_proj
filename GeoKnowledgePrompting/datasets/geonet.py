import json
import glob
import os
import pickle
import math
import random
from collections import defaultdict

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import read_json, write_json, mkdir_if_missing
from dassl.utils import listdir_nohidden


@DATASET_REGISTRY.register()
class Geonet(DatasetBase):
    dataset_dir = "GeoNet"
    tasks = ['GeoImnet', 'GeoPlaces', 'GeoUniDA']
    splits = ['train', 'test']
    domains = ["usa", 'asia']
    annotations = {
        'GeoImnet': "geoImnet_metadata.json",
        'GeoPlaces': "geoPlaces_metadata.json",
        'GeoUniDA': "geoUniDA_metadata.json",
    }

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.task_dir = os.path.join(self.dataset_dir, cfg.DATASET.TASK)
        self.check_input_domains(
            cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS
        )

        self.meta_data_path = os.path.join(self.dataset_dir, self.annotations[cfg.DATASET.TASK])
        self.task_meta_data = json.load(open(self.meta_data_path))
        self.classname_to_id = {c["category_name"]: int(c["category_id"]) for c in
                                self.task_meta_data['categories']}  # category_name to category_id
        self.id_to_classname = {v: k for k, v in self.classname_to_id.items()}  # category_id

        # self.anno_dir = os.path.join(self.dataset_dir, "annotations")
        # self.split_path = os.path.join(self.dataset_dir, "split_zhou_OxfordPets.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")  # todo check
        mkdir_if_missing(self.split_fewshot_dir)  # todo check

        train = self._read_data(cfg.DATASET.SOURCE_DOMAINS, "train")
        test = self._read_data(cfg.DATASET.TARGET_DOMAINS, "test")

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir,
                                        f"{cfg.DATASET.SOURCE_DOMAINS[0]}_shot_{num_shots}-seed_{seed}.pkl")

            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train = data["train"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                data = {"train": train}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, test = self.subsample_classes(train, test, subsample=subsample)

        super().__init__(train_x=train, test=test)

    def _read_data(self, input_domains, split):
        for domain, dname in enumerate(input_domains):

            data = self.task_meta_data['{}_{}'.format(dname, split)]
            images = data['images']
            annotations = data['annotations']
            items = []
            assert len(images) == len(annotations), "length of images and annotations do not match"
            for img, anno in zip(images, annotations):
                assert img['id'] == anno['image_id'], "image id does not match. Got {} and {}".format(
                    img['id', anno['image_id']])
                impath = os.path.join(self.task_dir, img['filename'])
                lid = anno['category']
                lname = self.id_to_classname[lid]
                image_id = img['id']
                item = Datum(impath=impath, label=lid, domain=dname, classname=lname)
                items.append(item)
        return items

    @staticmethod
    def subsample_classes(*args, subsample="all"):
        """Divide classes into two groups. The first group
        represents base classes while the second group represents
        new classes.

        Args:
            args: a list of datasets, e.g. train, val and test.
            subsample (str): what classes to subsample.
        """
        assert subsample in ["all", "base", "new"]

        if subsample == "all":
            return args

        dataset = args[0]
        labels = set()
        for item in dataset:
            labels.add(item.label)
        labels = list(labels)
        labels.sort()
        n = len(labels)
        # Divide classes into two halves
        m = math.ceil(n / 2)

        print(f"SUBSAMPLE {subsample.upper()} CLASSES!")
        if subsample == "base":
            selected = labels[:m]  # take the first half
        else:
            selected = labels[m:]  # take the second half
        relabeler = {y: y_new for y_new, y in enumerate(selected)}

        output = []
        for dataset in args:
            dataset_new = []
            for item in dataset:
                if item.label not in selected:
                    continue
                item_new = Datum(
                    impath=item.impath,
                    label=relabeler[item.label],
                    classname=item.classname
                )
                dataset_new.append(item_new)
            output.append(dataset_new)

        return output
