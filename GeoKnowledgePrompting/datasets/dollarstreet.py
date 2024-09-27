# DollarStreet dataset script 

from collections import OrderedDict
import csv
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden, mkdir_if_missing
import numpy as np
import os
from .oxford_pets import OxfordPets
import pickle
from PIL import Image
import re
from torch.utils.data import Dataset, DataLoader
import math

@DATASET_REGISTRY.register()
class DollarStreet(DatasetBase):

	# Key path information
	dataset_dir = "dollarstreet"

	# Constructor
	def __init__(self, cfg):
		print(cfg.DATASET.ROOT)
		root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
		self.dataset_dir = os.path.join(root, self.dataset_dir)

		# Key files/folders in dataset dir
		self.image_dir = os.path.join(self.dataset_dir, "images_resized")                              # Maybe change this eventually (higher-res)
		self.preprocessed = os.path.join(self.dataset_dir, "preprocessed.pkl")
		self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
		mkdir_if_missing(self.split_fewshot_dir)
		self.valid_image_set_dir = os.path.join(self.dataset_dir, "dollarstreet_img_set.pkl")
		self.mapping_dir = os.path.join(self.dataset_dir, 'mapping_obj_classes_in_official_dollar_street.txt')
		self.orig_obj_dir = os.path.join(self.dataset_dir, 'orig_obj_classes_in_official_dollar_street.txt')

		# Make dictionary_new_to_old classes (needed for parsing) 
		with open(self.mapping_dir, 'r') as f:
			new_cats = f.readlines()
		with open(self.orig_obj_dir, 'r') as g:
			orig_cats = g.readlines()
		dictionary_old_to_new_classes = dict()
		dictionary_new_to_old_classes = dict()
		for x, y in zip(orig_cats, new_cats):
			dictionary_old_to_new_classes[x.strip()] = y.strip()
			if y.strip() not in dictionary_new_to_old_classes:             # New can be one-to-many 
				dictionary_new_to_old_classes[y.strip()] = []
			dictionary_new_to_old_classes[y.strip()].append(x.strip())

		# Define DollarStreetDataset Loader - makes loading images easier 
		dollarstreet_dataset = DollarStreetDataset(anno_dir=self.valid_image_set_dir, img_dir=self.image_dir, dictionary_old_to_new_classes=dictionary_old_to_new_classes)
	
		# Get object class names
		with open(os.path.join(self.dataset_dir,'new_obj_classes_in_official_dollar_street.txt'), 'r') as f:
			self.class_ref_name_list = []
			new_cats = f.readlines()
			for n in new_cats:
				self.class_ref_name_list.append(n.strip())
		print('Class ref name list')
		print(self.class_ref_name_list)

		# Get GeoDe Class names
		dollarstreet_classnames = []
		with open(os.path.join(self.dataset_dir,'classnames.txt'), 'r') as f:
			for l in f.readlines():
				dollarstreet_classname = l.strip()
				dollarstreet_classnames.append(dollarstreet_classname)
		print('DollarStreet classnames')
		print(dollarstreet_classnames)

		# Load preprocessed train and test data 
		if os.path.exists(os.path.join(self.dataset_dir, cfg.DATASET.TRAIN_SPLIT + "_" + cfg.DATASET.TEST_SPLIT + "_preprocessed.pkl")):
			print(os.path.join(self.dataset_dir, cfg.DATASET.TRAIN_SPLIT + "_" + cfg.DATASET.TEST_SPLIT + "_preprocessed.pkl"))
			with open(os.path.join(self.dataset_dir, cfg.DATASET.TRAIN_SPLIT + "_" + cfg.DATASET.TEST_SPLIT + "_preprocessed.pkl"), "rb") as f:
				preprocessed = pickle.load(f)
				train = preprocessed["train"]
				test = preprocessed["test"]
		else:

			# Load data with all images
			print(cfg.DATASET.ROOT)
			with open(os.path.join(self.dataset_dir,'split_all_data_10_21_23.pkl'), 'rb') as f:
				all_data = pickle.load(f)

			dictionary_of_data_info = self.dollarstreet_parse_train_val_test_data(dollarstreet_dataset=dollarstreet_dataset, split_data=all_data)
			all_train_img_path_to_label_dict = dictionary_of_data_info['train_img_path_to_label_dict'] 
			all_val_img_path_to_label_dict = dictionary_of_data_info['val_img_path_to_label_dict']
			all_test_img_path_to_label_dict = dictionary_of_data_info['test_img_path_to_label_dict'] 
			all_by_region_train_img_path_to_label_dict = dictionary_of_data_info['by_region_train_img_path_to_label_dict'] 
			all_by_region_val_img_path_to_label_dict = dictionary_of_data_info['by_region_val_img_path_to_label_dict'] 
			all_by_region_test_img_path_to_label_dict = dictionary_of_data_info['by_region_test_img_path_to_label_dict']  
			all_by_country_train_img_path_to_label_dict = dictionary_of_data_info['by_country_train_img_path_to_label_dict']
			all_by_country_val_img_path_to_label_dict = dictionary_of_data_info['by_country_val_img_path_to_label_dict']
			all_by_country_test_img_path_to_label_dict = dictionary_of_data_info['by_country_test_img_path_to_label_dict'] 
			all_by_econ_train_img_path_to_label_dict = dictionary_of_data_info['by_econ_train_img_path_to_label_dict']
			all_by_econ_val_img_path_to_label_dict = dictionary_of_data_info['by_econ_val_img_path_to_label_dict']
			all_by_econ_test_img_path_to_label_dict = dictionary_of_data_info['by_econ_test_img_path_to_label_dict'] 



			print("Train Split: " + str(cfg.DATASET.TRAIN_SPLIT))
			if cfg.DATASET.TRAIN_SPLIT == "am_eu_train": # 5467
				train = {**all_by_region_train_img_path_to_label_dict["am"], **all_by_region_train_img_path_to_label_dict["eu"]}
			elif cfg.DATASET.TRAIN_SPLIT == "am_as_af_train":
				train = {**all_by_region_train_img_path_to_label_dict["am"], **all_by_region_train_img_path_to_label_dict["as"], **all_by_region_train_img_path_to_label_dict["af"]}
			elif cfg.DATASET.TRAIN_SPLIT == "eu_train": # 2434 
				train = all_by_region_train_img_path_to_label_dict["eu"]
			elif cfg.DATASET.TRAIN_SPLIT == "am_train":
				train = all_by_region_train_img_path_to_label_dict["am"]
			elif cfg.DATASET.TRAIN_SPLIT == "as_train": 
				train = all_by_region_train_img_path_to_label_dict["as"]
			elif cfg.DATASET.TRAIN_SPLIT == "af_train": 
				train = all_by_region_train_img_path_to_label_dict["af"]
			#for im in train:
			#	image, img_path, label, country, continent, index, econ = dollarstreet_dataset.__getitem_byid__(im.split('/')[-1][:-4])
			if cfg.DATASET.TEST_SPLIT == "am_eu_test":
				test = {**all_by_region_test_img_path_to_label_dict["am"], **all_by_region_test_img_path_to_label_dict["eu"]}
			elif cfg.DATASET.TEST_SPLIT == "am_as_af_test": 
				test1 = {**all_by_region_test_img_path_to_label_dict["am"], **all_by_region_test_img_path_to_label_dict["as"], **all_by_region_test_img_path_to_label_dict["af"]}
				test2 = {**all_by_region_val_img_path_to_label_dict["am"], **all_by_region_val_img_path_to_label_dict["as"], **all_by_region_val_img_path_to_label_dict["af"]}
				test = {**test1, **test2}
			elif cfg.DATASET.TEST_SPLIT == "as_af_test": 
				test = {**all_by_region_test_img_path_to_label_dict["as"], **all_by_region_test_img_path_to_label_dict["af"]}
			elif cfg.DATASET.TEST_SPLIT == "eu_test": 
				test = all_by_region_test_img_path_to_label_dict["eu"]
			elif cfg.DATASET.TEST_SPLIT == "am_test":
				test = all_by_region_test_img_path_to_label_dict["am"]
			elif cfg.DATASET.TEST_SPLIT == "as_test": 
				test = all_by_region_test_img_path_to_label_dict["as"]
			elif cfg.DATASET.TEST_SPLIT == "af_test": 
				test = all_by_region_test_img_path_to_label_dict["af"]

			elif cfg.DATASET.TEST_SPLIT == "bolivia_full":
				test = {**all_by_country_train_img_path_to_label_dict["Bolivia"], **all_by_country_val_img_path_to_label_dict["Bolivia"], **all_by_country_test_img_path_to_label_dict["Bolivia"]}
			elif cfg.DATASET.TEST_SPLIT == "brazil_full":
				test = {**all_by_country_train_img_path_to_label_dict["Brazil"], **all_by_country_val_img_path_to_label_dict["Brazil"], **all_by_country_test_img_path_to_label_dict["Brazil"]}
			elif cfg.DATASET.TEST_SPLIT == "canada_full":
				test = {**all_by_country_train_img_path_to_label_dict["Canada"], **all_by_country_val_img_path_to_label_dict["Canada"], **all_by_country_test_img_path_to_label_dict["Canada"]}
			elif cfg.DATASET.TEST_SPLIT == "colombia_full":
				test = {**all_by_country_train_img_path_to_label_dict["Colombia"], **all_by_country_val_img_path_to_label_dict["Colombia"], **all_by_country_test_img_path_to_label_dict["Colombia"]}
			elif cfg.DATASET.TEST_SPLIT == "guatemala_full":
				test = {**all_by_country_train_img_path_to_label_dict["Guatemala"], **all_by_country_val_img_path_to_label_dict["Guatemala"], **all_by_country_test_img_path_to_label_dict["Guatemala"]}
			elif cfg.DATASET.TEST_SPLIT == "haiti_full":
				test = {**all_by_country_train_img_path_to_label_dict["Haiti"], **all_by_country_val_img_path_to_label_dict["Haiti"], **all_by_country_test_img_path_to_label_dict["Haiti"]}
			elif cfg.DATASET.TEST_SPLIT == "mexico_full":
				test = {**all_by_country_train_img_path_to_label_dict["Mexico"], **all_by_country_val_img_path_to_label_dict["Mexico"], **all_by_country_test_img_path_to_label_dict["Mexico"]}
			elif cfg.DATASET.TEST_SPLIT == "peru_full":
				test = {**all_by_country_train_img_path_to_label_dict["Peru"], **all_by_country_val_img_path_to_label_dict["Peru"], **all_by_country_test_img_path_to_label_dict["Peru"]}
			elif cfg.DATASET.TEST_SPLIT == "unitedstates_full":
				test = {**all_by_country_train_img_path_to_label_dict["United States"], **all_by_country_val_img_path_to_label_dict["United States"], **all_by_country_test_img_path_to_label_dict["United States"]}

			elif cfg.DATASET.TEST_SPLIT == "eu_as_af_full":
				test_am = {**all_by_region_train_img_path_to_label_dict["eu"], **all_by_region_val_img_path_to_label_dict["eu"], **all_by_region_test_img_path_to_label_dict["eu"]}
				test_as = {**all_by_region_train_img_path_to_label_dict["as"], **all_by_region_val_img_path_to_label_dict["as"], **all_by_region_test_img_path_to_label_dict["as"]}
				test_af = {**all_by_region_train_img_path_to_label_dict["af"], **all_by_region_val_img_path_to_label_dict["af"], **all_by_region_test_img_path_to_label_dict["af"]}
				test = {**test_am, **test_as, **test_af}
			elif cfg.DATASET.TEST_SPLIT == "am_as_af_full":
				test_am = {**all_by_region_train_img_path_to_label_dict["am"], **all_by_region_val_img_path_to_label_dict["am"], **all_by_region_test_img_path_to_label_dict["am"]}
				test_as = {**all_by_region_train_img_path_to_label_dict["as"], **all_by_region_val_img_path_to_label_dict["as"], **all_by_region_test_img_path_to_label_dict["as"]}
				test_af = {**all_by_region_train_img_path_to_label_dict["af"], **all_by_region_val_img_path_to_label_dict["af"], **all_by_region_test_img_path_to_label_dict["af"]}
				test = {**test_am, **test_as, **test_af}
			elif cfg.DATASET.TEST_SPLIT == "as_af_full":
				test_as = {**all_by_region_train_img_path_to_label_dict["as"], **all_by_region_val_img_path_to_label_dict["as"], **all_by_region_test_img_path_to_label_dict["as"]}
				test_af = {**all_by_region_train_img_path_to_label_dict["af"], **all_by_region_val_img_path_to_label_dict["af"], **all_by_region_test_img_path_to_label_dict["af"]}
				test = {**test_as, **test_af}
			elif cfg.DATASET.TEST_SPLIT == "eu_full": 
				test = {**all_by_region_train_img_path_to_label_dict["eu"], **all_by_region_val_img_path_to_label_dict["eu"], **all_by_region_test_img_path_to_label_dict["eu"]}
			elif cfg.DATASET.TEST_SPLIT == "am_full":
				test = {**all_by_region_train_img_path_to_label_dict["am"], **all_by_region_val_img_path_to_label_dict["am"], **all_by_region_test_img_path_to_label_dict["am"]}
			elif cfg.DATASET.TEST_SPLIT == "as_full": 
				test = {**all_by_region_train_img_path_to_label_dict["as"], **all_by_region_val_img_path_to_label_dict["as"], **all_by_region_test_img_path_to_label_dict["as"]}
			elif cfg.DATASET.TEST_SPLIT == "af_full": 
				test = {**all_by_region_train_img_path_to_label_dict["af"], **all_by_region_val_img_path_to_label_dict["af"], **all_by_region_test_img_path_to_label_dict["af"]}
			elif cfg.DATASET.TEST_SPLIT == "high_econ_full": 
				test = {**all_by_econ_train_img_path_to_label_dict["high"], **all_by_econ_val_img_path_to_label_dict["high"], **all_by_econ_test_img_path_to_label_dict["high"]}
			elif cfg.DATASET.TEST_SPLIT == "medium_econ_full": 
				test = {**all_by_econ_train_img_path_to_label_dict["medium"], **all_by_econ_val_img_path_to_label_dict["medium"], **all_by_econ_test_img_path_to_label_dict["medium"]}
			elif cfg.DATASET.TEST_SPLIT == "low_econ_full": 
				test = {**all_by_econ_train_img_path_to_label_dict["low"], **all_by_econ_val_img_path_to_label_dict["low"], **all_by_econ_test_img_path_to_label_dict["low"]}
			else:
				print("Dataset not supported")
				exit()
			print('# Images in Test Split: ' + str(len(test)))
	
			# Make datum list
			train = self.read_data_dollarstreet(train, dollarstreet_classnames)
			test = self.read_data_dollarstreet(test, dollarstreet_classnames)

			preprocessed = {"train": train, "test": test}
			with open(os.path.join(self.dataset_dir, cfg.DATASET.TRAIN_SPLIT + "_" + cfg.DATASET.TEST_SPLIT + "_preprocessed.pkl"), 'wb') as f:
				pickle.dump(preprocessed, f, protocol=pickle.HIGHEST_PROTOCOL)


		print('Now creating shots')
		num_shots = cfg.DATASET.NUM_SHOTS
		if num_shots >= 1:
			seed = cfg.SEED

			preprocessed = os.path.join(self.split_fewshot_dir, cfg.DATASET.TRAIN_SPLIT + "_" + f"shot_{num_shots}-seed_{seed}.pkl")

			if os.path.exists(preprocessed):
				print(
					f"Loading preprocessed few-shot data from {preprocessed}")
				with open(preprocessed, "rb") as file:
					data = pickle.load(file)
					train = data["train"]
			else:
			
				train = self.generate_fewshot_dataset(train,
													  num_shots=num_shots)
				for t in train:
					print(t)
				data = {"train": train}
				print(f"Saving preprocessed few-shot data to {preprocessed}")
				with open(preprocessed, "wb") as file:
					pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)



		# Subsample classes if needed
		subsample = cfg.DATASET.SUBSAMPLE_CLASSES
		train, test = OxfordPets.subsample_classes(train,
												   test,
												   subsample=subsample)

		super().__init__(train_x=train, val=test, test=test)

		# This is necessary for cases where # classes changes 
		self._classnames = self.class_ref_name_list
		self._lab2cname = dict()
		for i, name in enumerate(self.class_ref_name_list):
			self._lab2cname[i] = name
		self._num_classes = len(self.class_ref_name_list)

		print("End of Constructor")

	def read_data_dollarstreet(self, dict_of_images, classnames):
		items = []
		for img_path in dict_of_images:
			curr_label = dict_of_images[img_path]
			curr_label_as_index = self.class_ref_name_list.index(curr_label)
			curr_label_as_name = classnames[curr_label_as_index]
			curr_full_img_path = img_path
			item = Datum(impath=curr_full_img_path, label=curr_label_as_index, classname=curr_label_as_name)
			items.append(item)
		return items

	def dollarstreet_parse_train_val_test_data(self, dollarstreet_dataset, split_data):

		dict_countries_to_continents = {'Burundi': 'af', 'Burkina Faso': 'af', 'India': 'as', 'Malawi': 'af', 'Tanzania': 'af', 'Somalia': 'af', 'Zimbabwe': 'af', 'Haiti': 'am', 'Nigeria': 'af', "Cote d'Ivoire": 'af', 'Togo': 'af', 'Myanmar': 'as', 'Papua New Guinea': 'as', 'Liberia': 'af', 'Rwanda': 'af', 'Cambodia': 'as', 'Bangladesh': 'as', 'Kenya': 'af', 'Peru': 'am', 'Nepal': 'as', 'Philippines': 'as', 'South Africa': 'af', 'Palestine': 'as', 'Tunisia': 'af', 'Indonesia': 'as', 'Colombia': 'am', 'China': 'as', 'Pakistan': 'as', 'Cameroon': 'af', 'Thailand': 'as', 'Bolivia': 'am', 'Serbia': 'eu', 'Ghana': 'af', 'Vietnam': 'as', 'Jordan': 'as', 'Guatemala': 'am', 'Brazil': 'am', 'Ethiopia': 'af', 'Mongolia': 'as', 'Ukraine': 'eu', 'United States': 'am', 'South Korea': 'as', 'Egypt': 'af', 'France': 'eu', 'Kyrgyzstan': 'as', 'Lebanon': 'as', 'Kazakhstan': 'as', 'Mexico': 'am', 'Sri Lanka': 'as', 'Netherlands': 'eu', 'Russia': 'eu', 'Austria': 'eu', 'Iran': 'as', 'Sweden': 'eu', 'United Kingdom': 'eu', 'Romania': 'eu', 'Switzerland': 'eu', 'Spain': 'eu', 'Czech Republic': 'eu', 'Turkey': 'eu', 'Canada': 'am', 'Italy': 'eu', 'Denmark': 'eu'}

		train_img_path_to_label_dict = dict()
		val_img_path_to_label_dict = dict()
		test_img_path_to_label_dict = dict()
		by_region_train_img_path_to_label_dict = dict()
		by_region_val_img_path_to_label_dict = dict()
		by_region_test_img_path_to_label_dict = dict()
		by_country_train_img_path_to_label_dict = dict()
		by_country_val_img_path_to_label_dict = dict()
		by_country_test_img_path_to_label_dict = dict()
		by_econ_train_img_path_to_label_dict = dict()
		by_econ_val_img_path_to_label_dict = dict()
		by_econ_test_img_path_to_label_dict = dict()

		for item in split_data['train']:
			curr_id = item[0]
			image, img_path, label, country, continent, index, econ = dollarstreet_dataset.__getitem_byid__(curr_id)
			train_img_path_to_label_dict[img_path] = label
			if continent not in by_region_train_img_path_to_label_dict:
				by_region_train_img_path_to_label_dict[continent] = dict()
			by_region_train_img_path_to_label_dict[continent][img_path] = label
			if country not in by_country_train_img_path_to_label_dict:
				by_country_train_img_path_to_label_dict[country] = dict()
			by_country_train_img_path_to_label_dict[country][img_path] = label
			if econ not in by_econ_train_img_path_to_label_dict:
				by_econ_train_img_path_to_label_dict[econ] = dict()
			if continent != 'eu':
				by_econ_train_img_path_to_label_dict[econ][img_path] = label
		
		for item in split_data['val']:
			curr_id = item[0]
			image, img_path, label, country, continent, index, econ = dollarstreet_dataset.__getitem_byid__(curr_id)
			val_img_path_to_label_dict[img_path] = label
			if continent not in by_region_val_img_path_to_label_dict:
				by_region_val_img_path_to_label_dict[continent] = dict()
			by_region_val_img_path_to_label_dict[continent][img_path] = label
			if country not in by_country_val_img_path_to_label_dict:
				by_country_val_img_path_to_label_dict[country] = dict()
			by_country_val_img_path_to_label_dict[country][img_path] = label
			if econ not in by_econ_val_img_path_to_label_dict:
				by_econ_val_img_path_to_label_dict[econ] = dict()
			if continent != 'eu':
				by_econ_val_img_path_to_label_dict[econ][img_path] = label

		for item in split_data['test']:
			curr_id = item[0]
			image, img_path, label, country, continent, index, econ = dollarstreet_dataset.__getitem_byid__(curr_id)
			test_img_path_to_label_dict[img_path] = label
			if continent not in by_region_test_img_path_to_label_dict:
				by_region_test_img_path_to_label_dict[continent] = dict()
			by_region_test_img_path_to_label_dict[continent][img_path] = label		
			if country not in by_country_test_img_path_to_label_dict:
				by_country_test_img_path_to_label_dict[country] = dict()
			by_country_test_img_path_to_label_dict[country][img_path] = label
			if econ not in by_econ_test_img_path_to_label_dict:
				by_econ_test_img_path_to_label_dict[econ] = dict()
			if continent != 'eu':
				by_econ_test_img_path_to_label_dict[econ][img_path] = label

		dict_to_return = dict()
		dict_to_return['train_img_path_to_label_dict'] = train_img_path_to_label_dict
		dict_to_return['val_img_path_to_label_dict'] = val_img_path_to_label_dict
		dict_to_return['test_img_path_to_label_dict'] = test_img_path_to_label_dict
		dict_to_return['by_region_train_img_path_to_label_dict'] = by_region_train_img_path_to_label_dict
		dict_to_return['by_region_val_img_path_to_label_dict'] = by_region_val_img_path_to_label_dict
		dict_to_return['by_region_test_img_path_to_label_dict'] = by_region_test_img_path_to_label_dict
		dict_to_return['by_country_train_img_path_to_label_dict'] = by_country_train_img_path_to_label_dict
		dict_to_return['by_country_val_img_path_to_label_dict'] = by_country_val_img_path_to_label_dict
		dict_to_return['by_country_test_img_path_to_label_dict'] = by_country_test_img_path_to_label_dict
		dict_to_return['by_econ_train_img_path_to_label_dict'] = by_econ_train_img_path_to_label_dict
		dict_to_return['by_econ_val_img_path_to_label_dict'] = by_econ_val_img_path_to_label_dict
		dict_to_return['by_econ_test_img_path_to_label_dict'] = by_econ_test_img_path_to_label_dict


		return dict_to_return


# Here is a DollarStreetDataset class to parse information 
class DollarStreetDataset(Dataset):

	# Constructor takes mapping dictionary 
	def __init__(self, anno_dir=None, img_dir=None, dictionary_old_to_new_classes=None):

		# These are locations of files in dataset
		ID_INDEX = 0
		COUNTRY_INDEX = 1
		CONTINENT_INDEX = 3
		IMG_PATH_INDEX = 5
		TOPICS_INDEX = 6
		ECON_INDEX = 8

		# Where the images are located 
		self.img_dir = img_dir 

		# Geography info 
		self.country_list = []
		self.continent_list = []
		self.country_to_continent_dict = dict()
		dict_id_to_string = {1: "low", 2: "medium", 3: "high"}


		# This will hold a dictionary mapping an integer to image information 
		self.img_labels = dict()
		self.img_info_by_img_id = dict()

		# Load img data 
		with open(anno_dir, 'rb') as f:
			img_data = pickle.load(f)

		# Go through image data and store in labels dictionary 
		for i, entry in enumerate(img_data):

			# Update geography stores 
			if entry[CONTINENT_INDEX] not in self.continent_list:
				self.continent_list.append(entry[CONTINENT_INDEX])
			if entry[COUNTRY_INDEX] not in self.country_list:
				self.country_list.append(entry[COUNTRY_INDEX])
				self.country_to_continent_dict[entry[COUNTRY_INDEX]] = entry[CONTINENT_INDEX]

			# Get label 
			curr_topics = re.split(r",(?!\s|-)", entry[TOPICS_INDEX])
			tops_after_proc = self.filter_topics(curr_topics, dictionary_old_to_new_classes)

			# Update img info 
			self.img_labels[i] = {"img_id": entry[ID_INDEX],
									"img_path": self.img_dir + '/' + entry[ID_INDEX] + '.jpg', 
									"class_label": tops_after_proc[0], 
									"country": entry[COUNTRY_INDEX], 
									"continent": entry[CONTINENT_INDEX],
									"econ": dict_id_to_string[round(math.log(float(entry[ECON_INDEX]))/3)]}

			self.img_info_by_img_id[entry[ID_INDEX]] = {"index": i,
									"img_path": self.img_dir + '/' + entry[ID_INDEX] + '.jpg', 
									"class_label": tops_after_proc[0], 
									"country": entry[COUNTRY_INDEX], 
									"continent": entry[CONTINENT_INDEX],
									"econ": dict_id_to_string[round(math.log(float(entry[ECON_INDEX]))/3)]}

	# This parses topic label from file 
	def filter_topics(self, topics, dictionary_old_to_new_classes):
		tops_after_proc = []
		for c in topics:
			if c.lower() in dictionary_old_to_new_classes:
				tops_after_proc.append(dictionary_old_to_new_classes[c.lower()])
		tops_after_proc = list(set(tops_after_proc))
		return tops_after_proc

	# How many images in dataset
	def __len__(self):
		return len(self.img_labels)

	# Get image and info; opens image as well 
	def __getitem__(self, idx):
		img_id = self.img_labels[idx]["img_id"]
		img_path = self.img_labels[idx]["img_path"]
		label = self.img_labels[idx]["class_label"]
		country = self.img_labels[idx]["country"]
		continent = self.img_labels[idx]["continent"]
		econ = self.img_labels[idx]["econ"]
		image = Image.open(img_path)
		return image, img_path, label, country, continent, img_id, econ

	# Get image and info; opens image as well 
	def __getitem_byid__(self, img_id):
		index = self.img_info_by_img_id[img_id]["index"]
		img_path = self.img_info_by_img_id[img_id]["img_path"]
		label = self.img_info_by_img_id[img_id]["class_label"]
		country = self.img_info_by_img_id[img_id]["country"]
		continent = self.img_info_by_img_id[img_id]["continent"]
		econ = self.img_info_by_img_id[img_id]["econ"]
		image = Image.open(img_path)
		return image, img_path, label, country, continent, index, econ