# Script for Geo Knowledge Prompting 

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from collections import OrderedDict
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
import os.path as osp
import pickle
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast


_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
	backbone_name = cfg.MODEL.BACKBONE.NAME
	url = clip._MODELS[backbone_name]
	model_path = clip._download(url)

	try:
		# loading JIT archive
		model = torch.jit.load(model_path, map_location="cpu").eval()
		state_dict = None

	except RuntimeError:
		state_dict = torch.load(model_path, map_location="cpu")

	model = clip.build_model(state_dict or model.state_dict())

	return model

# It is important to add to these 
CUSTOM_TEMPLATES = {
	"DollarStreet": "a photo of a {}.",
	"GeoDE": "a photo of a {}.",
	"OxfordPets": "a photo of a {}, a type of pet.",
	"OxfordFlowers": "a photo of a {}, a type of flower.",
	"FGVCAircraft": "a photo of a {}, a type of aircraft.",
	"DescribableTextures": "a photo of a {}, a type of texture.",
	"EuroSAT": "a centered satellite photo of {}.",
	"StanfordCars": "a photo of a {}.",
	"Food101": "a photo of {}, a type of food.",
	"SUN397": "a photo of a {}.",
	"Caltech101": "a photo of a {}.",
	"UCF101": "a photo of a person doing {}.",
	"ImageNet": "a photo of a {}.",
	"ImageNetSketch": "a photo of a {}.",
	"ImageNetV2": "a photo of a {}.",
	"ImageNetA": "a photo of a {}.",
	"ImageNetR": "a photo of a {}.",
}


# This CLIP 
class TextEncoder(nn.Module):
	def __init__(self, clip_model):
		super().__init__()
		self.transformer = clip_model.transformer
		self.positional_embedding = clip_model.positional_embedding
		self.ln_final = clip_model.ln_final
		self.text_projection = clip_model.text_projection
		self.dtype = clip_model.dtype

	def forward(self, prompts, tokenized_prompts):
		x = prompts + self.positional_embedding.type(self.dtype)
		x = x.permute(1, 0, 2)  # NLD -> LND
		x = self.transformer(x)
		x = x.permute(1, 0, 2)  # LND -> NLD
		x = self.ln_final(x).type(self.dtype)
		# x.shape = [batch_size, n_ctx, transformer.width]
		# take features from the eot embedding (eot_token is the highest number in each sequence)
		x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
		return x

# KEY CLASS FOR PROMPTING FUNCTIONALITY 
class PromptLearner(nn.Module):
	def __init__(self, cfg, classnames, clip_model):
		super().__init__()
		n_cls = len(classnames)
		n_ctx = cfg.TRAINER.COOP.N_CTX
		ctx_init = cfg.TRAINER.COOP.CTX_INIT
		dtype = clip_model.dtype
		ctx_dim = clip_model.ln_final.weight.shape[0]
		clip_imsize = clip_model.visual.input_resolution
		cfg_imsize = cfg.INPUT.SIZE[0]
		assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

		if ctx_init:
			# use given words to initialize context vectors
			temp = 'a photo of a'
			ctx_init = temp.replace("_", " ")
			n_ctx = len(ctx_init.split(" "))
			prompt = clip.tokenize(ctx_init)
			with torch.no_grad():
				embedding = clip_model.token_embedding(prompt).type(dtype)
			ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
			prompt_prefix = ctx_init

		else:
			# random initialization
			if cfg.TRAINER.COOP.CSC:
				print("Initializing class-specific contexts")
				ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
			else:
				print("Initializing a generic context")
				ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
			nn.init.normal_(ctx_vectors, std=0.02)
			prompt_prefix = " ".join(["X"] * n_ctx)

		print(f'Initial context: "{prompt_prefix}"')
		print(f"Number of context words (tokens): {n_ctx}")

		self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

		bias_vectors = torch.empty(1, 512, dtype=dtype)
		nn.init.normal_(bias_vectors, std=0.02)
		self.bias_vectors = nn.Parameter(bias_vectors)

		classnames = [name.replace("_", " ") for name in classnames]
		name_lens = [len(_tokenizer.encode(name)) for name in classnames]
		prompts = [prompt_prefix + " " + name + "." for name in classnames]

		clip_model_ = load_clip_to_cpu(cfg)
		clip_model_.cuda()

		#####################################################################################################
		# Custom Geo Knowledge Prompting Features 
		
		# Text feature general for default prompts  
		print('Generating default text features...')      
		temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
		if cfg.PROMPT.USE_COUNTRY_NAME:
			print('Appending country name...')
			temp = temp.strip('.') + ' in ' + cfg.PROMPT.SINGLE_COUNTRY_NAME_OF_INTEREST + '.'	
		prompts_ = [temp.format(c.replace("_", " ")) for c in classnames]
		prompts_ = torch.cat([clip.tokenize(p) for p in prompts_])
		prompts_ = prompts_.cuda()
		with torch.no_grad():
			text_features = clip_model_.encode_text(prompts_)
			text_features = text_features / text_features.norm(dim=-1, keepdim=True)
		self.text_features = text_features

		# Print status of whether default prompts will be used 
		if cfg.PROMPT.USE_LLM_DESC and not cfg.PROMPT.USE_AVG_LLM_AND_COUNTRY_NAME:
			print('We are using LLM-only, so default prompts will be ignored')
		else:
			print('Default text prompts will be used: ' + str(temp))

		# Get country list depending on mode 
		if cfg.PROMPT.USE_COUNTRY_ENSEMBLE:
			if cfg.PROMPT.PATH_TO_COUNTRY_LIST_FOR_ENSEMBLE:
				print("Path to country ensemble list: " + str(cfg.PROMPT.PATH_TO_COUNTRY_LIST_FOR_ENSEMBLE))
				country_list = []
				with open(cfg.PROMPT.PATH_TO_COUNTRY_LIST_FOR_ENSEMBLE, 'r') as f:
					for line in f.readlines():
						country = line.strip()
						country_list.append(country)
				print(country_list)
				print(str(len(country_list)) + ' countries') 
			else:
				print('No path specified for country ensemble')
				exit()
		else:
			print("We are not using country ensemble")

		# Country ensemble with default CLIP prompts setup  
		if cfg.PROMPT.USE_COUNTRY_ENSEMBLE and cfg.PROMPT.USE_COUNTRY_NAME:
			print("Making default country ensemble...")

			# Make ensemble classifier 
			country_ensemble_list_embed = []
			for country in country_list:
				print(country)
				temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
				temp = temp.strip('.') + ' in ' + country + '.'
				curr_prompts_ = [temp.format(c.replace("_", " ")) for c in classnames]
				curr_prompts_ = torch.cat([clip.tokenize(p) for p in curr_prompts_])
				curr_prompts_ = curr_prompts_.cuda() # 95 x 77
				with torch.no_grad():
					curr_text_features = clip_model_.encode_text(curr_prompts_)
					curr_text_features = curr_text_features / curr_text_features.norm(dim=-1, keepdim=True)
				country_ensemble_list_embed.append(curr_text_features)
			self.country_ensemble_default_features = torch.mean(torch.stack(country_ensemble_list_embed), dim=0) # 95 x 512 
			for i in range(len(classnames)):
				self.country_ensemble_default_features[i, :] /= self.country_ensemble_default_features[i, :].norm()

		# Setup for LLM processing
		if cfg.PROMPT.USE_LLM_DESC:
			print('Making prompts with LLM descriptors')
			
			# Determine which LLM file to use 
			if cfg.PROMPT.USE_COUNTRY_NAME and not cfg.PROMPT.USE_AVG_LLM_AND_COUNTRY_NAME:
				prepend = 'in_country_a_photo_of_a_'
			else:
				prepend = 'a_photo_of_a_'

			# Load prompts
			with open(cfg.PROMPT.ROOT_PATH_TO_LLM_DESC + prepend + cfg.PROMPT.LLM_FILE_NAME, 'rb') as f:
				country_filled_in_prompts = pickle.load(f)

			# Make single country prompts
			print("Prep single country descriptors")
			print('\tCountry LLM descriptors being used: ' + str(cfg.PROMPT.SINGLE_COUNTRY_NAME_OF_INTEREST))
			if cfg.PROMPT.USE_COUNTRY_ENSEMBLE:
				print("Here are sample prompts, though we use ensemble:")
			else:
				print("Here are our country of interest prompts")
			print('Prompts: ' + str(country_filled_in_prompts[cfg.PROMPT.SINGLE_COUNTRY_NAME_OF_INTEREST]))
			country_of_interest = cfg.PROMPT.SINGLE_COUNTRY_NAME_OF_INTEREST
			with torch.no_grad():
				norm_avgd_class_embed_list = []
				for geo_class in classnames:
					input_ex = torch.cat([clip.tokenize(p) for p in country_filled_in_prompts[country_of_interest][geo_class]]).cuda()
					avg_cls_embed = torch.mean(clip_model_.encode_text(input_ex), dim=0)
					norm_avg_cls_embed = avg_cls_embed / avg_cls_embed.norm()
					norm_avgd_class_embed_list.append(norm_avg_cls_embed)
				ensembled_general_descriptor_classifier = torch.stack(norm_avgd_class_embed_list)
				self.country_of_interest_llm_features = ensembled_general_descriptor_classifier

			# In addition to single country make an ensemble 
			if cfg.PROMPT.USE_COUNTRY_ENSEMBLE:
				print("Prep country ensemble descriptors (may take a second)")
				with torch.no_grad():
					country_classifier_list = []
					for country in country_list:
						print(country)
						norm_avgd_class_embed_list = []
						for geo_class in classnames:
							input_ex = torch.cat([clip.tokenize(p) for p in country_filled_in_prompts[country][geo_class]]).cuda()
							avg_cls_embed = torch.mean(clip_model_.encode_text(input_ex), dim=0)
							norm_avg_cls_embed = avg_cls_embed / avg_cls_embed.norm()
							norm_avgd_class_embed_list.append(norm_avg_cls_embed)
						ensembled_general_descriptor_classifier = torch.stack(norm_avgd_class_embed_list) 
						country_classifier_list.append(ensembled_general_descriptor_classifier)
					overall_country_classifiers = torch.stack(country_classifier_list)
					self.country_ensemble_llm_features = torch.mean(overall_country_classifiers, dim=0)
					for i in range(self.country_ensemble_llm_features.shape[0]):
						self.country_ensemble_llm_features[i,:] /= self.country_ensemble_llm_features[i,:].norm()

			# This is not supported right now for ensemble 
			if cfg.PROMPT.USE_AVG_LLM_AND_COUNTRY_NAME and cfg.PROMPT.USE_COUNTRY_NAME:
				norm_avgd_class_embed_list = []
				for geo_class_ind in range(self.text_features.shape[0]):
					avg_cls_embed = torch.mean(torch.stack([self.text_features[geo_class_ind, :], self.country_of_interest_llm_features[geo_class_ind, :]]), dim=0)
					norm_avg_cls_embed = avg_cls_embed / avg_cls_embed.norm()
					norm_avgd_class_embed_list.append(norm_avg_cls_embed)
				ensembled_general_descriptor_classifier = torch.stack(norm_avgd_class_embed_list)
				self.avg_in_country_and_llm_country_features = ensembled_general_descriptor_classifier

		# Back to other processing
		tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
		with torch.no_grad():
			embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
		# These token vectors will be saved when in save_model(),
		# but they should be ignored in load_model() as we want to use
		# those computed using the current class names
		self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
		self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

		self.meta_net = nn.Sequential(OrderedDict([
			("linear1", nn.Linear(512, 512)),
			("relu", nn.ReLU(inplace=True))
			#("linear2", nn.Linear(128, 512))
		]))
		if cfg.TRAINER.COCOOP.PREC == "fp16":
			self.meta_net.half()
		self.n_cls = n_cls
		self.n_ctx = n_ctx
		self.tokenized_prompts = tokenized_prompts  # torch.Tensor
		self.name_lens = name_lens
		self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

	def forward(self):
		ctx = self.ctx

		if ctx.dim() == 2:
			ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
		
		prefix = self.token_prefix
		suffix = self.token_suffix

		prompts = torch.cat(
			[
				prefix,  # (n_cls, 1, dim)
				ctx,
				suffix,  # (n_cls, *, dim)
			],
			dim=1,
		)

		return prompts


class Adapter(nn.Module):
	def __init__(self, c_in, reduction=4):
		super(Adapter, self).__init__()
		self.fc = nn.Sequential(
			nn.Linear(c_in, c_in // reduction, bias=False),
			nn.ReLU(inplace=True),
			nn.Linear(c_in // reduction, c_in, bias=False),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		x = self.fc(x)
		return x

class CustomCLIP(nn.Module):
	def __init__(self, cfg, classnames, clip_model):
		super().__init__()
		self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
		self.tokenized_prompts = self.prompt_learner.tokenized_prompts
		self.ori_embedding = self.prompt_learner.text_features
		self.image_encoder = clip_model.visual
		self.text_encoder = TextEncoder(clip_model)
		self.logit_scale = clip_model.logit_scale
		self.dtype = clip_model.dtype
		self.meta_net = self.prompt_learner.meta_net
		self.adapter = Adapter(512, 4).to(clip_model.dtype)

		# Which country classifiers to load 
		if cfg.PROMPT.USE_LLM_DESC:                                                    
			self.country_of_interest_llm_features = self.prompt_learner.country_of_interest_llm_features
		if cfg.PROMPT.USE_AVG_LLM_AND_COUNTRY_NAME and cfg.PROMPT.USE_LLM_DESC and cfg.PROMPT.USE_COUNTRY_NAME:
			self.avg_in_country_and_llm_country_features = self.prompt_learner.avg_in_country_and_llm_country_features
		if cfg.PROMPT.USE_COUNTRY_ENSEMBLE:
			if cfg.PROMPT.USE_COUNTRY_NAME:
				self.country_ensemble_default_features = self.prompt_learner.country_ensemble_default_features
			if cfg.PROMPT.USE_LLM_DESC:
				self.country_ensemble_llm_features = self.prompt_learner.country_ensemble_llm_features
		# Use arguments
		self.use_llm = cfg.PROMPT.USE_LLM_DESC
		self.use_avg_llm_and_country_name = cfg.PROMPT.USE_AVG_LLM_AND_COUNTRY_NAME
		self.use_country_name = cfg.PROMPT.USE_COUNTRY_NAME
		self.use_country_ensemble = cfg.PROMPT.USE_COUNTRY_ENSEMBLE

	# Key forward function 
	def forward(self, image):
		prompts = self.prompt_learner()
		image_features = self.image_encoder(image.type(self.dtype))
		tokenized_prompts = self.tokenized_prompts
		text_features = self.text_encoder(prompts, tokenized_prompts) 
		text_features_old = self.ori_embedding
		

		image_features = image_features / image_features.norm(dim=-1, keepdim=True)
		text_features = text_features / text_features.norm(dim=-1, keepdim=True)
		logit_scale = self.logit_scale.exp()
		logits = logit_scale * image_features @ text_features.t()

		cos = torch.nn.CosineSimilarity(dim=1,eps=1e-07)
		text_features_old = text_features_old / text_features_old.norm(dim=-1, keepdim=True)
		score = cos(text_features,text_features_old)

		score = 1.0-torch.mean(score)

		if self.use_country_ensemble and not self.use_llm:
			print('Default Country Ensemble')
			ensemble_country_features_old = self.country_ensemble_default_features
			ensemble_country_features_old = ensemble_country_features_old / ensemble_country_features_old.norm(dim=-1, keepdim=True)
			score_avg_ensemble_country = cos(text_features, ensemble_country_features_old)
			score_avg_ensemble_country = 1.0 - torch.mean(score_avg_ensemble_country)

		if self.use_llm and self.use_country_name and self.use_avg_llm_and_country_name:
			print('Avg(LLM Country, Default Country)')
			avg_in_country_and_llm_country_features_old = self.avg_in_country_and_llm_country_features
			avg_in_country_and_llm_country_features_old = avg_in_country_and_llm_country_features_old / avg_in_country_and_llm_country_features_old.norm(dim=-1, keepdim=True)
			score_avg_in_c_and_llm_c = cos(text_features, avg_in_country_and_llm_country_features_old)
			score_avg_in_c_and_llm_c = 1.0 - torch.mean(score_avg_in_c_and_llm_c)

		if self.use_llm:
			print("Using in country name: " + str(self.use_country_name))
			if self.use_country_ensemble:
				print("LLM Country Ensemble")
				avg_llm_country_old = self.country_ensemble_llm_features
				avg_llm_country_old = avg_llm_country_old / avg_llm_country_old.norm(dim=-1, keepdim=True)
				score_llm = cos(text_features, avg_llm_country_old)
				score_llm = 1.0 - torch.mean(score_llm)
			else:
				print("Country of Interest LLM")
				country_features_old = self.country_of_interest_llm_features
				country_features_old = country_features_old / country_features_old.norm(dim=-1, keepdim=True)
				score_llm = cos(text_features,country_features_old)
				score_llm = 1.0-torch.mean(score_llm)

		# LLM-only+incountry and average
		if self.use_country_ensemble and not self.use_llm:
			return logits, score_avg_ensemble_country
		elif self.use_llm and self.use_country_name and self.use_avg_llm_and_country_name:
			return logits, score_avg_in_c_and_llm_c
		# LLM-only case and LLM-only+incountry
		elif self.use_llm:
			print(score_llm)
			return logits, score_llm
		# Cases where this is called: default
		else:
			return logits, score


@TRAINER_REGISTRY.register()
class GeoKnowledgePrompting(TrainerX):

	def check_cfg(self, cfg):
		assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

	def build_model(self):
		cfg = self.cfg
		classnames = self.dm.dataset.classnames

		print('Length of classnames')
		print(len(classnames))

		print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
		clip_model = load_clip_to_cpu(cfg)
		
		if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
			# CLIP's default precision is fp16
			clip_model.float()

		print("Building custom CLIP")
		self.model = CustomCLIP(cfg, classnames, clip_model)


		self.w = cfg.TRAINER.COOP.W

		print("Turning off gradients in both the image and the text encoder")
		for name, param in self.model.named_parameters():
			#if "prompt_learner" not in name: # and "adapter" not in name:
			if "ctx" not in name: 
				param.requires_grad_(False)
			else:
				print(name)

		if cfg.MODEL.INIT_WEIGHTS:
			load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

		self.model.to(self.device)
		# NOTE: only give prompt_learner to the optimizer
		self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
		self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
		self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
		
		#self.optim_ = build_optimizer(self.model.adapter, cfg.OPTIM)
		#self.sched_ = build_lr_scheduler(self.optim, cfg.OPTIM)
		#self.register_model('clip_adapter', self.model.adapter, self.optim_, self.sched_)

		self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

		# Note that multi-gpu training could be slow because CLIP's size is
		# big, which slows down the copy operation in DataParallel
		device_count = torch.cuda.device_count()
		if device_count > 1:
			print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
			self.model = nn.DataParallel(self.model)

	def forward_backward(self, batch):
		image, label = self.parse_batch_train(batch)
		prec = self.cfg.TRAINER.COOP.PREC
		if prec == "amp":
			with autocast():
				output = self.model(image)
				loss = F.cross_entropy(output, label)
			self.optim.zero_grad()
			self.scaler.scale(loss).backward()
			self.scaler.step(self.optim)
			self.scaler.update()
		else:
			output, score = self.model(image)
			loss = F.cross_entropy(output, label) + self.w*score

			# Print which mode
			# This is used to ensure model training correctly 
			if self.model.use_country_ensemble and not self.model.use_llm:
				print('Using Default Country Ensemble')
				print(self.w)
			elif self.model.use_llm and self.model.use_country_name and self.model.use_avg_llm_and_country_name:
				print('Using Avg(LLM, inCountry)')
				print(self.w)
			elif self.model.use_llm:
				print("Using LLM")
				if self.model.use_country_ensemble == "ensemble":
					print("And ensemble")
				print(self.w)
			else:
				print("Default")
				print(self.w)
			
			print(loss)
			self.model_backward_and_update(loss)

		loss_summary = {
			"loss": loss.item(),
			"acc": compute_accuracy(output, label)[0].item(),
		}

		if (self.batch_idx + 1) == self.num_batches:
			#self.update_lr()
			self.sched.step()
			#self.sched_.step()
		return loss_summary

	def parse_batch_train(self, batch):
		input = batch["img"]
		label = batch["label"]
		input = input.to(self.device)
		label = label.to(self.device)
		return input, label

	def model_inference(self, input):
		#print('Model Inference')
		#print(input.shape)
		return self.model(input)[0]

	def load_model(self, directory, epoch=None):
		if not directory:
			print("Note that load_model() is skipped as no pretrained model is given")
			return
   
		names = self.get_model_names()

		# By default, the best model is loaded
		model_file = "model-best.pth.tar"

		if epoch is not None:
			model_file = "model.pth.tar-" + str(epoch)

		for name in names:
			model_path = osp.join(directory, name, model_file)

			if not osp.exists(model_path):
				raise FileNotFoundError('Model not found at "{}"'.format(model_path))

			checkpoint = load_checkpoint(model_path)
			state_dict = checkpoint["state_dict"]
			epoch = checkpoint["epoch"]

			# Ignore fixed token vectors
			if "token_prefix" in state_dict:
				del state_dict["token_prefix"]

			if "token_suffix" in state_dict:
				del state_dict["token_suffix"]

			if "token_midfix" in state_dict:
				del state_dict["token_midfix"]

			print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
			# set strict=False
			self._models[name].load_state_dict(state_dict, strict=False)