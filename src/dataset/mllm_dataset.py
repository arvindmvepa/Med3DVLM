import os
import json
import torch
import random
import numpy as np
import monai.transforms as mtf
import pandas as pd
from monai.data import set_track_meta
from torch.utils.data import Dataset, ConcatDataset
from src.dataset.prompt_templates import Caption_templates


class CapDataset(Dataset):
    def __init__(self, args, tokenizer, mode="train", test_size=1000):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode

        self.image_tokens = "<im_patch>" * args.proj_out_num

        with open(args.cap_data_path, "r") as file:
            self.json_file = json.load(file)
        self.data_list = self.json_file[mode]

        self.caption_prompts = Caption_templates

        train_transform = mtf.Compose(
            [
                mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),
                mtf.RandFlip(prob=0.10, spatial_axis=0),
                mtf.RandFlip(prob=0.10, spatial_axis=1),
                mtf.RandFlip(prob=0.10, spatial_axis=2),
                mtf.RandScaleIntensity(factors=0.1, prob=0.5),
                mtf.RandShiftIntensity(offsets=0.1, prob=0.5),
                mtf.ToTensor(dtype=torch.float),
            ]
        )

        val_transform = mtf.Compose(
            [
                mtf.ToTensor(dtype=torch.float),
            ]
        )
        set_track_meta(False)

        if mode == "train":
            self.transform = train_transform
        elif mode == "validation":
            self.transform = val_transform
        elif "test" in mode:
            self.transform = val_transform
            self.data_list = self.data_list[:test_size]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                data = self.data_list[idx]
                image_path = data["image"]
                image_abs_path = os.path.join(self.data_root, image_path)

                image = np.load(image_abs_path)  # nomalized 0-1, C,D,H,W
                # image = np.load(img_abs_path)[np.newaxis, ...]  # nomalized
                image = self.transform(image)

                text_path = data["text"]
                text_abs_path = os.path.join(self.data_root, text_path)
                with open(text_abs_path, "r") as text_file:
                    raw_text = text_file.read()
                answer = raw_text

                prompt_question = random.choice(self.caption_prompts)

                question = self.image_tokens + prompt_question

                text_tensor = self.tokenizer(
                    question + " " + answer,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )

                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                valid_len = torch.sum(attention_mask)
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id

                question_tensor = self.tokenizer(
                    question,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )

                question_len = torch.sum(question_tensor["attention_mask"][0])

                label = input_id.clone()
                label[:question_len] = -100
                if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                    label[label == self.tokenizer.pad_token_id] = -100
                    if valid_len < len(label):
                        label[valid_len] = self.tokenizer.eos_token_id
                else:
                    label[label == self.tokenizer.pad_token_id] = -100

                ret = {
                    "image": image,
                    "input_id": input_id,
                    "label": label,
                    "attention_mask": attention_mask,
                    "question": question,
                    "answer": answer,
                    "question_type": "Caption",
                }
                # if self.args.seg_enable:
                #     ret.update({"seg": torch.zeros_like(image)})

                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)


class VQADataset(Dataset):
    def __init__(self, args, tokenizer, close_ended=True, mode="train"):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode
        self.close_ended = close_ended

        self.image_tokens = "<im_patch>" * args.proj_out_num

        if mode == "train":
            self.data_list = self.read_data_file(args.vqa_data_train_path)
        elif mode == "validation":
            self.data_list = self.read_data_file(args.vqa_data_val_path, nrows=2048)
        elif "test" in mode:
            self.data_list = self.read_data_file(args.vqa_data_test_path)
        else:
            print("The mode is not desired ! ")

        train_transform = mtf.Compose(
            [
                mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),
                mtf.RandFlip(prob=0.10, spatial_axis=0),
                mtf.RandFlip(prob=0.10, spatial_axis=1),
                mtf.RandFlip(prob=0.10, spatial_axis=2),
                mtf.RandScaleIntensity(factors=0.1, prob=0.5),
                mtf.RandShiftIntensity(offsets=0.1, prob=0.5),
                mtf.ToTensor(dtype=torch.float),
            ]
        )

        val_transform = mtf.Compose(
            [
                mtf.ToTensor(dtype=torch.float),
            ]
        )
        set_track_meta(False)

        if mode == "train":
            self.transform = train_transform
        elif mode == "validation":
            self.transform = val_transform
        elif "test" in mode:
            self.transform = val_transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                data = self.data_list.iloc[idx]
                image_abs_path = os.path.join(self.args.data_root, data["Image Path"])

                image = np.load(image_abs_path)  # nomalized, 0-1, C,D,H,W
                # image = np.load(img_path)[np.newaxis, ...]  # nomalized
                image = self.transform(image)

                if self.close_ended:
                    question = data["Question"]
                    choices = "Choices: A. {} B. {} C. {} D. {}".format(
                        data["Choice A"],
                        data["Choice B"],
                        data["Choice C"],
                        data["Choice D"],
                    )
                    question = question + " " + choices
                    answer = "{}. {}".format(data["Answer Choice"], data["Answer"])
                else:
                    question = data["Question"]
                    answer = str(data["Answer"])

                question = self.image_tokens + " " + question
                text_tensor = self.tokenizer(
                    question + " " + answer,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )

                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                valid_len = torch.sum(attention_mask)
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id

                question_tensor = self.tokenizer(
                    question,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                question_len = torch.sum(question_tensor["attention_mask"][0])

                label = input_id.clone()
                label[:question_len] = -100
                if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                    label[label == self.tokenizer.pad_token_id] = -100
                    if valid_len < len(label):
                        label[valid_len] = self.tokenizer.eos_token_id
                else:
                    label[label == self.tokenizer.pad_token_id] = -100

                ret = {
                    "image": image,
                    "input_id": input_id,
                    "label": label,
                    "attention_mask": attention_mask,
                    "question": question,
                    "answer": answer,
                    "answer_choice": data["Answer Choice"],
                    "question_type": data["Question Type"],
                }

                # if self.args.seg_enable:
                #     ret.update({"seg": torch.zeros_like(image)})

                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)
    
    def read_data_file(self, vqa_data_file, **kwargs):
        data_list = pd.read_csv(vqa_data_file, **kwargs)
        return data_list
    

class VQABratsDataset(VQADataset):

    def read_data_file(self, vqa_data_file, **kwargs):
        with open(vqa_data_file, 'r') as f:
            data_list = json.load(f)
        return data_list

    def __getitem__(self, idx):
        # TODO: Figure out how to use more modalities
        data = self.data_list[idx]

        image = []
        for modality in ["t1c", "t1n", "t2f", "t2w"]:
            image_abs_path = data["volume_non_seg_files"][modality]
            new_image_abs_path = self.convert_file_path_to_npy(image_abs_path)
            image_ = np.load(new_image_abs_path)
            image_ = self.transform(image_)
            image.append(image_)
        image = torch.cat(image, axis=1)  # Concatenate along depth dimension (axis=1)

        question = data["question"]
        answer = str(data["answer"])


        question = self.image_tokens + ' ' + question
        text_tensor = self.tokenizer(
            question + ' ' + answer, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt",
        )

        input_id = text_tensor["input_ids"][0]
        attention_mask = text_tensor["attention_mask"][0]

        valid_len = torch.sum(attention_mask)
        if valid_len < len(input_id):
            input_id[valid_len] = self.tokenizer.eos_token_id

        question_tensor = self.tokenizer(
            question, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt"
        )
        question_len = torch.sum(question_tensor["attention_mask"][0])

        label = input_id.clone()
        label[:question_len] = -100
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            label[label == self.tokenizer.pad_token_id] = -100
            if valid_len < len(label):
                label[valid_len] = self.tokenizer.eos_token_id
        else:
            label[label == self.tokenizer.pad_token_id] = -100

        ret = {
            'image': image,
            'input_id': input_id,
            'label': label,
            'attention_mask': attention_mask,
            'question': question,
            'answer': answer,
            'answer_choice': answer,
            'question_type': 'open_ended',
            'qid': data.get("qid", 0),
            'volume_file_id': data.get("volume_file_id", 0),
            'volume_file_dir': data.get("volume_file_dir", ""),
            'question_clean': data.get("question", ""),
            'label_name': data.get("label_name", ""),
            'q_lang': data.get("q_lang", ""),
            'content_type': data.get("content_type", ""),
            'study_name': data.get("study_name", ""),
        }
        return ret

    def convert_file_path_to_npy(self, image_abs_path):
        volume_abs_dir = os.path.dirname(image_abs_path)
        base_dir = os.path.dirname(volume_abs_dir)
        new_base_dir = base_dir + "_npy"

        volume_dir = os.path.basename(volume_abs_dir)
        image_file = os.path.basename(image_abs_path)
        new_image_abs_path = os.path.join(new_base_dir, volume_dir, image_file + ".npy")
        return new_image_abs_path

class VQAYNDataset(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode

        self.image_tokens = "<im_patch>" * args.proj_out_num

        if mode == "train":
            self.data_list = pd.read_csv(args.vqa_yn_data_train_path)
        elif mode == "validation":
            self.data_list = pd.read_csv(args.vqa_yn_data_val_path, nrows=2048)
        elif "test" in mode:
            self.data_list = pd.read_csv(args.vqa_yn_data_test_path)
        else:
            print("The mode is not desired ! ")

        train_transform = mtf.Compose(
            [
                mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),
                mtf.RandFlip(prob=0.10, spatial_axis=0),
                mtf.RandFlip(prob=0.10, spatial_axis=1),
                mtf.RandFlip(prob=0.10, spatial_axis=2),
                mtf.RandScaleIntensity(factors=0.1, prob=0.5),
                mtf.RandShiftIntensity(offsets=0.1, prob=0.5),
                mtf.ToTensor(dtype=torch.float),
            ]
        )

        val_transform = mtf.Compose(
            [
                mtf.ToTensor(dtype=torch.float),
            ]
        )
        set_track_meta(False)

        if mode == "train":
            self.transform = train_transform
        elif mode == "validation":
            self.transform = val_transform
        elif "test" in mode:
            self.transform = val_transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                data = self.data_list.iloc[idx]
                image_abs_path = os.path.join(self.args.data_root, data["Image Path"])

                image = np.load(image_abs_path)  # nomalized, 0-1, C,D,H,W
                # image = np.load(img_path)[np.newaxis, ...]  # nomalized
                image = self.transform(image)

                question = data["Question"]
                answer = str(data["Answer"])

                question = self.image_tokens + " " + question
                text_tensor = self.tokenizer(
                    question + " " + answer,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )

                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                valid_len = torch.sum(attention_mask)
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id

                question_tensor = self.tokenizer(
                    question,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                question_len = torch.sum(question_tensor["attention_mask"][0])

                label = input_id.clone()
                label[:question_len] = -100
                if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                    label[label == self.tokenizer.pad_token_id] = -100
                    if valid_len < len(label):
                        label[valid_len] = self.tokenizer.eos_token_id
                else:
                    label[label == self.tokenizer.pad_token_id] = -100

                ret = {
                    "image": image,
                    "input_id": input_id,
                    "label": label,
                    "attention_mask": attention_mask,
                    "question": question,
                    "answer": answer,
                    "answer_choice": data["Answer Choice"],
                    "question_type": data["Question Type"],
                }

                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)


class TextDatasets(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        super(TextDatasets, self).__init__()
        self.ds_list = [
            VQABratsDataset(args, tokenizer, mode)
            ]
        self.dataset = ConcatDataset(self.ds_list)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class TextYNDatasets(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        super(TextYNDatasets, self).__init__()
        self.ds_list = [
            VQABratsDataset(args, tokenizer, mode)
            ]
        self.dataset = ConcatDataset(self.ds_list)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
