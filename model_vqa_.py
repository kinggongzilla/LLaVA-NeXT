



import argparse
import json
import os
import os.path as osp
import math
import torch
from PIL import Image
from tqdm import tqdm


from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from PIL import Image
import requests
import copy
import torch

import sys
import warnings

warnings.filterwarnings("ignore")
# pretrained = "lmms-lab/llava-onevision-qwen2-7b-ov"
# pretrained = "/system/user/publicdata/llm/Llava_weights/llava-onevision-qwen2-7b-ov"
model_name = "llava_llama"
device = "cuda"
device_map = "auto"
conv_template = "llava_llama_3"  # Make sure you use correct chat template for different models
def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def eval_model(args):
    torch.set_grad_enabled(False)
    if args.question_file.endswith("jsonl"):
        questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
        questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
        answers_file = os.path.expanduser(args.answers_file)
        os.makedirs(os.path.dirname(answers_file), exist_ok=True)
        ans_file = open(answers_file, "w")
        if args.load_8bit:
            print("Loading 8-bit model")
        elif args.load_4bit:
            print("Loading 4-bit model")
        tokenizer, model, image_processor, max_length = load_pretrained_model(args.model_path, None, model_name,
                                                                              device_map=device_map, load_8bit=args.load_8bit, load_4bit=args.load_4bit)
        model.eval()


        for data_item in tqdm(questions):
            image_list = data_item['image']
            image_list = [osp.join(args.image_folder, img) for img in image_list]
            images = [Image.open(img).convert('RGB') for img in image_list]
            image_tensor = process_images(images, image_processor, model.config)
            if isinstance(image_tensor, list):
                image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]
            else:
                image_tensor = image_tensor.to(dtype=torch.float16, device=device)

            question = data_item['text']

            conv = copy.deepcopy(conv_templates[conv_template])
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt_question = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX,
                                              return_tensors="pt").unsqueeze(0).to(device)
            image_sizes = [image.size for image in images]

            cont = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                do_sample=False,
                temperature=0,
                max_new_tokens=4096,
            )

            text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
            final_answer = text_outputs[0]
            print(final_answer)
            ans_file.write(json.dumps(
                {"question_id": data_item['question_id'],
                 'image': data_item['image'],
                 'prompt': data_item['text'],
                 "text": final_answer
                 }) + "\n")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/home/david/JKU/master/thesis/llama3-llava-next-8b")
    parser.add_argument("--question-file", type=str, default="/home/david/JKU/master/thesis/LLaVA-NeXT/eval/llava-seed-bench_onemissing.jsonl")
    parser.add_argument("--answers-file", type=str, default="/home/david/JKU/master/thesis/LLaVA-NeXT/eval/llava-seed-bench_onemissing_model-answers.jsonl")
    parser.add_argument("--image-folder", type=str, default="/home/david/JKU/master/thesis/LLaVA-NeXT/eval/images")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    # parser.add_argument("--max-new-tokens", type=int, default=128)

    args = parser.parse_args()
    eval_model(args)