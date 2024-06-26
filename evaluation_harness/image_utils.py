from typing import List

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from transformers import (
    Blip2ForConditionalGeneration,
    Blip2Processor,
)
import torch
import sys
sys.path.append('/mnt/nvme0n1p1/hongxin_li/agent-eval/')
from lmms_eval.models.model_utils.qwen.qwen_generate_utils import make_context
from transformers import AutoModelForCausalLM, AutoTokenizer
import uuid
import os

def construct_multiple_choice_string(points):
    multiple_choice_string = ""
    
    # Generate labels from 'A' onwards
    labels = [chr(i) for i in range(ord('A'), ord('A') + len(points) + 1)]
    
    for label, point in zip(labels, points):
        multiple_choice_string += f"\n{label}. <ref>{point}"
    
    # Add the last label for "None of the above"
    last_label = labels[len(points)]
    multiple_choice_string += f"\n{last_label}. None of the above.\n"
    
    return multiple_choice_string

def get_captioning_fn(
    device, dtype, model_name: str = "Salesforce/blip2-flan-t5-xl"
) -> callable:
    if "blip2" in model_name:
        captioning_processor = Blip2Processor.from_pretrained(model_name)
        captioning_model = Blip2ForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=dtype
        )
        captioning_model.to(device)
        def caption_images(
            images: List[Image.Image],
            prompt: List[str] = None,
            max_new_tokens: int = 32,
        ) -> List[str]:
            if prompt is None:
                # Perform VQA
                inputs = captioning_processor(
                    images=images, return_tensors="pt"
                ).to(device, dtype)
                generated_ids = captioning_model.generate(
                    **inputs, max_new_tokens=max_new_tokens
                )
                captions = captioning_processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )
            else:
                # Regular captioning. Prompt is a list of strings, one for each image
                assert len(images) == len(
                    prompt
                ), "Number of images and prompts must match, got {} and {}".format(
                    len(images), len(prompt)
                )
                inputs = captioning_processor(
                    images=images, text=prompt, return_tensors="pt"
                ).to(device, dtype)
                generated_ids = captioning_model.generate(
                    **inputs, max_new_tokens=max_new_tokens
                )
                captions = captioning_processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )

            return captions

    elif 'funcpred' in model_name:
        captioning_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, trust_remote_code=True).eval() # load_in_4bit=True
        # captioning_model_gnd = AutoModelForCausalLM.from_pretrained("/mnt/nvme0n1p1/hongxin_li/seeclick_exp/checkpoints/seeclick_scaling_funcpred625k_llava150k_cauldron197k_refGnd_resample_v2", torch_dtype=dtype, trust_remote_code=True).eval() # load_in_4bit=True
        tokenizer = AutoTokenizer.from_pretrained("/mnt/nvme0n1p1/hongxin_li/hf_home/hub/models--Qwen--Qwen-VL-Chat/snapshots/f57cfbd358cb56b710d963669ad1bcfb44cdcdd8",  # "/mnt/nvme0n1p1/hongxin_li/hf_home/hub/models--Qwen--Qwen-VL-Chat/snapshots/f57cfbd358cb56b710d963669ad1bcfb44cdcdd8"
                                                  trust_remote_code=True,
                                                  multidigit=True,
                                                  )
        captioning_model.to(device)
        # captioning_model_gnd.to(device)
        def caption_images(
            images,
            prompt,
            multiple_choice = None,
            max_new_tokens: int = 32,
        ) -> List[str]:
            query = []
            visual_paths = []
            for visual in images:
                name = uuid.uuid4().hex.upper()[0:6]
                visual.save(f"/tmp/{name}.png")
                visual_paths.append(f"/tmp/{name}.png")
            if multiple_choice is None:
                prompt = f"In this web page image, please locate the element based on the \"{prompt}\" (with point)."
                model = captioning_model_gnd
            else:
                mc = f"Please select the appropriate answer from the following options, choose \"None of the above\" if none apply:{construct_multiple_choice_string(multiple_choice)}"
                prompt = f"In this UI design, to process the instruction \"{prompt}\", where should I activate ? {mc}"
                model = captioning_model
            query.append({"image": visual_paths[0]})
            query.append({"text": prompt})
            questions = tokenizer.from_list_format(query)
            # https://huggingface.co/cckevinn/SeeClick/blob/main/generation_config.json
            gen_kwargs = {}
            gen_kwargs["max_new_tokens"] = max_new_tokens
            gen_kwargs["temperature"] = 0.5
            gen_kwargs["top_p"] = None
            gen_kwargs["num_beams"] = 1
            gen_kwargs["chat_format"] = "chatml"
            gen_kwargs["do_sample"] = True
            gen_kwargs["eos_token_id"] = 151643
            gen_kwargs["max_window_size"] = 1024
            gen_kwargs["pad_token_id"] = 151643
            gen_kwargs["top_k"] = 0
            gen_kwargs["transformers_version"] = "4.36.2"
            text_output, history = model.chat(tokenizer, 
                                            query=questions, 
                                            history=None,
                                            **gen_kwargs)
            for visual_path in visual_paths:
                try:
                    os.remove(visual_path)
                except:
                    pass
            return text_output
        
        def logits(
            images,
            prompt,
            multiple_choice,
            max_new_tokens: int = 32,
        ) -> List[str]:
            query = []
            visual_paths = []
            for visual in images:
                name = uuid.uuid4().hex.upper()[0:6]
                visual.save(f"/tmp/{name}.png")
                visual_paths.append(f"/tmp/{name}.png")
            prompt = f"In this web page image, please locate the element based on the \"{prompt}\" (with point)."
            query.append({"image": visual_paths[0]})
            query.append({"text": prompt})
            # split the choice into a list of strings
            choice_prob = []
            for choice in multiple_choice:
                # split the choice into a list of strings
                # for i in range(len(continuations)):
                #     if continuations[i] == ',':
                #         continue
                    # cur_cont = ''.join(continuations[:i])
                context_query = [
                    {"image": visual_paths[0]},
                    {"text": prompt}
                ]
                answer = [
                    {"image": visual_paths[0]},
                    {"text": choice}
                ]
                context_query = tokenizer.from_list_format(context_query)
                # answer = tokenizer.from_list_format(answer)
                raw_contxt_text, context_tokens = make_context(
                tokenizer, context_query, history=None, system="You are a helpful assistant", max_window_size=captioning_model.generation_config.max_window_size, chat_format=captioning_model.generation_config.chat_format
                )
                context_tokens = torch.tensor([context_tokens]).to(captioning_model.device)
                
                answer_tokens = tokenizer(answer[1]['text'])['input_ids']
                answer_tokens = torch.tensor([answer_tokens]).to(captioning_model.device)
                def get_stop_words_ids(chat_format, tokenizer):
                    if chat_format == "raw":
                        stop_words_ids = [tokenizer.encode("Human:"), [tokenizer.eod_id]]
                    elif chat_format == "chatml":
                        stop_words_ids = [[tokenizer.im_end_id], [tokenizer.im_start_id]]
                    else:
                        raise NotImplementedError(f"Unknown chat format {chat_format!r}")
                    return stop_words_ids
                stop_words_ids = []
                stop_words_ids.extend(get_stop_words_ids(
                    captioning_model.generation_config.chat_format, tokenizer
                ))
                stop_words_ids = torch.tensor([[st[0] for st in stop_words_ids]]).to(captioning_model.device)
                continues_tokens = torch.concat([context_tokens, answer_tokens, stop_words_ids], dim=-1)
                attn_mask = torch.ones_like(continues_tokens).to(captioning_model.device)
                
                labels = continues_tokens.clone().to(captioning_model.device)
                labels[:, : context_tokens.shape[1]] = -100
                
                with torch.inference_mode():
                    outputs = captioning_model(input_ids=continues_tokens, labels=labels, attention_mask=attn_mask)
                loss = outputs.loss
                logits = outputs["logits"] # 预测下一个token
                probs = logits.softmax(dim=-1).detach()
                # greedy_tokens = logits.argmax(dim=-1)
                gen_probs = torch.gather(probs, 2, continues_tokens[:, :, None]).squeeze(-1)
                answer_logits = logits[:, context_tokens.shape[1] : continues_tokens.shape[1]]  # [1, seq]
                text_sequence = []
                for i in range(context_tokens.shape[1], continues_tokens.shape[1]):
                    next_tokens_prob = probs[0, i - 1, :]
                    next_token_id = continues_tokens[0, i]
                    if next_token_id in [151857, 151858,151859,151644,151645]:
                        continue
                    text_sequence.append((tokenizer.decode(next_token_id), next_tokens_prob[next_token_id].item()))
                # for token, p in zip(continues_tokens[0, context_tokens.shape[1]-1:-1], gen_probs[0, context_tokens.shape[1]:]):
                #     if token in [151857, 151858,151859,151644,151645]: # 7: '(', 8: ')', 11: ',' 
                #         continue
                #     text_sequence.append((tokenizer.decode(token),p.item()))
                prod = 0
                for i in text_sequence:
                    prod += i[1] 
                choice_prob.append(prod)
            for visual_path in visual_paths:
                try:
                    os.remove(visual_path)
                except:
                    pass
            index = choice_prob.index(max(choice_prob))
            return choice_prob # multiple_choice[index]

    else:
        raise NotImplementedError(
            "Only BLIP-2 models are currently supported"
        )

    return logits


def get_image_ssim(imageA, imageB):
    # Determine the size to which we should resize
    new_size = max(imageA.size[0], imageB.size[0]), max(
        imageA.size[1], imageB.size[1]
    )

    # Resize images
    imageA = imageA.resize(new_size, Image.LANCZOS)
    imageB = imageB.resize(new_size, Image.LANCZOS)

    # Convert images to grayscale
    grayA = imageA.convert("L")
    grayB = imageB.convert("L")

    # Convert grayscale images to numpy arrays for SSIM computation
    grayA = np.array(grayA)
    grayB = np.array(grayB)

    # Compute the Structural Similarity Index (SSIM) between the two images
    score, _ = ssim(grayA, grayB, full=True)
    return score
