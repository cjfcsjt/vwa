import argparse
import json
from typing import Any, Optional

import tiktoken
from beartype import beartype
from PIL import Image

from agent.prompts import *
from browser_env import Trajectory
from browser_env.actions import (
    Action,
    ActionParsingError,
    create_id_based_action,
    create_none_action,
    create_playwright_action,
)
from browser_env.utils import Observation, StateInfo, find_nearest_points
from llms import (
    call_llm,
    generate_from_huggingface_completion,
    generate_from_openai_chat_completion,
    generate_from_openai_completion,
    lm_config,
)
from llms.tokenizers import Tokenizer
import os 
from PIL import Image, ImageDraw

class Agent:
    """Base class for the agent"""

    def __init__(self, *args: Any) -> None:
        pass

    def next_action(
        self, trajectory: Trajectory, intent: str, meta_data: Any
    ) -> Action:
        """Predict the next action given the observation"""
        raise NotImplementedError

    def reset(
        self,
        test_config_file: str,
    ) -> None:
        raise NotImplementedError


class TeacherForcingAgent(Agent):
    """Agent that follows a pre-defined action sequence"""

    def __init__(self) -> None:
        super().__init__()

    def set_action_set_tag(self, tag: str) -> None:
        self.action_set_tag = tag

    def set_actions(self, action_seq: str | list[str]) -> None:
        if isinstance(action_seq, str):
            action_strs = action_seq.strip().split("\n")
        else:
            action_strs = action_seq
        action_strs = [a.strip() for a in action_strs]

        actions = []
        for a_str in action_strs:
            try:
                if self.action_set_tag == "playwright":
                    cur_action = create_playwright_action(a_str)
                elif self.action_set_tag == "id_accessibility_tree":
                    cur_action = create_id_based_action(a_str)
                else:
                    raise ValueError(
                        f"Unknown action type {self.action_set_tag}"
                    )
            except ActionParsingError as e:
                cur_action = create_none_action()

            cur_action["raw_prediction"] = a_str
            actions.append(cur_action)

        self.actions: list[Action] = actions

    def next_action(
        self, trajectory: Trajectory, intent: str, meta_data: Any
    ) -> Action:
        """Predict the next action given the observation"""
        return self.actions.pop(0)

    def reset(
        self,
        test_config_file: str,
    ) -> None:
        with open(test_config_file) as f:
            ref_actions = json.load(f)["reference_action_sequence"]
            tag = ref_actions["action_set_tag"]
            action_seq = ref_actions["action_sequence"]
            self.set_action_set_tag(tag)
            self.set_actions(action_seq)


class PromptAgent(Agent):
    """prompt-based agent that emits action given the history"""

    @beartype
    def __init__(
        self,
        action_set_tag: str,
        lm_config: lm_config.LMConfig,
        prompt_constructor: PromptConstructor,
        captioning_fn = None,
    ) -> None:
        super().__init__()
        self.lm_config = lm_config
        self.prompt_constructor = prompt_constructor
        self.action_set_tag = action_set_tag
        self.captioning_fn = captioning_fn

        # Check if the model is multimodal.
        if ("gemini" in lm_config.model or "gpt-4" in lm_config.model and "vision" in lm_config.model or '4o' in lm_config.model) and type(prompt_constructor) == MultimodalCoTPromptConstructor:
            self.multimodal_inputs = True
        else:
            self.multimodal_inputs = False

    def set_action_set_tag(self, tag: str) -> None:
        self.action_set_tag = tag

    @beartype
    def next_action(
        self, trajectory: Trajectory, intent: str, meta_data: dict[str, Any], images: Optional[list[Image.Image]] = None,
        output_response: bool = False, logger = None
    ) -> Action:
        # Create page screenshot image for multimodal models.
        if self.multimodal_inputs:
            page_screenshot_arr = trajectory[-1]["observation"]["image"]
            page_screenshot_arr_ori = trajectory[-1]["observation"]["screenshot"]
            page_screenshot_img = Image.fromarray(
                page_screenshot_arr
            )  # size = (viewport_width, viewport_width)
            page_screenshot_img_ori = Image.fromarray(
                page_screenshot_arr_ori
            )

        # Caption the input image, if provided.
        if images is not None and len(images) > 0:
            if self.captioning_fn is not None:
                image_input_caption = ""
                for image_i, image in enumerate(images):
                    if image_i == 0:
                        image_input_caption += f'Input image {image_i+1}: "{self.captioning_fn([image])[0]}"'
                    else:
                        image_input_caption += f'input image {image_i+1}: "{self.captioning_fn([image])[0]}"'
                    if len(images) > 1:
                        image_input_caption += ", "
                # Update intent to include captions of input images.
                intent = f"{image_input_caption}\nIntent: {intent}"
            elif not self.multimodal_inputs:
                print(
                    "WARNING: Input image provided but no image captioner available."
                )

        if self.multimodal_inputs:
            width, height = page_screenshot_img.size
            # gnd caption prior
            gnd_prompt = self.prompt_constructor.construct_gnd(
                trajectory, intent, page_screenshot_img, images, meta_data
            )
            subtask_response = call_llm(self.lm_config, gnd_prompt)
            mc, mc_str, ele_id, points_oriscale= [], [], [],[]
            i = 0
            for x_c, y_c, w, h in list(trajectory[-1]['info']['observation_metadata']['image']['obs_nodes_info'].values()):
                i+=1
                point = [int(x_c / width * 100), int(y_c / height * 100)]
                points_oriscale.append([x_c, y_c])
                mc.append(point)
                mc_str.append(f"({point[0]},{point[1]})")
                ele_id.append(i)
            
            if self.captioning_fn.__name__ == 'logits':
                choice_prob = self.captioning_fn([page_screenshot_img_ori], subtask_response, mc_str)
                index = choice_prob.index(max(choice_prob))
                gnd_response = mc_str[index]
                point = list(eval(gnd_response))
                point_oriscale = points_oriscale[index]
            else:
                gnd_response = self.captioning_fn([page_screenshot_img_ori], subtask_response)
                seed_point = list(eval(gnd_response))
                near_point = find_nearest_points(seed_point, mc, num_nearest=8)
                near_point_str = [f"({np[0]},{np[1]})" for np in near_point]
                choice_response = self.captioning_fn([page_screenshot_img_ori], subtask_response, near_point_str)
                alphatoint = {'A': 0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'H':7 }
                gnd_response = near_point_str[alphatoint[choice_response]]
                point = list(eval(gnd_response))
                meta_data["seed_point"] = (seed_point[0] / 100 * width, seed_point[1] / 100 * height)
            
            meta_data["gnd_response"] = (point[0] / 100 * width, point[1] / 100 * height)
            meta_data["id_response"] = index+1
            
            prompt = self.prompt_constructor.construct(
                trajectory, intent, page_screenshot_img, images, meta_data
            )
        else:
            prompt = self.prompt_constructor.construct(
                trajectory, intent, meta_data
            )
        lm_config = self.lm_config
        n = 0
        while True:
            response = call_llm(lm_config, prompt)
            force_prefix = self.prompt_constructor.instruction[
                "meta_data"
            ].get("force_prefix", "")
            response = f"{force_prefix}{response}"
            if output_response:
                draw = ImageDraw.Draw(page_screenshot_img)
                # Define the radius of the point to draw (adjust as necessary)
                radius = 10

                # Draw the point (circle) on the image
                if 'seed_point' in meta_data:
                    draw.ellipse((meta_data["seed_point"][0] - radius, meta_data["seed_point"][1] - radius,
                                meta_data["seed_point"][0] + radius, meta_data["seed_point"][1] + radius), fill='red', outline='red')
                draw.ellipse((meta_data["gnd_response"][0] - radius, meta_data["gnd_response"][1] - radius,
                            meta_data["gnd_response"][0] + radius, meta_data["gnd_response"][1] + radius), fill='blue', outline='blue')
                draw.ellipse([(point_oriscale[0] - radius, point_oriscale[1]- radius), (point_oriscale[0] + radius, point_oriscale[1]+radius)], fill='green', outline = 'green')
                # Save the image with a new name or display it
                new_image_path = os.path.join('results_prob_multidigit', f'point_screenshot.jpg')
                page_screenshot_img.save(new_image_path)
                logger.info(f"target element id: {prompt[-1]['content'][0]['text'].split('TARGET ELEMENTS ID:')[-1]}")
                logger.info(f'Agent: {response}')
            n += 1
            try:
                parsed_response = self.prompt_constructor.extract_action(
                    response
                )
                if self.action_set_tag == "id_accessibility_tree":
                    action = create_id_based_action(parsed_response)
                elif self.action_set_tag == "playwright":
                    action = create_playwright_action(parsed_response)
                elif self.action_set_tag == "som":
                    action = create_id_based_action(parsed_response)
                else:
                    raise ValueError(
                        f"Unknown action type {self.action_set_tag}"
                    )
                action["raw_prediction"] = response
                action["screenshot_point"] = page_screenshot_img
                action["subtask"] = subtask_response
                if 'choice_prob' in action:
                    action['choice_prob'] = choice_prob
                action['ele_id'] = ele_id
                break
            except ActionParsingError as e:
                if n >= lm_config.gen_config["max_retry"]:
                    action = create_none_action()
                    action["raw_prediction"] = response
                    action["screenshot_point"] = page_screenshot_img
                    action["subtask"] = subtask_response
                    if 'choice_prob' in action:
                        action['choice_prob'] = choice_prob
                    action['ele_id'] = ele_id
                    break

        return action

    def reset(self, test_config_file: str) -> None:
        pass


def construct_agent(args: argparse.Namespace, captioning_fn=None) -> Agent:
    llm_config = lm_config.construct_llm_config(args)

    agent: Agent
    if args.agent_type == "teacher_forcing":
        agent = TeacherForcingAgent()
    elif args.agent_type == "prompt":
        with open(args.instruction_path) as f:
            constructor_type = json.load(f)["meta_data"]["prompt_constructor"]
        tokenizer = Tokenizer(args.provider, args.model)
        prompt_constructor = eval(constructor_type)(
            args.instruction_path, lm_config=llm_config, tokenizer=tokenizer
        )
        agent = PromptAgent(
            action_set_tag=args.action_set_tag,
            lm_config=llm_config,
            prompt_constructor=prompt_constructor,
            captioning_fn=captioning_fn
        )
    else:
        raise NotImplementedError(
            f"agent type {args.agent_type} not implemented"
        )
    return agent
