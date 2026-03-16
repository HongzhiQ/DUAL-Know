import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from typing import List, Optional
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import (
    ANESGLM_MODEL_PATH, DEVICE, TORCH_DTYPE,
    LLM_TEMPERATURE, LLM_TOP_P, LLM_MAX_GEN_LENGTH,
)


class StopOnStrings(StoppingCriteria):


    def __init__(self, tokenizer, stop_strings: List[str], prompt_length: int):
        self.tokenizer = tokenizer
        self.stop_strings = stop_strings
        self.prompt_length = prompt_length

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:

        generated_ids = input_ids[0][self.prompt_length:]
        if len(generated_ids) == 0:
            return False

        check_ids = generated_ids[-20:]
        text = self.tokenizer.decode(check_ids, skip_special_tokens=True)

        for s in self.stop_strings:
            if s in text:
                return True
        return False


class AnesGLM:
    def __init__(self, model_path: str = None, device: str = None):
        self.model_path = model_path or ANESGLM_MODEL_PATH
        self.device = device or DEVICE
        self.model = None
        self.tokenizer = None
        self._loaded = False

    def load(self):
        if self._loaded:
            return self
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, torch_dtype=TORCH_DTYPE,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
        )
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        self.model.eval()
        self._loaded = True
        return self

    def _build_input(self, prompt, system_prompt):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return self.tokenizer(text, return_tensors="pt").to(self.device)

    def generate(self, prompt: str,
                 system_prompt: str = "你是一位经验丰富的临床麻醉医师，精通麻醉学知识。",
                 temperature=None, top_p=None, max_new_tokens=None,
                 stop_strings: Optional[List[str]] = None) -> str:

        if not self._loaded: self.load()
        inputs = self._build_input(prompt, system_prompt)
        prompt_length = inputs["input_ids"].shape[1]


        stopping_criteria = None
        if stop_strings:
            stopping_criteria = StoppingCriteriaList([
                StopOnStrings(self.tokenizer, stop_strings, prompt_length)
            ])

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens or LLM_MAX_GEN_LENGTH,
                temperature=temperature or LLM_TEMPERATURE,
                top_p=top_p or LLM_TOP_P,
                do_sample=True, repetition_penalty=1.05,
                stopping_criteria=stopping_criteria,
            )
        generated = outputs[0][prompt_length:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()


        if stop_strings:
            text = self._truncate_at_stop(text, stop_strings)

        return text

    def generate_with_logprobs(self, prompt: str,
                                system_prompt: str = "你是一位经验丰富的临床麻醉医师，精通麻醉学知识。",
                                temperature=None, top_p=None, max_new_tokens=None,
                                stop_strings: Optional[List[str]] = None) -> tuple:

        if not self._loaded: self.load()
        inputs = self._build_input(prompt, system_prompt)
        prompt_length = inputs["input_ids"].shape[1]


        stopping_criteria = None
        if stop_strings:
            stopping_criteria = StoppingCriteriaList([
                StopOnStrings(self.tokenizer, stop_strings, prompt_length)
            ])

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens or LLM_MAX_GEN_LENGTH,
                temperature=temperature or LLM_TEMPERATURE,
                top_p=top_p or LLM_TOP_P,
                do_sample=True, repetition_penalty=1.05,
                output_scores=True, return_dict_in_generate=True,
                stopping_criteria=stopping_criteria,
            )
        generated_ids = outputs.sequences[0][prompt_length:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


        if stop_strings:
            response = self._truncate_at_stop(response, stop_strings)

        log_probs = []
        for step, score in enumerate(outputs.scores):
            probs = torch.softmax(score[0], dim=-1)
            tid = generated_ids[step]
            if tid < len(probs):
                log_probs.append(torch.log(probs[tid] + 1e-10).item())

            decoded_so_far = self.tokenizer.decode(generated_ids[:step + 1], skip_special_tokens=True)
            if stop_strings and any(s in decoded_so_far for s in stop_strings):
                break
        avg_lp = sum(log_probs) / max(len(log_probs), 1)
        return response, avg_lp

    @staticmethod
    def _truncate_at_stop(text: str, stop_strings: List[str]) -> str:

        earliest_pos = len(text)
        for s in stop_strings:
            pos = text.find(s)
            if pos != -1 and pos + len(s) < earliest_pos:
                earliest_pos = pos + len(s)
        return text[:earliest_pos]