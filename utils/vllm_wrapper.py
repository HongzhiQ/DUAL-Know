import os
import sys
import math
from typing import List, Dict, Optional, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import (
    ANESGLM_MODEL_PATH, DEVICE, TORCH_DTYPE,
    LLM_TEMPERATURE, LLM_TOP_P, LLM_MAX_GEN_LENGTH,
)


class AnesGLM_vLLM:


    def __init__(
        self,
        model_path: str = None,
        device: str = None,
        mode: str = "offline",
        server_url: str = "http://localhost:8000",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        max_model_len: int = 4096,
        quantization: Optional[str] = None,
    ):
        self.model_path = model_path or ANESGLM_MODEL_PATH
        self.device = device or DEVICE
        self.mode = mode
        self.server_url = server_url.rstrip("/")
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.quantization = quantization

        self._llm = None
        self._tokenizer = None
        self._loaded = False

    def load(self) -> "AnesGLM_vLLM":
        if self._loaded:
            return self

        if self.mode == "offline":
            self._load_offline()
        elif self.mode == "server":
            self._load_server()
        else:
            raise ValueError(f"不支持的 mode: {self.mode}, 请选择 'offline' 或 'server'")

        self._loaded = True
        return self

    def _load_offline(self):

        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError(
                "请先安装 vLLM: pip install vllm\n"
                "要求: Python 3.8+, CUDA 11.8+, PyTorch 2.1+\n"
                "详见: https://docs.vllm.ai/en/latest/getting_started/installation.html"
            )

        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )

        self._llm = LLM(
            model=self.model_path,
            tokenizer=self.model_path,
            trust_remote_code=True,
            dtype="float16" if "cuda" in (self.device or "cuda") else "float32",
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            quantization=self.quantization,

        )

    def _load_server(self):

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "服务模式需要 openai 客户端: pip install openai"
            )

        self._client = OpenAI(
            api_key="EMPTY",
            base_url=f"{self.server_url}/v1",
        )


        try:
            models = self._client.models.list()
            model_ids = [m.id for m in models.data]
            self._server_model = model_ids[0] if model_ids else "AnesGLM"
        except Exception as e:
            raise ConnectionError(
                f"[vLLM-Server] 连接失败: {self.server_url}\n"
                f"请确认 vLLM server 已启动: {e}"
            )


    def _build_prompt(self, prompt: str, system_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        if self._tokenizer is not None:
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        return None


    def generate(
        self,
        prompt: str,
        system_prompt: str = "你是一位经验丰富的临床麻醉医师，精通麻醉学知识。",
        temperature: float = None,
        top_p: float = None,
        max_new_tokens: int = None,
        stop_strings: Optional[List[str]] = None,
    ) -> str:
        if not self._loaded:
            self.load()

        temperature = temperature or LLM_TEMPERATURE
        top_p = top_p or LLM_TOP_P
        max_new_tokens = max_new_tokens or LLM_MAX_GEN_LENGTH

        if self.mode == "offline":
            return self._generate_offline(
                prompt, system_prompt, temperature, top_p,
                max_new_tokens, stop_strings, logprobs=False
            )[0]
        else:
            return self._generate_server(
                prompt, system_prompt, temperature, top_p,
                max_new_tokens, stop_strings, logprobs=False
            )[0]

    def generate_with_logprobs(
        self,
        prompt: str,
        system_prompt: str = "你是一位经验丰富的临床麻醉医师，精通麻醉学知识。",
        temperature: float = None,
        top_p: float = None,
        max_new_tokens: int = None,
        stop_strings: Optional[List[str]] = None,
    ) -> Tuple[str, float]:

        if not self._loaded:
            self.load()

        temperature = temperature or LLM_TEMPERATURE
        top_p = top_p or LLM_TOP_P
        max_new_tokens = max_new_tokens or LLM_MAX_GEN_LENGTH

        if self.mode == "offline":
            return self._generate_offline(
                prompt, system_prompt, temperature, top_p,
                max_new_tokens, stop_strings, logprobs=True
            )
        else:
            return self._generate_server(
                prompt, system_prompt, temperature, top_p,
                max_new_tokens, stop_strings, logprobs=True
            )

    def generate_batch(
        self,
        prompts: List[str],
        system_prompt: str = "你是一位经验丰富的临床麻醉医师，精通麻醉学知识。",
        temperature: float = None,
        top_p: float = None,
        max_new_tokens: int = None,
        stop_strings: Optional[List[str]] = None,
        logprobs: bool = False,
    ) -> List[Tuple[str, Optional[float]]]:

        if not self._loaded:
            self.load()

        temperature = temperature or LLM_TEMPERATURE
        top_p = top_p or LLM_TOP_P
        max_new_tokens = max_new_tokens or LLM_MAX_GEN_LENGTH

        if self.mode == "offline":
            return self._generate_batch_offline(
                prompts, system_prompt, temperature, top_p,
                max_new_tokens, stop_strings, logprobs
            )
        else:

            from concurrent.futures import ThreadPoolExecutor
            results = []
            with ThreadPoolExecutor(max_workers=len(prompts)) as executor:
                futures = [
                    executor.submit(
                        self._generate_server,
                        p, system_prompt, temperature, top_p,
                        max_new_tokens, stop_strings, logprobs
                    )
                    for p in prompts
                ]
                for f in futures:
                    results.append(f.result())
            return results


    def _generate_offline(
        self, prompt, system_prompt, temperature, top_p,
        max_new_tokens, stop_strings, logprobs
    ) -> Tuple[str, float]:
        from vllm import SamplingParams

        full_prompt = self._build_prompt(prompt, system_prompt)

        params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            repetition_penalty=1.05,
            stop=stop_strings or [],
            logprobs=1 if logprobs else None,
        )

        outputs = self._llm.generate([full_prompt], params)
        output = outputs[0].outputs[0]

        text = output.text.strip()
        if stop_strings:
            text = self._truncate_at_stop(text, stop_strings)

        avg_lp = 0.0
        if logprobs and output.logprobs:
            lps = []
            for step_logprob in output.logprobs:

                if step_logprob:
                    for token_id, lp_obj in step_logprob.items():
                        lps.append(lp_obj.logprob)
                        break
            avg_lp = sum(lps) / max(len(lps), 1)

        return text, avg_lp

    def _generate_batch_offline(
        self, prompts, system_prompt, temperature, top_p,
        max_new_tokens, stop_strings, logprobs
    ) -> List[Tuple[str, Optional[float]]]:

        from vllm import SamplingParams

        full_prompts = [
            self._build_prompt(p, system_prompt) for p in prompts
        ]

        params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            repetition_penalty=1.05,
            stop=stop_strings or [],
            logprobs=1 if logprobs else None,
        )


        outputs = self._llm.generate(full_prompts, params)

        results = []
        for output in outputs:
            text = output.outputs[0].text.strip()
            if stop_strings:
                text = self._truncate_at_stop(text, stop_strings)

            avg_lp = None
            if logprobs and output.outputs[0].logprobs:
                lps = []
                for step_logprob in output.outputs[0].logprobs:
                    if step_logprob:
                        for token_id, lp_obj in step_logprob.items():
                            lps.append(lp_obj.logprob)
                            break
                avg_lp = sum(lps) / max(len(lps), 1)

            results.append((text, avg_lp))

        return results


    def _generate_server(
        self, prompt, system_prompt, temperature, top_p,
        max_new_tokens, stop_strings, logprobs
    ) -> Tuple[str, float]:

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        kwargs = dict(
            model=self._server_model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            stop=stop_strings or None,
            extra_body={"repetition_penalty": 1.05},
        )

        if logprobs:
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = 1

        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        text = (choice.message.content or "").strip()

        if stop_strings:
            text = self._truncate_at_stop(text, stop_strings)

        avg_lp = 0.0
        if logprobs and choice.logprobs and choice.logprobs.content:
            lps = [t.logprob for t in choice.logprobs.content if t.logprob is not None]
            avg_lp = sum(lps) / max(len(lps), 1)

        return text, avg_lp

    @staticmethod
    def _truncate_at_stop(text: str, stop_strings: List[str]) -> str:
        earliest_pos = len(text)
        for s in stop_strings:
            pos = text.find(s)
            if pos != -1 and pos + len(s) < earliest_pos:
                earliest_pos = pos + len(s)
        return text[:earliest_pos]



def create_llm(
    backend: str = "transformers",
    model_path: str = None,
    device: str = None,
    **kwargs,
):

    if backend == "transformers":
        from utils.llm_wrapper import AnesGLM
        return AnesGLM(model_path=model_path, device=device)

    elif backend == "vllm_offline":
        return AnesGLM_vLLM(
            model_path=model_path,
            device=device,
            mode="offline",
            **kwargs,
        )

    elif backend == "vllm_server":
        return AnesGLM_vLLM(
            model_path=model_path,
            device=device,
            mode="server",
            **kwargs,
        )

    else:
        raise ValueError(f"不支持的 backend: {backend}")


def print_server_launch_command(model_path: str = None):

    model_path = model_path or ANESGLM_MODEL_PATH
    cmd = f"""
# ============================================
# vLLM OpenAI-Compatible Server 启动命令
# ============================================

python -m vllm.entrypoints.openai.api_server \\
    --model {model_path} \\
    --served-model-name AnesGLM \\
    --host 0.0.0.0 \\
    --port 8000 \\
    --dtype float16 \\
    --max-model-len 4096 \\
    --gpu-memory-utilization 0.90 \\
    --trust-remote-code \\
    --disable-log-requests

# 多卡并行（2张GPU）:
# python -m vllm.entrypoints.openai.api_server \\
#     --model {model_path} \\
#     --tensor-parallel-size 2 \\
#     ...其他参数同上

# 量化推理（AWQ）:
# python -m vllm.entrypoints.openai.api_server \\
#     --model {model_path} \\
#     --quantization awq \\
#     ...其他参数同上
"""
    print(cmd)


if __name__ == "__main__":
    print_server_launch_command()
