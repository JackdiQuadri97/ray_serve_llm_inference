import os
from starlette.requests import Request
from typing import Dict
import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
from ray import serve

MODEL_ID = "Aleph-Alpha/Pharia-1-LLM-7B-control-aligned-hf"
MAX_NEW_TOKENS = 50


@serve.deployment()
class LLM:
    def __init__(self):
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_ID)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16).to(self.device)

    def __call__(self, request: Request) -> Dict:
        input_text = request.query_params["text"]
        inputs = self.tokenizer(input_text, return_token_type_ids=False, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
        generated_text = self.tokenizer.decode(outputs[0])
        return {"response": generated_text}

app = LLM.bind()
