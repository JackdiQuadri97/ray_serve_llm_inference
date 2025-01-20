import os
from starlette.requests import Request
from typing import Dict
import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
from ray import serve

@serve.deployment()
class LLM:
    def __init__(self, model_id: str, max_new_tokens: int):
        self.max_new_tokens = max_new_tokens
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_ID)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.bfloat16).to(self.device)

    def __call__(self, request: Request) -> Dict:
        input_text = request.query_params["text"]
        inputs = self.tokenizer(input_text, return_token_type_ids=False, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        generated_text = self.tokenizer.decode(outputs[0])
        return {"response": generated_text}

app = LLM.bind(
    model_id=os.environ.get('MODEL_ID')
    max_new_tokens=os.environ.get('MAX_NEW_TOKENS')
)
