from typing import List, Optional
from cog import BasePredictor, Input
from transformers import LLaMAForCausalLM, LLaMATokenizer
import torch

from train import PROMPT

CACHE_DIR = 'homer_out'

class Predictor(BasePredictor):
    def setup(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = LLaMAForCausalLM.from_pretrained("final_homer_two_out", cache_dir=CACHE_DIR, local_files_only=True)
        self.model = self.model.half()
        self.model.to(self.device)
        self.tokenizer = LLaMATokenizer.from_pretrained("final_homer_two_out", cache_dir=CACHE_DIR, local_files_only=True)
        self.tokenizer.pad_token_id = 0

    def predict(
        self,
        prompt: str = Input(description=f"Prompt to sened to model.", default=PROMPT),
        character: str = Input(description=f"Simpsons character to respond as", default="Homer Simpson"),
        n: int = Input(description="Number of output sequences to generate", default=1, ge=1, le=5),
        total_tokens: int = Input(
            description="Maximum number of tokens for input + generation. A word is generally 2-3 tokens",
            ge=1,
            default=4000
        ),
        temperature: float = Input(
            description="Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic, 0.75 is a good starting value.",
            ge=0.01,
            le=5,
            default=0.75,
        ),
        top_p: float = Input(
            description="When decoding text, samples from the top p percentage of most likely tokens; lower to ignore less likely tokens",
            ge=0.01,
            le=1.0,
            default=1.0
        ),
        repetition_penalty: float = Input(
            description="Penalty for repeated words in generated text; 1 is no penalty, values greater than 1 discourage repetition, less than 1 encourage it.",
            ge=0.01,
            le=5,
            default=1
        )
        ) -> List[str]:
        prompt = ''.join([prompt, '\n', character, ':'])
        print(f'final prompt:\n{prompt}')
        input = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        temperature = float(temperature)

        outputs = self.model.generate(
            input,
            num_return_sequences=n,
            max_length=total_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty
        )
        out = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # removing prompt b/c it's returned with every input 
        out = [val.split(f'{character}:')[-1] for val in out]
        return out
        