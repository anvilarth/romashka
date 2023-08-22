import torch
from typing import Optional, List, Union
from transformers import StoppingCriteriaList, StoppingCriteria, GenerationConfig

# Mark models than can use HF generate method with input_embeddings
USE_HF_GENERATE = [
    "GPT2", "GPTJ", "GPTNeo", "LlamaForCausalLM", "Llama"
]


def isin(elements: torch.Tensor, test_elements: torch.Tensor):
    """
    Idea taken from: https://github.com/pytorch/pytorch/issues/3025
    """
    return (elements[..., None] == test_elements).any(-1)


class AnsweredQACriteria(StoppingCriteria):
    """
    A custom criteria to stop generation as soon as all the sequences in the batch have at least
    one Yes/No or any other fixed answer token (! single one !) after the prompt.
    """

    def __init__(self, prompt_length: int, answer_tokens_ids: List[int]):
        """
        Create a new criteria instance for a given generation run.
        Parameters
        ----------
        prompt_length : int
            The length of the prompt in tokens used to distinguish answer tokens.
            For a batch of multiple prompts of different
            lengths this should be the length of the longest prompt and other prompts should be
            padded.
        answer_tokens_ids: List[int]
            The answer tokens ids.
            TODO: In case if answer corresponds to multiple tokenizer's subtokens,
            we need to search for ids sequence appearence in input_ids.
        """
        self.prompt_length = prompt_length
        self.answer_tokens_ids = answer_tokens_ids

    def __call__(self, input_ids: torch.LongTensor, score: torch.FloatTensor = None, **kwargs) -> bool:
        all_batch_answered = sum([self._has_answer(input_ids, answer_token_id)
                                  for answer_token_id in self.answer_tokens_ids])
        return all_batch_answered.all()  # a single bool indicated that all batch questions have been answered

    def _has_answer(self, input_ids: torch.LongTensor, answer_token_id: int) -> torch.BoolTensor:
        has_answer = (input_ids[:, self.prompt_length:] == answer_token_id)
        return has_answer.any(dim=-1)  # bool tensor of shape [batch_size,]