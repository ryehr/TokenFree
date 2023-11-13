from re import sub
from typing import List, Optional

from transformers.models.byt5.tokenization_byt5 import ByT5Tokenizer


class ByGPT5Tokenizer(ByT5Tokenizer):
    def __init__(
        self,
        add_prefix_space=False,
        add_bos_token=False,
        add_eos_token=False,
        **kwargs
    ):
        super().__init__(
            add_prefix_space=add_prefix_space,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            **kwargs,
        )
        self.add_prefix_space = add_prefix_space
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token

    def get_special_tokens_mask(
        self, token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False
    ) -> List[int]:
        return super(ByT5Tokenizer, self).get_special_tokens_mask(
            token_ids_0=token_ids_0,
            token_ids_1=token_ids_1,
            already_has_special_tokens=already_has_special_tokens,
        )

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        if self.add_bos_token:
            bos_token_ids = [self.bos_token_id]
        else:
            bos_token_ids = []
        if self.add_eos_token:
            eos_token_ids = [self.eos_token_id]
        else:
            eos_token_ids = []

        if token_ids_1 is None:
            return bos_token_ids + token_ids_0 + eos_token_ids # pyright: ignore

        return bos_token_ids + token_ids_0 + bos_token_ids + token_ids_1 + eos_token_ids # pyright: ignore

    def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
        if is_split_into_words or self.add_prefix_space:
            text = " " + text
        return (text, kwargs)

    @staticmethod
    def clean_up_tokenization(out_string: str) -> str:
        out_string = ByT5Tokenizer.clean_up_tokenization(out_string)
        # English poetry training data contains some tokenization artifacts
        # (i.e, model sometimes generates "He 'll" instead of "He'll"). The
        # following pattern fixes this
        out_string = sub(r"\s('\w{1,2})", r"\1", out_string)
        return out_string
