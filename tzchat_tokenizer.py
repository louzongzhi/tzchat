import os
import re
import inspect
import sentencepiece as spm
from shutil import copyfile
from typing import Any, Dict, List, Optional, Union
from mindformers.tools import logger
from mindformers.models.tokenization_utils import PreTrainedTokenizer, AddedToken
from mindformers.models.tokenization_utils_base import TensorType, BatchEncoding
from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
from utils import _compile_jinja_template, _render_with_assistant_indices, get_json_schema

VOCAB_FILES_NAMES = {"vocab_file": "tokenizer.model"}


@MindFormerRegister.register(MindFormerModuleType.TOKENIZER)
class tzchatTokenizer(PreTrainedTokenizer):
    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]
    FILE_LIST = ['tokenizer_config.json']

    def __init__(
            self,
            vocab_file,
            unk_token="<unk>",
            bos_token="<_start>",
            eos_token="<_end>",
            pad_token="<_pad>",
            sp_model_kwargs: Optional[Dict[str, Any]] = None,
            add_bos_token=False,
            add_eos_token=False,
            clean_up_tokenization_spaces=False,
            **kwargs,
    ):
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs
        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False, single_word=False, normalized=True) \
            if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False, single_word=True, normalized=True) \
            if isinstance(eos_token, str) else eos_token
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False, single_word=True, normalized=True) \
            if isinstance(unk_token, str) else unk_token
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False, single_word=True, normalized=True) \
            if isinstance(pad_token, str) else pad_token

        self.vocab_file = vocab_file
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(vocab_file)

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            sp_model_kwargs=self.sp_model_kwargs,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        state["sp_model"] = None
        return state

    def __setstate__(self, d):
        self.__dict__ = d
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(self.vocab_file)

    @property
    def vocab_size(self):
        return self.sp_model.get_piece_size()

    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text):
        return self.sp_model.encode(text, out_type=str)

    def _convert_token_to_id(self, token):
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, index):
        token = self.sp_model.IdToPiece(index)
        return token

    def convert_tokens_to_string(self, tokens):
        current_sub_tokens = []
        out_string = ""
        for token in tokens:
            # make sure that special tokens are not decoded using sentencepiece model
            if token in self.all_special_tokens:
                out_string += self.sp_model.decode(current_sub_tokens) + token
                current_sub_tokens = []
            else:
                current_sub_tokens.append(token)
        out_string += self.sp_model.decode(current_sub_tokens)
        return out_string

    # pylint: disable=R1710
    def save_vocabulary(self, save_directory, filename_prefix=None):
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return None
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        elif not os.path.isfile(self.vocab_file):
            flags_ = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
            with os.fdopen(os.open(out_vocab_file, flags_, 0o750), 'wb') as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        return out_vocab_file

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        bos_token_id = [self.bos_token_id] if self.add_bos_token else []
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []

        output = bos_token_id + token_ids_0 + eos_token_id

        if token_ids_1 is not None:
            output = output + bos_token_id + token_ids_1 + eos_token_id

        return output

    def get_special_tokens_mask(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None,
                                already_has_special_tokens: bool = False):
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        bos_token_id = [1] if self.add_bos_token else []
        eos_token_id = [1] if self.add_eos_token else []

        if token_ids_1 is None:
            return bos_token_id + ([0] * len(token_ids_0)) + eos_token_id
        return (
            bos_token_id
            + ([0] * len(token_ids_0))
            + eos_token_id
            + bos_token_id
            + ([0] * len(token_ids_1))
            + eos_token_id
        )

    def create_token_type_ids_from_sequences(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None):
        bos_token_id = [self.bos_token_id] if self.add_bos_token else []
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []

        output = [0] * len(bos_token_id + token_ids_0 + eos_token_id)

        if token_ids_1 is not None:
            output += [1] * len(bos_token_id + token_ids_1 + eos_token_id)

        return output

    # pylint: disable=R1702
    def apply_chat_template(
            self,
            conversation: Union[List[Dict[str, str]], List[List[Dict[str, str]]]],
            tools: Optional[List[Dict]] = None,
            documents: Optional[List[Dict[str, str]]] = None,
            chat_template: Optional[str] = None,
            add_generation_prompt: bool = False,
            continue_final_message: bool = False,
            tokenize: bool = True,
            padding: bool = False,
            truncation: bool = False,
            max_length: Optional[int] = None,
            return_tensors: Optional[Union[str, TensorType]] = None,
            return_dict: bool = False,
            return_assistant_tokens_mask: bool = False,
            tokenizer_kwargs: Optional[Dict[str, Any]] = None,
            **kwargs,
    ) -> Union[str, List[int], List[str], List[List[int]], BatchEncoding]:
        if return_dict and not tokenize:
            raise ValueError(
                "`return_dict=True` is incompatible with `tokenize=False`, because there is no dict "
                "of tokenizer outputs to return."
            )

        if return_assistant_tokens_mask and not return_dict:
            raise ValueError("`return_assistant_tokens_mask=True` is incompatible with `return_dict=False`")

        if tokenizer_kwargs is None:
            tokenizer_kwargs = {}

        chat_template = self.get_chat_template(chat_template, tools)

        if return_assistant_tokens_mask and not re.search(r"\{\%-?\s*generation\s*-?\%\}", chat_template):
            logger.warning_once(
                "return_assistant_tokens_mask==True but chat template does not contain `{% generation %}` keyword."
            )

        # Compilation function uses a cache to avoid recompiling the same template
        compiled_template = _compile_jinja_template(chat_template)

        if isinstance(conversation, (list, tuple)) and (
                isinstance(conversation[0], (list, tuple)) or hasattr(conversation[0], "messages")
        ):
            conversations = conversation
            is_batched = True
        else:
            conversations = [conversation]
            is_batched = False

        if continue_final_message:
            if add_generation_prompt:
                raise ValueError(
                    "continue_final_message and add_generation_prompt are not compatible. "
                    "Use continue_final_message when you want the model to continue the final message, "
                    "and add_generation_prompt when you want to add a header "
                    "that will prompt it to start a new assistant message instead."
                )
            if return_assistant_tokens_mask:
                raise ValueError("continue_final_message is not compatible with return_assistant_tokens_mask.")

            # We accept either JSON schemas or functions for tools. If we get functions, we convert them to schemas
        if tools is not None:
            tool_schemas = []
            for tool in tools:
                if isinstance(tool, dict):
                    tool_schemas.append(tool)
                elif inspect.isfunction(tool):
                    tool_schemas.append(get_json_schema(tool))
                else:
                    raise ValueError(
                        "Tools should either be a JSON schema, or a callable function with type hints "
                        "and a docstring suitable for auto-conversion to a schema."
                    )
        else:
            tool_schemas = None

        if documents is not None:
            for document in documents:
                if not isinstance(document, dict):
                    raise TypeError("Documents should be a list of dicts with 'title' and 'text' keys!")

        rendered = []
        all_generation_indices = []
        template_kwargs = {**self.special_tokens_map, **kwargs}  # kwargs overwrite special tokens if both are present
        for chat in conversations:
            if hasattr(chat, "messages"):
                # Indicates it's a Conversation object
                chat = chat.messages
            if return_assistant_tokens_mask:
                rendered_chat, generation_indices = _render_with_assistant_indices(
                    compiled_template=compiled_template,
                    messages=chat,
                    tools=tool_schemas,
                    documents=documents,
                    add_generation_prompt=add_generation_prompt,
                    **template_kwargs,
                )
                all_generation_indices.append(generation_indices)
            else:
                rendered_chat = compiled_template.render(
                    messages=chat,
                    tools=tool_schemas,
                    documents=documents,
                    add_generation_prompt=add_generation_prompt,
                    **template_kwargs,
                )
            if continue_final_message:
                final_message = chat[-1]["content"]
                if isinstance(final_message, (list, tuple)):
                    final_message = final_message[-1]["text"]
                final_message = final_message.strip()
                rendered_chat = rendered_chat[: rendered_chat.rindex(final_message) + len(final_message)].rstrip()
            rendered.append(rendered_chat)

        if not is_batched:
            rendered = rendered[0]

        if tokenize:
            out = self(
                rendered,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                add_special_tokens=False,
                return_tensors=return_tensors,
                **tokenizer_kwargs,
            )
            if return_dict:
                if return_assistant_tokens_mask:
                    assistant_masks = []
                    if is_batched or return_tensors:
                        input_ids = out["input_ids"]
                    else:
                        input_ids = [out["input_ids"]]
                    for i, cur_input_id in enumerate(input_ids):
                        current_mask = [0] * len(cur_input_id)
                        for assistant_start_char, assistant_end_char in all_generation_indices[i]:
                            start_token = out.char_to_token(i, assistant_start_char)
                            end_token = out.char_to_token(i, assistant_end_char - 1)
                            if start_token is None:
                                # start_token is out of bounds maybe due to truncation.
                                break
                            for token_id in range(start_token, end_token + 1 if end_token else len(cur_input_id)):
                                current_mask[token_id] = 1
                        assistant_masks.append(current_mask)
                    out["assistant_masks"] = assistant_masks if is_batched else assistant_masks[0]
                return out
            return out["input_ids"]
        return rendered

    def get_chat_template(self, chat_template: Optional[str] = None, tools: Optional[List[Dict]] = None) -> str:
        # First, handle the cases when the model has a dict of multiple templates
        if isinstance(self.chat_template, dict):
            template_dict = self.chat_template
            if chat_template is not None and chat_template in template_dict:
                # The user can pass the name of a template to the chat template argument instead of an entire template
                chat_template = template_dict[chat_template]
            elif chat_template is None:
                if tools is not None and "tool_use" in template_dict:
                    chat_template = template_dict["tool_use"]
                elif "default" in template_dict:
                    chat_template = template_dict["default"]
                else:
                    raise ValueError(
                        "This model has multiple chat templates with no default specified! Please either pass a chat "
                        "template or the name of the template you wish to use to the `chat_template` argument. "
                        f"Available template names are {sorted(template_dict.keys())}."
                    )
        elif chat_template is None:
            # These are the cases when the model has a single template
            # priority: `chat_template` argument > `tokenizer.chat_template`
            if self.chat_template is not None:
                chat_template = self.chat_template
            else:
                raise ValueError(
                    "Cannot use chat template functions because tokenizer.chat_template is not set and no template "
                    "argument was passed! For information about writing templates and setting the "
                    "tokenizer.chat_template attribute, please see the documentation at "
                    "https://huggingface.co/docs/transformers/main/en/chat_templating"
                )
        return chat_template
