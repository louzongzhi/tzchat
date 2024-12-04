import copy
from collections import deque


class History:
    def __init__(self, tokenizer, history):
        self.input_history = deque()
        self.tokenizer = tokenizer
        if history:
            self._transfer_from_list(history)

    def _transfer_from_list(self, history):
        for message in history:
            content = message.get("content")
            # the token result may not be equal to the result model gen
            message.update(self.tokenizer(content))
            self.input_history.append(message)

    def __len__(self):
        return self.input_history.__len__()

    def __str__(self):
        return self.input_history.__str__()

    def __copy__(self):
        new_instance = type(self)(self.tokenizer, [])
        new_instance.input_history = copy.copy(self.input_history)
        return new_instance

    def __deepcopy__(self, memodict=None):
        new_instance = type(self)(self.tokenizer, [])
        new_instance.input_history = copy.deepcopy(self.input_history)
        return new_instance

    def append(self, message):
        content = message.get("content")
        if "input_ids" not in message or "attention_mask" not in message:
            message.update(self.tokenizer(content))
        self.input_history.append(message)

    def append_left(self, message):
        content = message.get("content")
        if "input_ids" not in message or "attention_mask" not in message:
            message.update(self.tokenizer(content))
        self.input_history.appendleft(message)

    def pop(self):
        x = self.input_history.pop()
        return x

    def pop_left(self):
        x = self.pop_left()
        return x

    def update(self, message):
        self.input_history.pop()
        self.append(message)
