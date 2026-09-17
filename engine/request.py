class Request:
    def __init__(self, id, prompt_tokens, arrival_time, max_new_tokens):
        self.id = id
        self.prompt_tokens: list[int] = prompt_tokens
        self.arrival_time: int = arrival_time
        self.max_new_tokens: int = max_new_tokens

        self.status: str = "waiting"
        self.generated_tokens: list[int] = []
