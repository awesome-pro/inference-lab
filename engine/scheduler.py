from collections import deque

from engine.request import Request


class Scheduler:
    def __init__(self, max_running_requests, max_batch_tokens, static=False):
        self.max_running_requests: int = max_running_requests
        self.max_batch_tokens: int = max_batch_tokens
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self.finished: list[Request] = []

    def add_request(self, request):
        if len(request.prompt_tokens) > self.max_batch_tokens:
            raise ValueError(
                f"{request.id}: prompt is {len(request.prompt_tokens)} tokens but "
                f"max_batch_tokens is {self.max_batch_tokens}"
            )

        request.status = "waiting"
        self.waiting.append(request)

    def has_work(self):
        return bool(self.waiting or self.running)

    def free_slots(self):
        return self.max_running_requests - len(self.running)

    def token_budget_left(self):
        return self.max_batch_tokens - self.batch_token_cost()

    def batch_token_cost(self):
        return sum(
            len(req.prompt_tokens) if req.status == "prefilling" else 1
            for req in self.running
        )

    def schedule(self, now):
        for _ in range(self.free_slots()):
            if not self.waiting:
                break

            request = self.waiting[0]
            if request.arrival_time > now:
                break

            prompt_cost = len(request.prompt_tokens)
            if self.batch_token_cost() + prompt_cost > self.max_batch_tokens:
                break

            self.waiting.popleft()
            request.status = "prefilling"
            self.running.append(request)

    def retire_finished(self):
        if not self.running:
            return

        finished_req = [
            req for req in self.running if req.status == "finished"
        ]

        for req in finished_req:
            self.running.remove(req)
            self.finished.append(req)
