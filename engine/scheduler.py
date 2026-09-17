from collections import deque
from engine.request import Request


class Scheduler:
    def __init__(self, max_running_requests):
        self.max_running_requests: int = max_running_requests
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self.finished: list[Request] = []

    def add_request(self, request):
        request.status = "waiting"
        self.waiting.append(request)

    def has_work(self):
        return bool(self.waiting or self.running)

    def free_slots(self):
        return self.max_running_requests - len(self.running)

    def schedule(self):
        available_slots = self.max_running_requests - len(self.running)
        if available_slots <= 0:
            return

        for _ in range(available_slots):
            if not self.waiting:
                break

            request = self.waiting.popleft()
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
