from collections import deque
from engine.request import Request


class Scheduler:
    def __init__(self, max_running_requests, static=False):
        self.max_running_requests: int = max_running_requests
        self.static: bool = static
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
        # Static batching: hold the current batch until it is completely
        # drained, even though finished requests have already freed slots.
        if self.static and self.running:
            return

        for _ in range(self.free_slots()):
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
