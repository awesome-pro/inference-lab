class Engine:
    def __init__(self, scheduler, runner):
        self.scheduler = scheduler
        self.runner = runner
        self.now = 0
        self.history = []

    def submit(self, request):
        self.scheduler.add_request(request)

    def snapshot(self):
        """Current state of all three queues, as plain ids."""
        return {
            "time": self.now,
            "waiting": [r.id for r in self.scheduler.waiting],
            "running": [r.id for r in self.scheduler.running],
            "finished": [r.id for r in self.scheduler.finished],
        }

    def step(self):
        self.scheduler.schedule(self.now)
        self.history.append(self.snapshot())

        for req in self.scheduler.running:
            if req.status == "prefilling":
                self.runner.run_prefill(req)
            elif req.status == "decoding":
                self.runner.run_decode(req)

        self.scheduler.retire_finished()
        self.now += 1

    def run(self):
        while self.scheduler.has_work():
            self.step()
        return self.scheduler.finished
