class Engine:
    def __init__(self, scheduler, runner):
        self.scheduler = scheduler
        self.runner = runner
        self.now = 0

    def submit(self, request):
        self.scheduler.add_request(request)

    def step(self):
        self.scheduler.schedule()
        
        for req in self.scheduler.running:
            if req.status == "prefilling":
                self.runner.run_prefill(req)
            elif req.status == "decoding":
                self.runner.run_decode(req)

        self.scheduler.retire_finished()

    def run(self):
        while self.scheduler.has_work():
            self.step()
            self.now += 1
        return self.scheduler.finished
