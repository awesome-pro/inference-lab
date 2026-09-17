from engine.request import Request



class Runner:
    def run_prefill(self, request: Request):
        print(f"{request.id} is running prefill")
        request.status = "decoding"

    def run_decode(self, request: Request):
        print(f"{request.id} is decoding")
        request.generated_tokens.append(0)

        if len(request.generated_tokens) >= request.max_new_tokens:
            request.status = "finished"
