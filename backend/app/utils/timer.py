import time


class Timer:
    def __enter__(self):
        self._enter_time = time.time()

        return self

    def __exit__(self, *exc_args):
        self._exit_time = time.time()
        print(f"{self._exit_time - self._enter_time:.2f} seconds elapsed")

        return self

    def elapsed(self):
        return time.time() - self._enter_time
