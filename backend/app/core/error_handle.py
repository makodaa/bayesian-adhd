import traceback

def error_handle(function):
    def wrapped(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except Exception as e:
            print(f"Error in {function.__name__}: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            return {"error": str(e)}
    return wrapped