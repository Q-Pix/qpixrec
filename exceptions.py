class Error(Exception):
    """Base class for other exceptions"""
    pass

class DataNotPresentError(Error):
    """No reset data is present in the given directory"""
    def __init__(self, message="No reset data file (resets_output.txt) is present in the given directory"):
        self.message = message
        super().__init__(self.message)