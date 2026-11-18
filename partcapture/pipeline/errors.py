"""User-facing capture failures."""


class CaptureError(Exception):
    """A problem the operator can fix by retaking photos or reprinting sheets."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message
