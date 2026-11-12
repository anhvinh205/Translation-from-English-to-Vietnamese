"""Typed errors raised by the engine and mapped to HTTP responses by the API layer."""


class EngineError(Exception):
    """Base class for errors the translation engine raises on purpose."""


class InvalidInputError(EngineError):
    """The request cannot be processed (empty after sanitization, too long for the model, ...)."""


class EngineBusyError(EngineError):
    """All generation slots are busy; the caller should retry later."""


class ModelOutOfMemoryError(EngineError):
    """The model ran out of (GPU) memory while generating."""
