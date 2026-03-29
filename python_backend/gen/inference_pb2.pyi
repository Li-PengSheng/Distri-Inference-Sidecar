from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class InferRequest(_message.Message):
    __slots__ = ("request_id", "input_data", "model_name")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    INPUT_DATA_FIELD_NUMBER: _ClassVar[int]
    MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    input_data: bytes
    model_name: str
    def __init__(self, request_id: _Optional[str] = ..., input_data: _Optional[bytes] = ..., model_name: _Optional[str] = ...) -> None: ...

class InferResponse(_message.Message):
    __slots__ = ("request_id", "output_data", "latency_ms", "error")
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_DATA_FIELD_NUMBER: _ClassVar[int]
    LATENCY_MS_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    output_data: bytes
    latency_ms: int
    error: str
    def __init__(self, request_id: _Optional[str] = ..., output_data: _Optional[bytes] = ..., latency_ms: _Optional[int] = ..., error: _Optional[str] = ...) -> None: ...

class HealthRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class HealthResponse(_message.Message):
    __slots__ = ("healthy", "vram_used_mb", "vram_total_mb")
    HEALTHY_FIELD_NUMBER: _ClassVar[int]
    VRAM_USED_MB_FIELD_NUMBER: _ClassVar[int]
    VRAM_TOTAL_MB_FIELD_NUMBER: _ClassVar[int]
    healthy: bool
    vram_used_mb: float
    vram_total_mb: float
    def __init__(self, healthy: _Optional[bool] = ..., vram_used_mb: _Optional[float] = ..., vram_total_mb: _Optional[float] = ...) -> None: ...
