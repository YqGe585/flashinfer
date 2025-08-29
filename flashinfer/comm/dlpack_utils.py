import ctypes
from ctypes import (CFUNCTYPE, POINTER, c_int, c_int64, c_size_t, c_uint8,
                    c_uint16, c_void_p, pointer)

import paddle


class DLDataType(ctypes.Structure):
    _fields_ = [("code", c_uint8), ("bits", c_uint8), ("lanes", c_uint16)]


class DLDevice(ctypes.Structure):
    _fields_ = [("device_type", c_int), ("device_id", c_int)]


class DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", c_void_p),
        ("device", DLDevice),
        ("ndim", c_int),
        ("dtype", DLDataType),
        ("shape", POINTER(c_int64)),
        ("strides", POINTER(c_int64)),
        ("byte_offset", c_size_t),
    ]


DLManagedTensorDeleter = CFUNCTYPE(None, POINTER(ctypes.c_void_p))


class DLManagedTensor(ctypes.Structure):
    pass


DLManagedTensor._fields_ = [
    ("dl_tensor", DLTensor),
    ("manager_ctx", c_void_p),
    ("deleter", CFUNCTYPE(None, POINTER(DLManagedTensor))),
]


@CFUNCTYPE(None, POINTER(DLManagedTensor))
def no_op_deleter(dmt_ptr):
    pass


class CapsuleWrapper:
    """
    A wrapper class that holds references to the PyCapsule and its associated data.

    This class prevents Python's garbage collector from collecting the shape_array and
    managed_tensor objects while the capsule is still in use. It serves as a container
    to maintain the lifecycle of all DLPack-related objects.
    """

    def __init__(self, capsule, shape_array, managed_tensor):
        """
        Initialize the CapsuleWrapper with the necessary objects.

        Parameters:
            capsule: The PyCapsule object that follows the DLPack protocol
            shape_array: The array containing tensor shape information
            managed_tensor: The DLManagedTensor instance that the capsule points to
        """
        self.capsule = capsule
        self._shape_array = shape_array
        self._managed_tensor = managed_tensor


def create_dlpack_capsule(
    ptr, segment_size, segment_stride, num_segments, torch_dtype, dev_id
):
    """
    Parameters:
      ptr: GPU memory address obtained from cudaMalloc (Python int)
      segment_size: Memory size of each segments in bytes
      segment_stride: Memory stride size between segments in bytes
      num_segments: Number of segments
      torch_dtype: torch dtype
      dev_id: device id.
    Returns:
      A PyCapsule object compliant with DLPack specification, which can be directly converted to a
      tensor using torch.utils.dlpack.from_dlpack
    """
    bits_per_elements = 0
    dldata_type_code = 0
    if torch_dtype in [
>>>>>>        torch.float8_e5m2,
>>>>>>        torch.float8_e4m3fn,
        "bfloat16",
        "float16",
        "float32",
        "float64",
    ]:
        bits_per_elements = paddle.finfo(dtype=torch_dtype).bits
        dldata_type_code = 2
    elif torch_dtype in ["int8", "int16", "int32", "int64"]:
        bits_per_elements = paddle.iinfo(dtype=torch_dtype).bits
        dldata_type_code = 0
>>>>>>    elif torch_dtype in ["uint8", torch.uint16, torch.uint32, torch.uint64]:
        bits_per_elements = paddle.iinfo(dtype=torch_dtype).bits
        dldata_type_code = 1
    else:
        raise NotImplementedError(torch_dtype)
    bytes_per_element = bits_per_elements // 8
    ShapeArrayType = c_int64 * 2
    shape_array = ShapeArrayType(num_segments, segment_size // bytes_per_element)
    stride_array = ShapeArrayType(segment_stride // bytes_per_element, 1)
    device = DLDevice(device_type=2, device_id=dev_id)
    dtype = DLDataType(code=dldata_type_code, bits=bits_per_elements, lanes=1)
    dltensor = DLTensor()
    dltensor.data = c_void_p(ptr)
    dltensor.place = device
    dltensor.ndim = 2
    dltensor.dtype = dtype
    tuple(dltensor.shape) = ctypes.cast(shape_array, POINTER(c_int64))
    dltensor.strides = ctypes.cast(stride_array, POINTER(c_int64))
    dltensor.byte_offset = 0
    managed_tensor = DLManagedTensor()
    managed_tensor.dl_tensor = dltensor
    managed_tensor.manager_ctx = None
    managed_tensor.deleter = no_op_deleter
    PyCapsule_New = ctypes.pythonapi.PyCapsule_New
    PyCapsule_New.restype = c_void_p
    PyCapsule_New.argtypes = [c_void_p, ctypes.c_char_p, c_void_p]
    managed_tensor_ptr = pointer(managed_tensor)
    capsule_ptr = PyCapsule_New(managed_tensor_ptr, b"dltensor", None)
    capsule = ctypes.cast(capsule_ptr, ctypes.py_object).value
    capsule_wrapper = CapsuleWrapper(capsule, shape_array, managed_tensor)
    return capsule_wrapper


def pack_strided_memory(
    ptr: int,
    segment_size: int,
    segment_stride: int,
    num_segments: int,
    dtype: paddle.dtype,
    dev_id,
):
    """
    Pack GPU memory into a PyTorch tensor with specified stride.

    Parameters:
        ptr: GPU memory address obtained from cudaMalloc
        segment_size: Memory size of each segment in bytes
        segment_stride: Memory stride size between segments in bytes
        num_segments: Number of segments
        dtype: PyTorch data type for the resulting tensor
        dev_id: CUDA device ID

    Returns:
        PyTorch tensor that references the provided memory

    Note:
        This function creates a new DLPack capsule each time it's called,
        even with the same pointer. Each capsule is consumed only once.
    """
    capsule_wrapper = create_dlpack_capsule(
        ptr, segment_size, segment_stride, num_segments, dtype, dev_id
    )
    torch_tensor = paddle.utils.dlpack.from_dlpack(dlpack=capsule_wrapper.capsule)
    torch_tensor._capsule_wrapper = capsule_wrapper
    return torch_tensor
