import functools
import os
import re
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import List, Optional

import paddle
from packaging.version import Version
from paddle.utils.cpp_extension.cpp_extension import CUDA_HOME

from . import env as jit_env

# torch compat polyfills
_TORCH_PATH = paddle.__path__[0]
def _get_num_workers(verbose: bool) -> Optional[int]:
    max_jobs = os.environ.get('MAX_JOBS')
    if max_jobs is not None and max_jobs.isdigit():
        return int(max_jobs)
    return None



@functools.cache
def get_cuda_version() -> Version:
    if CUDA_HOME is None:
        nvcc = "nvcc"
    else:
        nvcc = os.path.join(CUDA_HOME, "bin/nvcc")
    txt = subprocess.check_output([nvcc, "--version"], text=True)
    matches = re.findall("release (\\d+\\.\\d+),", txt)
    if not matches:
        raise RuntimeError(
            f"Could not parse CUDA version from nvcc --version output: {txt}"
        )
    return Version(matches[0])


def is_cuda_version_at_least(version_str: str) -> bool:
    return get_cuda_version() >= Version(version_str)


def _get_glibcxx_abi_build_flags() -> List[str]:
    glibcxx_abi_cflags = [
        # TODO: Provide a python interface like PyTorch
        # "-D_GLIBCXX_USE_CXX11_ABI=" + str(int(torch._C._GLIBCXX_USE_CXX11_ABI))
        "-D_GLIBCXX_USE_CXX11_ABI=1"
    ]
    return glibcxx_abi_cflags


def join_multiline(vs: List[str]) -> str:
    return " $\n    ".join(vs)


def generate_ninja_build_for_op(
    name: str,
    sources: List[Path],
    extra_cflags: Optional[List[str]],
    extra_cuda_cflags: Optional[List[str]],
    extra_ldflags: Optional[List[str]],
    extra_include_dirs: Optional[List[Path]],
    needs_device_linking: bool = False,
) -> str:
    system_includes = [
        sysconfig.get_path("include"),
        "$torch_home/include",
        "$torch_home/include/torch/csrc/api/include",
        "$cuda_home/include",
        jit_env.FLASHINFER_INCLUDE_DIR.resolve(),
        jit_env.FLASHINFER_CSRC_DIR.resolve(),
    ]
    system_includes += [p.resolve() for p in jit_env.CUTLASS_INCLUDE_DIRS]
    system_includes.append(jit_env.SPDLOG_INCLUDE_DIR.resolve())
    common_cflags = [
        "-DTORCH_EXTENSION_NAME=$name",
        "-DTORCH_API_INCLUDE_EXTENSION_H",
        "-DPy_LIMITED_API=0x03090000",
        "-DPADDLE_WITH_CUDA",
    ]

    # TODO: Provide a python interface like torch
    # common_cflags += _get_pybind11_abi_build_flags()
    common_cflags += ['-DPYBIND11_COMPILER_TYPE=\\"_gcc\\"', '-DPYBIND11_STDLIB=\\"_libstdcpp\\"', '-DPYBIND11_BUILD_ABI=\\"_cxxabi1018\\"']
    common_cflags += _get_glibcxx_abi_build_flags()
    if extra_include_dirs is not None:
        for extra_dir in extra_include_dirs:
            common_cflags.append(f"-I{extra_dir.resolve()}")
    for sys_dir in system_includes:
        common_cflags.append(f"-isystem {sys_dir}")
    cflags = ["$common_cflags", "-fPIC"]
    if extra_cflags is not None:
        cflags += extra_cflags
    cuda_cflags: List[str] = []
    cc_env = os.environ.get("CC")
    if cc_env is not None:
        cuda_cflags += ["-ccbin", cc_env]
    cuda_cflags += [
        "$common_cflags",
        "--compiler-options=-fPIC",
        "--expt-relaxed-constexpr",
    ]
    cuda_version = get_cuda_version()
    if cuda_version >= Version("12.8"):
        cuda_cflags += [
            "-static-global-template-stub=false",
        ]
    # TODO: Provide a python interface, currently the `_get_cuda_arch_flags(extra_cuda_cflags)` returns []
    # cuda_cflags += _get_cuda_arch_flags(extra_cuda_cflags)
    if extra_cuda_cflags is not None:
        cuda_cflags += extra_cuda_cflags
    ldflags = [
        "-shared",
        "-L$torch_home/lib",
        "-L$torch_home/base",
        "-L$cuda_home/lib64",
        "-lpaddle",
        "-lphi",
        "-lphi_core",
        "-lphi_gpu",
        "-lcommon",
        "-lcudart",
        "-lcuda",
    ]
    env_extra_ldflags = os.environ.get("FLASHINFER_EXTRA_LDFLAGS")
    if env_extra_ldflags:
        try:
            import shlex

            ldflags += shlex.split(env_extra_ldflags)
        except ValueError as e:
            print(
                f"Warning: Could not parse FLASHINFER_EXTRA_LDFLAGS with shlex: {e}. Falling back to simple split.",
                file=sys.stderr,
            )
            ldflags += env_extra_ldflags.split()
    if extra_ldflags is not None:
        ldflags += extra_ldflags
    cxx = os.environ.get("CXX", "c++")
    cuda_home = CUDA_HOME or "/usr/local/cuda"
    nvcc = os.environ.get("PYTORCH_NVCC", "$cuda_home/bin/nvcc")
    lines = [
        "ninja_required_version = 1.3",
        f"name = {name}",
        f"cuda_home = {cuda_home}",
        f"torch_home = {_TORCH_PATH}",
        f"cxx = {cxx}",
        f"nvcc = {nvcc}",
        "",
        "common_cflags = " + join_multiline(common_cflags),
        "cflags = " + join_multiline(cflags),
        "post_cflags =",
        "cuda_cflags = " + join_multiline(cuda_cflags),
        "cuda_post_cflags =",
        "ldflags = " + join_multiline(ldflags),
        "",
        "rule compile",
        "  command = $cxx -MMD -MF $out.d $cflags -c $in -o $out $post_cflags",
        "  depfile = $out.d",
        "  deps = gcc",
        "",
        "rule cuda_compile",
        "  command = $nvcc --generate-dependencies-with-compile --dependency-output $out.d $cuda_cflags -c $in -o $out $cuda_post_cflags",
        "  depfile = $out.d",
        "  deps = gcc",
        "",
    ]
    if needs_device_linking:
        lines.extend(
            ["rule nvcc_link", "  command = $nvcc -shared $in $ldflags -o $out", ""]
        )
    else:
        lines.extend(["rule link", "  command = $cxx $in $ldflags -o $out", ""])
    objects = []
    for source in sources:
        is_cuda = source.suffix == ".cu"
        object_suffix = ".cuda.o" if is_cuda else ".o"
        cmd = "cuda_compile" if is_cuda else "compile"
        obj_name = source.with_suffix(object_suffix).name
        obj = f"$name/{obj_name}"
        objects.append(obj)
        lines.append(f"build {obj}: {cmd} {source.resolve()}")
    lines.append("")
    link_rule = "nvcc_link" if needs_device_linking else "link"
    lines.append(f"build $name/$name.so: {link_rule} " + " ".join(objects))
    lines.append("default $name/$name.so")
    lines.append("")
    return "\n".join(lines)


def run_ninja(workdir: Path, ninja_file: Path, verbose: bool) -> None:
    workdir.mkdir(parents=True, exist_ok=True)
    command = [
        "ninja",
        "-v",
        "-C",
        str(workdir.resolve()),
        "-f",
        str(ninja_file.resolve()),
    ]
    num_workers = _get_num_workers(verbose)
    if num_workers is not None:
        command += ["-j", str(num_workers)]
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        subprocess.run(
            command,
            stdout=None if verbose else subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(workdir.resolve()),
            check=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        msg = "Ninja build failed."
        if e.output:
            msg += " Ninja output:\n" + e.output
        raise RuntimeError(msg) from e
