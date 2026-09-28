#!/usr/bin/env python
"""Shim around the SWE-bench harness (runs `swebench.harness.run_evaluation`).

Copies files into containers as root-owned tar entries (required on rootless Podman,
harmless on Docker), widens the Docker-SDK timeout for slow first image pulls, and
retries transient container/image list errors.
"""
import os
import runpy
import tarfile
from pathlib import Path

import swebench.harness.docker_utils as _du
import docker as _docker

_orig_from_env = _docker.from_env

def _copy_to_container(container, src, dst):
    src, dst = Path(src), Path(dst)
    if os.path.dirname(dst) == "":
        raise ValueError(f"Destination path parent directory cannot be empty!, dst: {dst}")

    def _rootify(ti: "tarfile.TarInfo") -> "tarfile.TarInfo":
        ti.uid = 0
        ti.gid = 0
        ti.uname = ""
        ti.gname = ""
        return ti

    tar_path = src.with_suffix(".tar")
    with tarfile.open(tar_path, "w") as tar:
        tar.add(src, arcname=dst.name, filter=_rootify)
    with open(tar_path, "rb") as fh:
        data = fh.read()
    container.exec_run(f"mkdir -p {dst.parent}")
    container.put_archive(os.path.dirname(dst), data)
    tar_path.unlink()

_du.copy_to_container = _copy_to_container

def _from_env_long_timeout(*args, **kwargs):
    kwargs.setdefault("timeout", int(os.environ.get("SWE_DOCKER_TIMEOUT", "1200")))
    return _orig_from_env(*args, **kwargs)

_docker.from_env = _from_env_long_timeout

import time as _time
from docker.errors import APIError as _APIError
import docker.models.containers as _dmc
import docker.models.images as _dmi

def _tolerant_list(orig):
    def _wrapped(self, *args, **kwargs):
        for attempt in range(6):
            try:
                return orig(self, *args, **kwargs)
            except _APIError:
                if attempt == 5:
                    return []
                _time.sleep(0.5 * (attempt + 1))
    return _wrapped

_dmc.ContainerCollection.list = _tolerant_list(_dmc.ContainerCollection.list)
_dmi.ImageCollection.list = _tolerant_list(_dmi.ImageCollection.list)

if __name__ == "__main__":
    runpy.run_module("swebench.harness.run_evaluation", run_name="__main__", alter_sys=True)
