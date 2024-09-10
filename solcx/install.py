"""
Install solc
"""
import argparse
from concurrent.futures import process
import logging
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import warnings
import zipfile
from base64 import b64encode
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Union

import requests
from semantic_version import SimpleSpec, Version

from solcx import wrapper
from solcx.exceptions import (
    DownloadError,
    SolcInstallationError,
    SolcNotInstalled,
    UnexpectedVersionError,
    UnexpectedVersionWarning,
    UnsupportedVersionError,
)
from solcx.utils.lock import get_process_lock

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


BINARY_DOWNLOAD_BASE = "https://binaries.soliditylang.org/{}-amd64/{}"
SOURCE_DOWNLOAD_BASE = "https://github.com/ethereum/solidity/releases/download/v{}/{}"
GITHUB_RELEASES = "https://api.github.com/repos/ethereum/solidity/releases?per_page=100"

MINIMAL_SOLC_VERSION = Version("0.4.11")
LOGGER = logging.getLogger("solcx")

SOLCX_BINARY_PATH_VARIABLE = "SOLCX_BINARY_PATH"

_default_solc_binary = None
_target_os = None


def set_target_os(platform: Optional[str] = None):
    """
    Set the target platform for the solc binaries. If unset, it defaults to the current os.
    """
    global _target_os
    if platform is None or platform in ("linux", "macosx", "windows"):
        _target_os = platform
    else:
        raise OSError(f"Unsupported target OS: '{platform}' - py-solc-x supports 'linux', 'macosx', or 'windows'.")


def _get_target_os() -> str:
    return _target_os if _target_os else _get_os_name()


def _get_os_name() -> str:
    if sys.platform.startswith("linux"):
        return "linux"
    if sys.platform == "darwin":
        return "macosx"
    if sys.platform == "win32":
        return "windows"
    raise OSError(f"Unsupported OS: '{sys.platform}' - py-solc-x supports Linux, OSX and Windows")


def _convert_and_validate_version(version: Union[str, Version]) -> Version:
    # take a user-supplied version as a string or Version
    # validate the value, and return a Version object
    if not isinstance(version, Version):
        version = Version(version.lstrip("v"))
    if version not in SimpleSpec(">=0.4.11"):
        raise UnsupportedVersionError("py-solc-x does not support solc versions <0.4.11")
    return version


def _unlink_solc(solc_path: Path) -> None:
    solc_path.unlink()
    if _get_target_os() == "windows":
        shutil.rmtree(solc_path.parent)


def get_solcx_install_folder(solcx_binary_path: Union[Path, str] = None) -> Path:
    """
    Return the directory where `py-solc-x` stores installed `solc` binaries.

    By default, this is `~/.solcx`

    Arguments
    ---------
    solcx_binary_path : Path | str, optional
        User-defined path, used to override the default installation directory.

    Returns
    -------
    Path
        Subdirectory where `solc` binaries are are saved.
    """
    if _get_target_os() == _get_os_name():
        if os.getenv(SOLCX_BINARY_PATH_VARIABLE):
            return Path(os.environ[SOLCX_BINARY_PATH_VARIABLE])
        elif solcx_binary_path is not None:
            return Path(solcx_binary_path)
        else:
            path = Path.home().joinpath(".solcx")
            path.mkdir(exist_ok=True)
            return path
    else:
        path = Path.home().joinpath(f".solcx-{_get_target_os()}")
        path.mkdir(exist_ok=True)
        return path


def _get_which_solc() -> Path:
    # get the path for the currently installed `solc` version, if any
    if _get_os_name() == "windows":
        response = subprocess.check_output(["where.exe", "solc"], encoding="utf8").strip()
    else:
        response = subprocess.check_output(["which", "solc"], encoding="utf8").strip()

    return Path(response)


def import_installed_solc(solcx_binary_path: Union[Path, str] = None) -> List[Version]:
    """
    Search for and copy installed `solc` versions into the local installation folder.

    Arguments
    ---------
    solcx_binary_path : Path | str, optional
        User-defined path, used to override the default installation directory.

    Returns
    -------
    List
        Imported solc versions
    """
    try:
        path_list = [_get_which_solc()]
    except (FileNotFoundError, subprocess.CalledProcessError):
        path_list = []

    # on OSX, also copy all versions of solc from cellar
    if _get_os_name() == "macosx":
        path_list.extend(Path("/usr/local/Cellar").glob("solidity*/**/solc"))

    imported_versions = []
    for path in path_list:
        try:
            version = wrapper._get_solc_version(path)
            assert version not in get_installed_solc_versions()
        except Exception:
            continue

        copy_path = get_solcx_install_folder(solcx_binary_path).joinpath(f"solc-v{version}")
        if _get_os_name() == "windows":
            copy_path.mkdir()
            copy_path = copy_path.joinpath("solc.exe")

        shutil.copy(path, copy_path)
        try:
            # confirm that solc still works after being copied
            assert version == wrapper._get_solc_version(copy_path)
            imported_versions.append(version)
        except Exception:
            _unlink_solc(copy_path)

    return imported_versions


def get_executable(
    version: Union[str, Version] = None, solcx_binary_path: Union[Path, str] = None
) -> Path:
    """
    Return the Path to an installed `solc` binary.

    Arguments
    ---------
    version : str | Version, optional
        Installed `solc` version to get the path of. If not given, returns the
        path of the active version.
    solcx_binary_path : Path | str, optional
        User-defined path, used to override the default installation directory.

    Returns
    -------
    Path
        `solc` executable.
    """
    if not version:
        if not _default_solc_binary:
            raise SolcNotInstalled(
                "Solc is not installed. Call solcx.get_installable_solc_versions()"
                " to view for available versions and solcx.install_solc() to install."
            )
        return _default_solc_binary

    version = _convert_and_validate_version(version)
    solc_bin = get_solcx_install_folder(solcx_binary_path).joinpath(f"solc-v{version}")
    if _get_target_os() == "windows":
        solc_bin = solc_bin.joinpath("solc.exe")
    if not solc_bin.exists():
        raise SolcNotInstalled(
            f"solc {version} has not been installed."
            f" Use solcx.install_solc('{version}') to install."
        )
    return solc_bin


def set_solc_version(
    version: Union[str, Version], silent: bool = False, solcx_binary_path: Union[Path, str] = None
) -> None:
    """
    Set the currently active `solc` binary.

    Arguments
    ---------
    version : str | Version, optional
        Installed `solc` version to get the path of. If not given, returns the
        path of the active version.
    silent : bool, optional
        If True, do not generate any logger output.
    solcx_binary_path : Path | str, optional
        User-defined path, used to override the default installation directory.
    """
    version = _convert_and_validate_version(version)
    global _default_solc_binary
    _default_solc_binary = get_executable(version, solcx_binary_path)
    if not silent:
        LOGGER.info(f"Using solc version {version}")


def _select_pragma_version(pragma_string: str, version_list: List[Version]) -> Optional[Version]:
    pragma_string = re.sub(r"(\D)0+(\d)", r"\1\2", pragma_string)
    comparator_set_range = pragma_string.replace(" ", "").split("||")
    comparator_regex = re.compile(r"(([<>]?=?|\^)\d+\.\d+\.\d+)")
    version = None

    for comparator_set in comparator_set_range:
        spec = SimpleSpec(",".join((i[0] for i in comparator_regex.findall(comparator_set))))
        selected = spec.select(version_list)
        if selected and (not version or version < selected):
            version = selected

    return version

def _select_pragma_version_by_set(pragma_set: set, version_list: List[Version]) -> Optional[Version]:
    comparator_set_range = []
    for pragma_string in pragma_set:
        pragma_string = re.sub(r"(\D)0+(\d)", r"\1\2", pragma_string)
        comparator_set_range.extend(pragma_string.replace(" ", "").split("||"))
    comparator_regex = re.compile(r"(([<>]?=?|\^)\d+\.\d+\.\d+)")

    version = None
    find_versions = []
    for comparator_set in comparator_set_range:
        find_versions.extend([i[0] for i in comparator_regex.findall(comparator_set)])

    max_version = -1
    for v in find_versions:
        if v[0:2] in ['<0', '^0', '>0']:
            if max_version < int(v[3]):
                max_version = int(v[3])
        elif v[0:2] in ['>=', '<=']:
            if max_version < int(v[4]):
                max_version = int(v[4])
        elif v[0:2] == '0.':
            if max_version < int(v[2]):
                max_version = int(v[2])

    new_find_versions = []
    for v in find_versions:
        if v[0:2] == '>0' and int(v[3]) == max_version:
            new_find_versions.append('^' + v[1:])
        elif v[0:2] == '>=' and int(v[4]) == max_version:
            new_find_versions.append('^' + v[2:])
        else:
            new_find_versions.append(v)

    spec = SimpleSpec(",".join(new_find_versions))
    selected_set = list(spec.filter(version_list))
    selected_set = sorted(selected_set)
    if selected_set:
        if Version('0.8.11') in selected_set:
            version = Version('0.8.11')
        else:
            version = selected_set[-1]

    return version
    #     v_list = [i[0] for i in comparator_regex.findall(comparator_set)]
    #     for v in v_list:
    #         if v[0:2] == '0.':
    #             strict_version = True
    #     spec = SimpleSpec(",".join(v_list))
    #     selected_set = list(spec.filter(version_list))
    #     selected_set = sorted(selected_set)
    #     if selected_set:
    #         if Version('0.8.11') in selected_set:
    #             selected = Version('0.8.11')
    #         else:
    #             selected = selected_set[-1]
    #
    #     if strict_version:
    #         version = selected
    #         return version
    #
    #     if selected and (not version or version < selected):
    #         version = selected
    #
    # return version


def set_solc_version_pragma(
    pragma_string: str, silent: bool = False, check_new: bool = False
) -> Version:
    """
    Set the currently active `solc` binary based on a pragma statement.

    The newest installed version that matches the pragma is chosen. Raises
    `SolcNotInstalled` if no installed versions match.

    Arguments
    ---------
    pragma_string : str
        Pragma statement, e.g. "pragma solidity ^0.4.22;"
    silent : bool, optional
        If True, do not generate any logger output.
    check_new : bool, optional
        If True, also check if there is a newer compatible version that has not
        been installed.

    Returns
    -------
    Version
        The new active `solc` version.
    """
    version = _select_pragma_version(pragma_string, get_installed_solc_versions())
    if version is None:
        raise SolcNotInstalled(
            f"No compatible solc version installed."
            f" Use solcx.install_solc_version_pragma('{version}') to install."
        )
    set_solc_version(version, silent)
    if check_new:
        latest = install_solc_pragma(pragma_string, False)
        if latest > version:
            LOGGER.info(f"Newer compatible solc version exists: {latest}")

    return version

def install_solc_pragma_by_set(
    pragma_set: set,
    install: bool = True,
    show_progress: bool = False,
    solcx_binary_path: Union[Path, str] = None,
) -> Version:
    version = _select_pragma_version(pragma_set, get_installable_solc_versions())
    if not version:
        raise UnsupportedVersionError("Compatible solc version does not exist")
    if install:
        install_solc(version, show_progress=show_progress, solcx_binary_path=solcx_binary_path)

    return version


def install_solc_pragma(
    pragma_string: str,
    install: bool = True,
    show_progress: bool = False,
    solcx_binary_path: Union[Path, str] = None,
) -> Version:
    """
    Find, and optionally install, the latest compatible `solc` version based on
    a pragma statement.

    Arguments
    ---------
    pragma_string : str
        Pragma statement, e.g. "pragma solidity ^0.4.22;"
    install : bool, optional
        If True, installs the version of `solc`.
    show_progress : bool, optional
        If True, display a progress bar while downloading. Requires installing
        the `tqdm` package.
    solcx_binary_path : Path | str, optional
        User-defined path, used to override the default installation directory.

    Returns
    -------
    Version
        Installed `solc` version.
    """
    version = _select_pragma_version(pragma_string, get_installable_solc_versions())
    if not version:
        raise UnsupportedVersionError("Compatible solc version does not exist")
    if install:
        install_solc(version, show_progress=show_progress, solcx_binary_path=solcx_binary_path)

    return version


def install_solc_pragma_by_set(
    pragma_set: set,
    install: bool = True,
    show_progress: bool = False,
    solcx_binary_path: Union[Path, str] = None,
) -> Version:
    """
    Find, and optionally install, the latest compatible `solc` version based on
    a pragma statement.

    Arguments
    ---------
    pragma_string : str
        Pragma statement, e.g. "pragma solidity ^0.4.22;"
    install : bool, optional
        If True, installs the version of `solc`.
    show_progress : bool, optional
        If True, display a progress bar while downloading. Requires installing
        the `tqdm` package.
    solcx_binary_path : Path | str, optional
        User-defined path, used to override the default installation directory.

    Returns
    -------
    Version
        Installed `solc` version.
    """
    version = _select_pragma_version_by_set(pragma_set, get_installable_solc_versions(update=False))
    if not version:
        raise UnsupportedVersionError("Compatible solc version does not exist")
    if install:
        install_solc(version, show_progress=show_progress, solcx_binary_path=solcx_binary_path)

    return version


def get_installable_solc_versions(update=True) -> List[Version]:
    """
    Return a list of all `solc` versions that can be installed by py-solc-x.

    Returns
    -------
    List
        List of Versions objects of installable `solc` versions.
    """
    if update:
        data = requests.get(BINARY_DOWNLOAD_BASE.format(_get_os_name(), "list.json"))
        if data.status_code != 200:
            raise ConnectionError(
                f"Status {data.status_code} when getting solc versions from binaries.soliditylang.org"
            )
        version_list = sorted((Version(i) for i in data.json()["releases"]), reverse=True)
    else:
        data = ['0.8.18', '0.8.17', '0.8.16', '0.8.15', '0.8.14', '0.8.13', '0.8.12', '0.8.11', '0.8.10', '0.8.9', '0.8.8', '0.8.7', '0.8.6', '0.8.5', '0.8.4', '0.8.3', '0.8.2', '0.8.1', '0.8.0', '0.7.6', '0.7.5', '0.7.4', '0.7.3', '0.7.2', '0.7.1', '0.7.0', '0.6.12', '0.6.11', '0.6.10', '0.6.9', '0.6.8', '0.6.7', '0.6.6', '0.6.5', '0.6.4', '0.6.3', '0.6.2', '0.6.1', '0.6.0', '0.5.17', '0.5.16', '0.5.15', '0.5.14', '0.5.13', '0.5.12', '0.5.11', '0.5.10', '0.5.9', '0.5.8', '0.5.7', '0.5.6', '0.5.5', '0.5.4', '0.5.3', '0.5.2', '0.5.1', '0.5.0', '0.4.26', '0.4.25', '0.4.24', '0.4.23', '0.4.22', '0.4.21', '0.4.20', '0.4.19', '0.4.18', '0.4.17', '0.4.16', '0.4.15', '0.4.14', '0.4.13', '0.4.12', '0.4.11', '0.4.10', '0.4.9', '0.4.8', '0.4.7', '0.4.6', '0.4.5', '0.4.4', '0.4.3', '0.4.2', '0.4.1', '0.4.0', '0.3.6']
        version_list = sorted((Version(i) for i in data), reverse=True)
    version_list = [i for i in version_list if i >= MINIMAL_SOLC_VERSION]
    return version_list


def get_compilable_solc_versions(headers: Optional[Dict] = None) -> List[Version]:
    """
    Return a list of all `solc` versions that can be compiled from source by py-solc-x.

    Arguments
    ---------
    headers : Dict, optional
        Headers to include in the request to Github.

    Returns
    -------
    List
        List of Versions objects of installable `solc` versions.
    """
    if _get_target_os() == "windows":
        raise OSError("Compiling from source is not supported on Windows systems")

    version_list = []
    pattern = "solidity_[0-9].[0-9].[0-9]{1,}.tar.gz"

    if headers is None and os.getenv("GITHUB_TOKEN") is not None:
        auth = b64encode(os.environ["GITHUB_TOKEN"].encode()).decode()
        headers = {"Authorization": f"Basic {auth}"}

    data = requests.get(GITHUB_RELEASES, headers=headers)
    if data.status_code != 200:
        msg = (
            f"Status {data.status_code} when getting solc versions from Github:"
            f" '{data.json()['message']}'"
        )
        if data.status_code == 403:
            msg += (
                "\n\nIf this issue persists, generate a Github API token and store"
                " it as the environment variable `GITHUB_TOKEN`:\n"
                "https://github.blog/2013-05-16-personal-api-tokens/"
            )
        raise ConnectionError(msg)

    for release in data.json():
        try:
            version = Version.coerce(release["tag_name"].lstrip("v"))
        except ValueError:
            # ignore non-standard releases (e.g. the 0.8.x preview)
            continue

        asset = next((i for i in release["assets"] if re.match(pattern, i["name"])), False)
        if asset:
            version_list.append(version)
        if version == MINIMAL_SOLC_VERSION:
            break
    return sorted(version_list, reverse=True)


def get_installed_solc_versions(solcx_binary_path: Union[Path, str] = None) -> List[Version]:
    """
    Return a list of currently installed `solc` versions.

    Arguments
    ---------
    solcx_binary_path : Path | str, optional
        User-defined path, used to override the default installation directory.

    Returns
    -------
    List
        List of Version objects of installed `solc` versions.
    """
    install_path = get_solcx_install_folder(solcx_binary_path)
    return sorted([Version(i.name[6:]) for i in install_path.glob("solc-v*")], reverse=True)


def install_solc(
    version: Union[str, Version] = "latest",
    show_progress: bool = False,
    solcx_binary_path: Union[Path, str] = None,
) -> Version:
    """
    Download and install a precompiled version of `solc`.

    Arguments
    ---------
    version : str | Version, optional
        Version of `solc` to install. Default is the newest available version.
    show_progress : bool, optional
        If True, display a progress bar while downloading. Requires installing
        the `tqdm` package.
    solcx_binary_path : Path | str, optional
        User-defined path, used to override the default installation directory.

    Returns
    -------
    Version
        installed solc version
    """
    if version == "latest":
        version = get_installable_solc_versions()[0]
    else:
        version = _convert_and_validate_version(version)

    target_os = _get_target_os()
    this_os = _get_os_name()
    process_lock = get_process_lock('install'+str(version))

    # print('hahha')
    with process_lock:
        if _check_for_installed_version(version, solcx_binary_path):
            path = get_solcx_install_folder(solcx_binary_path).joinpath(f"solc-v{version}")
            LOGGER.info(f"solc {version} already installed at: {path}")
            return version

        data = requests.get(BINARY_DOWNLOAD_BASE.format(target_os, "list.json"))
        if data.status_code != 200:
            raise ConnectionError(
                f"Status {data.status_code} when getting solc versions from binaries.soliditylang.org"
            )
        try:
            filename = data.json()["releases"][str(version)]
        except KeyError:
            raise SolcInstallationError(f"Solc binary for v{version} is not available for this OS")
        if target_os == "linux":
            _install_solc_unix(version, filename, show_progress, solcx_binary_path)
        elif target_os == "macosx":
            _install_solc_unix(version, filename, show_progress, solcx_binary_path)
        elif target_os == "windows":
            _install_solc_windows(version, filename, show_progress, solcx_binary_path)

        try:
            _validate_installation(version, solcx_binary_path)
        except SolcInstallationError as exc:
            if target_os != "windows" and target_os == this_os:
                exc.args = (
                    f"{exc.args[0]} If this issue persists, you can try to compile from "
                    f"source code using `solcx.compile_solc('{version}')`.",
                )
            raise exc

    return version


def compile_solc(
    version: Version, show_progress: bool = False, solcx_binary_path: Union[Path, str] = None
) -> Version:
    """
    Install a version of `solc` by downloading and compiling source code.

    Arguments
    ---------
    version : str | Version, optional
        Version of `solc` to install. Default is the newest available version.
    show_progress : bool, optional
        If True, display a progress bar while downloading. Requires installing
        the `tqdm` package.
    solcx_binary_path : Path | str, optional
        User-defined path, used to override the default installation directory.

    Returns
    -------
    Version
        installed solc version
    """
    if _get_os_name() != _get_target_os():
        raise OSError("Cross-compiling is not supported")
    if _get_os_name() == "windows":
        raise OSError("Compiling from source is not supported on Windows systems")

    if version == "latest":
        version = get_compilable_solc_versions()[0]
    else:
        version = _convert_and_validate_version(version)

    process_lock = get_process_lock(str(version))

    with process_lock:
        if _check_for_installed_version(version, solcx_binary_path):
            path = get_solcx_install_folder(solcx_binary_path).joinpath(f"solc-v{version}")
            LOGGER.info(f"solc {version} already installed at: {path}")
            return version

        temp_path = _get_temp_folder()
        download = SOURCE_DOWNLOAD_BASE.format(version, f"solidity_{version}.tar.gz")
        install_path = get_solcx_install_folder(solcx_binary_path).joinpath(f"solc-v{version}")

        content = _download_solc(download, show_progress)
        with tarfile.open(fileobj=BytesIO(content)) as tar:
            tar.extractall(temp_path)
        temp_path = temp_path.joinpath(f"solidity_{version}")

        try:
            LOGGER.info("Running dependency installation script `install_deps.sh`...")
            subprocess.check_call(
                ["sh", temp_path.joinpath("scripts/install_deps.sh")], stderr=subprocess.DEVNULL
            )
        except subprocess.CalledProcessError as exc:
            LOGGER.warning(exc, exc_info=True)

        original_path = os.getcwd()
        temp_path.joinpath("build").mkdir(exist_ok=True)
        os.chdir(str(temp_path.joinpath("build").resolve()))
        try:
            for cmd in (["cmake", ".."], ["make"]):
                LOGGER.info(f"Running `{cmd[0]}`...")
                subprocess.check_call(cmd, stderr=subprocess.DEVNULL)
            temp_path.joinpath("build/solc/solc").rename(install_path)
        except subprocess.CalledProcessError as exc:
            err_msg = (
                f"{cmd[0]} returned non-zero exit status {exc.returncode}"
                " while attempting to build solc from the source.\n"
                "This is likely due to a missing or incorrect version of a build dependency."
            )
            if _get_target_os() == "macosx":
                err_msg = (
                    f"{err_msg}\n\nFor suggested installation options: "
                    "https://github.com/iamdefinitelyahuman/py-solc-x/wiki/Installing-Solidity-on-OSX"  # noqa: E501
                )
            raise SolcInstallationError(err_msg)

        finally:
            os.chdir(original_path)

        install_path.chmod(install_path.stat().st_mode | stat.S_IEXEC)
        _validate_installation(version, solcx_binary_path)

    return version


def _check_for_installed_version(
    version: Version, solcx_binary_path: Union[Path, str] = None
) -> bool:
    path = get_solcx_install_folder(solcx_binary_path).joinpath(f"solc-v{version}")
    return path.exists()


def _get_temp_folder() -> Path:
    path = Path(tempfile.gettempdir()).joinpath(f"solcx-tmp-{os.getpid()}")
    if path.exists():
        shutil.rmtree(str(path))
    path.mkdir()
    return path


def _download_solc(url: str, show_progress: bool) -> bytes:
    LOGGER.info(f"Downloading from {url}")
    response = requests.get(url, stream=show_progress)
    if response.status_code == 404:
        raise DownloadError(
            "404 error when attempting to download from {} - are you sure this"
            " version of solidity is available?".format(url)
        )
    if response.status_code != 200:
        raise DownloadError(
            f"Received status code {response.status_code} when attempting to download from {url}"
        )
    if not show_progress:
        return response.content

    total_size = int(response.headers.get("content-length", 0))
    progress_bar = tqdm(total=total_size, unit="iB", unit_scale=True)
    content = bytes()

    for data in response.iter_content(1024, decode_unicode=True):
        progress_bar.update(len(data))
        content += data
    progress_bar.close()

    return content


def _install_solc_unix(
    version: Version, filename: str, show_progress: bool, solcx_binary_path: Union[Path, str, None]
) -> None:
    download = BINARY_DOWNLOAD_BASE.format(_get_target_os(), filename)
    install_path = get_solcx_install_folder(solcx_binary_path).joinpath(f"solc-v{version}")

    content = _download_solc(download, show_progress)
    with open(install_path, "wb") as fp:
        fp.write(content)

    install_path.chmod(install_path.stat().st_mode | stat.S_IEXEC)


def _install_solc_windows(
    version: Version, filename: str, show_progress: bool, solcx_binary_path: Union[Path, str, None]
) -> None:
    download = BINARY_DOWNLOAD_BASE.format(_get_target_os(), filename)
    install_path = get_solcx_install_folder(solcx_binary_path).joinpath(f"solc-v{version}")

    temp_path = _get_temp_folder()
    content = _download_solc(download, show_progress)

    if Path(filename).suffix == ".exe":
        install_path.mkdir()
        with open(install_path.joinpath("solc.exe"), "wb") as fp:
            fp.write(content)

    else:
        with zipfile.ZipFile(BytesIO(content)) as zf:
            zf.extractall(str(temp_path))
            temp_path.rename(install_path)


def _validate_installation(version: Version, solcx_binary_path: Union[Path, str, None]) -> None:
    if _get_target_os() != _get_os_name():
        return
    binary_path = get_executable(version, solcx_binary_path)
    try:
        installed_version = wrapper._get_solc_version(binary_path)
    except Exception:
        _unlink_solc(binary_path)
        raise SolcInstallationError(
            "Downloaded binary would not execute, or returned unexpected output."
        )
    if installed_version.truncate() != version.truncate():
        _unlink_solc(binary_path)
        raise UnexpectedVersionError(
            f"Attempted to install solc v{version}, but got solc v{installed_version}"
        )
    if installed_version != version:
        warnings.warn(f"Installed solc version is v{installed_version}", UnexpectedVersionWarning)
    if not _default_solc_binary:
        set_solc_version(version)
    LOGGER.info(f"solc {version} successfully installed at: {binary_path}")


try:
    # try to set the result of `which`/`where` as the default
    _default_solc_binary = _get_which_solc()
except Exception:
    # if not available, use the most recent solcx installed version
    if get_installed_solc_versions():
        set_solc_version(get_installed_solc_versions()[0], silent=True)


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("version")
    argument_parser.add_argument("--solcx-binary-path", default=None)
    args = argument_parser.parse_args()
    install_solc(args.version, solcx_binary_path=args.solcx_binary_path)
