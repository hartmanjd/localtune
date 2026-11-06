# Tests for installing llama.cpp. Real downloads are hundreds of megabytes, so these build tiny
# pretend archives inside the test and "download" them from a local folder instead. That
# checks the whole install (fingerprints, unpacking, the final folder) without the internet.

import hashlib
import io
import tarfile
import zipfile

import pytest

from localtune import install
from localtune.install import InstallError

NVIDIA_SMI_HEADER = """
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.95.02    Driver Version: 581.57         CUDA Version: 13.0               |
+-----------------------------------------------------------------------------------------+
"""


NEWER_NVIDIA_SMI_HEADER = """
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 615.65.06              KMD Version: 616.56        CUDA UMD Version: 13.4     |
+-----------------------------------------------------------------------------------------+
"""


@pytest.fixture(autouse=True)
def temporary_home(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALTUNE_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("PATH", "")
    monkeypatch.delenv("LOCALTUNE_LLAMA_DIR", raising=False)


def make_archive(path, files, links=()):
    with tarfile.open(path, "w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(f"top/{name}")
            info.size = len(data)
            info.mode = 0o755
            tar.addfile(info, io.BytesIO(data))
        for name, target in links:
            info = tarfile.TarInfo(f"top/{name}")
            info.type = tarfile.SYMTYPE
            info.linkname = target
            tar.addfile(info)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_zip(path, files):
    # Like the Windows downloads: the files sit directly in the archive, with no top folder.
    with zipfile.ZipFile(path, "w") as zip_file:
        for name, data in files.items():
            zip_file.writestr(name, data)
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def release(tmp_path, monkeypatch):
    # A pretend release with the two downloads one build needs.
    folder = tmp_path / "release"
    folder.mkdir()
    llama = make_archive(
        folder / "llama.tar.gz",
        {"llama-bench": b"bench", "libllama.so.0": b"lib"},
        links=[("libllama.so", "libllama.so.0")],
    )
    cudart = make_archive(folder / "cudart.tar.gz", {"libcudart.so.13": b"cuda"})
    monkeypatch.setitem(install.DOWNLOADS, "test", {"llama.tar.gz": llama, "cudart.tar.gz": cudart})
    return folder.as_uri()


def test_reads_the_drivers_cuda_version():
    assert install.parse_cuda_version(NVIDIA_SMI_HEADER) == (13, 0)
    assert install.parse_cuda_version(NEWER_NVIDIA_SMI_HEADER) == (13, 4)
    assert install.parse_cuda_version("no version here") is None


def test_picks_a_build_the_driver_supports():
    assert install.pick_build((13, 0)) == "linux-cuda-13.4"
    assert install.pick_build((12, 9)) == "linux-cuda-12.8"
    assert install.pick_build((12, 4)) is None
    assert install.pick_build(None) is None


def test_picks_a_windows_build_the_driver_supports():
    assert install.pick_build((13, 4), "windows") == "windows-cuda-13.4"
    assert install.pick_build((12, 8), "windows") == "windows-cuda-12.4"
    assert install.pick_build((12, 4), "windows") == "windows-cuda-12.4"
    assert install.pick_build((12, 2), "windows") is None


def test_unsupported_computers_are_told_why(monkeypatch):
    monkeypatch.setattr(install.platform, "system", lambda: "Darwin")
    with pytest.raises(InstallError, match="only works on Windows, Linux and WSL"):
        install.choose_build()


def test_every_build_has_fingerprints_and_a_size():
    for build, files in install.DOWNLOADS.items():
        assert len(files) == 2
        assert all(len(sha256) == 64 for sha256 in files.values())
        assert build in install.DOWNLOAD_MB


def test_install_puts_everything_in_one_folder(release):
    seen = []
    folder = install.install_llama_cpp("test", lambda name, done, total: seen.append(name), release)
    assert folder == install.install_dir()
    assert (folder / "llama-bench").read_bytes() == b"bench"
    assert (folder / "libcudart.so.13").exists()
    assert (folder / "libllama.so").read_bytes() == (folder / "libllama.so.0").read_bytes()
    assert set(seen) == {"llama.tar.gz", "cudart.tar.gz"}


def test_a_link_windows_refuses_becomes_a_copy(release, monkeypatch):
    def refused(self, target, target_is_directory=False):
        raise OSError("A required privilege is not held by the client")

    monkeypatch.setattr(install.Path, "symlink_to", refused)
    folder = install.install_llama_cpp("test", lambda name, done, total: None, release)
    assert not (folder / "libllama.so").is_symlink()
    assert (folder / "libllama.so").read_bytes() == (folder / "libllama.so.0").read_bytes()


def test_install_unpacks_windows_downloads(tmp_path, monkeypatch):
    folder = tmp_path / "release"
    folder.mkdir()
    llama = make_zip(folder / "llama.zip", {"llama-bench.exe": b"bench", "llama.dll": b"lib"})
    cudart = make_zip(folder / "cudart.zip", {"cudart64_13.dll": b"cuda"})
    monkeypatch.setitem(install.DOWNLOADS, "test", {"llama.zip": llama, "cudart.zip": cudart})
    folder = install.install_llama_cpp("test", base_url=folder.as_uri())
    assert (folder / "llama-bench.exe").read_bytes() == b"bench"
    assert (folder / "cudart64_13.dll").exists()


@pytest.mark.parametrize(
    "system, program", [("Linux", "llama-bench"), ("Windows", "llama-bench.exe")]
)
def test_installed_tools_are_found(monkeypatch, system, program):
    monkeypatch.setattr(install.platform, "system", lambda: system)
    assert install.find_tool("llama-bench") is None
    install.install_dir().mkdir(parents=True)
    (install.install_dir() / program).write_text("")
    assert install.find_tool("llama-bench") == str(install.install_dir() / program)


# A llama.cpp someone built or keeps themselves, named in LOCALTUNE_LLAMA_DIR, is used before
# the PATH and the installed copy; a program that folder doesn't have is still found elsewhere.


def own_build(tmp_path, *programs):
    folder = tmp_path / "my-llama" / "build" / "bin"
    folder.mkdir(parents=True)
    for program in programs:
        (folder / program).write_text("")
    return folder


@pytest.mark.parametrize(
    "system, program", [("Linux", "llama-bench"), ("Windows", "llama-bench.exe")]
)
def test_own_llama_cpp_folder_comes_first(tmp_path, monkeypatch, system, program):
    monkeypatch.setattr(install.platform, "system", lambda: system)
    install.install_dir().mkdir(parents=True)
    (install.install_dir() / program).write_text("")
    on_path = tmp_path / "on-path"
    on_path.mkdir()
    (on_path / program).write_text("")
    (on_path / program).chmod(0o755)
    monkeypatch.setenv("PATH", str(on_path))
    folder = own_build(tmp_path, program)
    monkeypatch.setenv("LOCALTUNE_LLAMA_DIR", str(folder))
    assert install.find_tool("llama-bench") == str(folder / program)


def test_own_llama_cpp_folder_can_start_with_home(tmp_path, monkeypatch):
    monkeypatch.setattr(install.platform, "system", lambda: "Linux")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    folder = own_build(tmp_path, "llama-bench")
    monkeypatch.setenv("LOCALTUNE_LLAMA_DIR", "~/my-llama/build/bin")
    assert install.find_tool("llama-bench") == str(folder / "llama-bench")


def test_a_program_missing_from_own_folder_is_found_elsewhere(tmp_path, monkeypatch):
    monkeypatch.setattr(install.platform, "system", lambda: "Linux")
    folder = own_build(tmp_path, "llama-bench")
    monkeypatch.setenv("LOCALTUNE_LLAMA_DIR", str(folder))
    assert install.find_tool("llama-perplexity") is None
    install.install_dir().mkdir(parents=True)
    (install.install_dir() / "llama-perplexity").write_text("")
    assert install.find_tool("llama-perplexity") == str(install.install_dir() / "llama-perplexity")


def test_a_saved_folder_is_used_unless_one_is_named(tmp_path, monkeypatch):
    monkeypatch.setattr(install.platform, "system", lambda: "Linux")
    saved = own_build(tmp_path, "llama-bench")
    install.save_llama_dir(saved)
    assert install.find_tool("llama-bench") == str(saved.resolve() / "llama-bench")
    named = tmp_path / "other"
    named.mkdir()
    (named / "llama-bench").write_text("")
    monkeypatch.setenv("LOCALTUNE_LLAMA_DIR", str(named))
    assert install.find_tool("llama-bench") == str(named / "llama-bench")


@pytest.mark.parametrize(
    "system, build, program",
    [
        ("Linux", ("build", "bin"), "llama-bench"),
        ("Windows", ("build", "bin", "Release"), "llama-bench.exe"),
    ],
)
def test_source_builds_are_found_in_usual_places(tmp_path, monkeypatch, system, build, program):
    monkeypatch.setattr(install.platform, "system", lambda: system)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    assert install.find_own_builds() == []
    folder = tmp_path.joinpath("code", "llama.cpp", *build)
    folder.mkdir(parents=True)
    (folder / program).write_text("")
    assert install.find_own_builds() == [folder]


def test_a_damaged_download_is_refused(release, monkeypatch):
    monkeypatch.setitem(install.DOWNLOADS["test"], "llama.tar.gz", "0" * 64)
    with pytest.raises(InstallError, match="didn't download correctly"):
        install.install_llama_cpp("test", base_url=release)
    assert not install.install_dir().exists()


@pytest.mark.parametrize(
    "name, link", [("../escape", None), ("evil-link", "/etc/passwd"), ("up-link", "../../x")]
)
def test_unsafe_archives_are_refused(tmp_path, name, link):
    archive = tmp_path / "bad.tar.gz"
    if link:
        make_archive(archive, {}, links=[(name, link)])
    else:
        make_archive(archive, {name: b"x"})
    with pytest.raises(InstallError, match="unsafe"):
        install.unpack(archive, tmp_path / "out")


@pytest.mark.parametrize("name", ["../escape.dll", "/top.dll", "C:/Windows/evil.dll", "..\\up.dll"])
def test_unsafe_zip_archives_are_refused(tmp_path, name):
    archive = tmp_path / "bad.zip"
    make_zip(archive, {name: b"x"})
    with pytest.raises(InstallError, match="unsafe"):
        install.unpack(archive, tmp_path / "out")
