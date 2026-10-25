# Tests for installing llama.cpp. Real downloads are hundreds of megabytes, so these build tiny
# pretend archives inside the test and "download" them from a local folder instead. That
# checks the whole install (fingerprints, unpacking, the final folder) without the internet.

import hashlib
import io
import tarfile

import pytest

from localtune import install
from localtune.install import InstallError

NVIDIA_SMI_HEADER = """
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.95.02    Driver Version: 581.57         CUDA Version: 13.0               |
+-----------------------------------------------------------------------------------------+
"""


@pytest.fixture(autouse=True)
def temporary_home(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALTUNE_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("PATH", "")


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
    assert install.parse_cuda_version("no version here") is None


def test_picks_a_build_the_driver_supports():
    assert install.pick_build((13, 0)) == "cuda-13.4"
    assert install.pick_build((12, 9)) == "cuda-12.8"
    assert install.pick_build((12, 4)) is None
    assert install.pick_build(None) is None


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
    assert (folder / "libllama.so").resolve() == (folder / "libllama.so.0").resolve()
    assert set(seen) == {"llama.tar.gz", "cudart.tar.gz"}


def test_installed_tools_are_found():
    assert install.find_tool("llama-bench") is None
    install.install_dir().mkdir(parents=True)
    (install.install_dir() / "llama-bench").write_text("")
    assert install.find_tool("llama-bench") == str(install.install_dir() / "llama-bench")


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
