
import platform
from importlib.metadata import PackageNotFoundError, version


def get_reproducibility_metadata() -> dict:
    """Collect environment details useful for reproducing a run."""
    packages = {}

    for package in (
        "llm-eval-benchmark",
        "transformers",
        "torch",
        "sentence-transformers",
        "evaluate",
    ):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None

    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "package_versions": packages,
    }
