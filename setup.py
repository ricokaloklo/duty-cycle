import setuptools
from pathlib import Path

setuptools.setup(
    name="duty-cycle",
    version="0.0.1",
    description="Simulate the duty cycle of a gravitational-wave detector.",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    packages=[
        "duty_cycle",
    ],
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "torch",
        "gwpy",
    ],
    classifiers=[
        "License :: OSI Approved :: MIT License",
    ],
    python_requires='>=3.9',
)