import setuptools
from pathlib import Path
import glob

setuptools.setup(
    name="duty-cycle",
    version="0.1.1",
    description="Simulate duty cycles of a gravitational-wave detector network",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    packages=[
        "duty_cycle",
    ],
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "matplotlib",
        "torch",
        "gwpy",
        "sbi",
        "gwsumm",
        "bilby",
        "KDEpy",
        "obspy",
    ],
    package_data={
        "duty_cycle": [
            'data/detector_loc.json',
            'data/o1_H1_times.dat',
            'data/o1_L1_times.dat',
            'data/o2_H1_times.dat',
            'data/o2_L1_times.dat',
            'data/o2_V1_times.dat',
            'data/o3a_H1_times.dat',
            'data/o3a_L1_times.dat',
            'data/o3a_V1_times.dat',
            'data/o3b_H1_times.dat',
            'data/o3b_L1_times.dat',
            'data/o3b_V1_times.dat',
            'data/o3gk_G1_times.dat',
            'data/o3gk_K1_times.dat',
            'data/o4a_H1_times.dat',
            'data/o4a_L1_times.dat',
        ]
    },
    classifiers=[
        "License :: OSI Approved :: MIT License",
    ],
    python_requires='>=3.9',
)
