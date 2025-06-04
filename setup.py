from setuptools import setup, find_packages

requirements = [
    'numpy>=1.22',
    'matplotlib>=3.5',
    'psutil>=5.9',
    "torch>=2.1",
    "pyvista>=0.39",
]

setup(
    name="voxelsss",
    version="0.1.0",
    description="Voxel-based structure simulation solvers",
    author="Simon Daubner",
    author_email="s.daubner@imperial.ac.uk",
    packages=find_packages(),
    install_requires=requirements,
    license="MIT license",
    python_requires='>=3.9',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Image Processing',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Environment :: GPU',
        'Environment :: GPU :: NVIDIA CUDA',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    zip_safe=False,
)