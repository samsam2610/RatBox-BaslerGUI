from setuptools import setup, find_packages

# Read version from a single source if you keep it in a module
__version__ = "1.0.0"

setup(
    name="BaslerGUI",
    version=__version__,
    author="Sam Tran",
    author_email="sam@example.com",
    description="Basler camera control and image acquisition GUI using wxPython and pypylon.",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests", "docs"]),
    python_requires=">=3.8,<3.11",
    install_requires=[
        # core GUI and camera control
        "wxPython>=4.2.1",
        "pypylon",
        # scientific and image processing stack
        "numpy>=1.23",
        "opencv-python-headless>=4.8",
        "scikit-image>=0.21",
        "matplotlib>=3.7",
        "pillow>=10.0",
    ],
    entry_points={
        "gui_scripts": [
            "basler-gui=BaslerGUI.baslerGUI:main",  # expects a main() in baslerGUI.py
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Software Development :: User Interfaces",
    ],
)
