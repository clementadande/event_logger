from setuptools import setup, find_packages

setup(
    name="event_logger",
    version="0.1.0",
    package_dir={"": "src"},    # <--- THIS IS THE CRITICAL LINE
    packages=find_packages(where="src"),
)
