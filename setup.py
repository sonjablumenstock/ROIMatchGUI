from setuptools import setup, find_packages

setup(
    name='roimatch-gui',
    version='0.1.0',
    description='Interactive GUI for matching ROIs across imaging sessions',
    author='Sonja Blumenstock',
    packages=find_packages(),  # Automatically finds your roimatch_gui package
    include_package_data=True,
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'PyQt5',
        'opencv-python',
        'pandas',
        'scikit-image',
        'uuid',
    ],
    entry_points={
        'console_scripts': [
            'roimatchgui=roimatch_gui.gui.roi_match_gui:main'
        ],
    },
    python_requires='>=3.8',
)