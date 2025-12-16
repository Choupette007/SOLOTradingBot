from setuptools import setup, find_packages
import os

# Read requirements.txt
requirements_file = 'requirements.txt'
install_requires = []
if os.path.exists(requirements_file):
    with open(requirements_file, 'r') as f:
        install_requires = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="solana_trading_bot_bundle",
    version="0.1.0",
    packages=find_packages(include=['solana_trading_bot_bundle', 'solana_trading_bot_bundle.*']),
    install_requires=install_requires,
    author="Effie Choupette",
    author_email="effie_choupette@outlook.com",
    description="A Solana trading bot bundle",
    long_description=open('README.md').read() if os.path.exists('README.md') else '',
    long_description_content_type="text/markdown",
    python_requires=">=3.6",
    package_data={
        'solana_trading_bot_bundle': ['*.yaml', '*.txt'],
    },
    include_package_data=True,
)