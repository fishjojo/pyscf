#!/usr/bin/env bash
python -m pip install --upgrade pip
pip install "numpy!=1.16,!=1.17" "scipy<=1.1" h5py pytest pytest-cov pytest-timer
pip install pyberny geometric
pip install jaxlib jax

#cppe
version=$(python -c 'import sys; version=sys.version_info[:2]; print("{0}.{1}".format(*version))')
if [ $version != '2.7' ] && [ $version != '3.5' ]; then
    pip install cppe
fi
