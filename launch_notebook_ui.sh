#!/bin/sh

PYANETI_HOME="$( cd "$( dirname "$0" )" && pwd )"

echo "To access UI for TESS, open pyaneti_extras/TIC_Pyaneti.ipynb in Jupyter."
echo

jupyter notebook --notebook-dir="$PYANETI_HOME" $@

