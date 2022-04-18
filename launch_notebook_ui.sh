#!/bin/sh

PYANETI_HOME="$( cd "$( dirname "$0" )" && pwd )"

jupyter notebook --notebook-dir="$PYANETI_HOME" $@ &

sleep 5
echo ""
echo "To access UI for TESS, open pyaneti_extras/TIC_Pyaneti.ipynb in Jupyter."
