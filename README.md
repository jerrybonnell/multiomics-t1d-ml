# multiomics-t1d-ml

* `stacking.py` for proposed multi-view ensemble model, as shown in paper.
* `supervised.py` testbed for supervised LOOCV experiments. Please see comments to toggle between experiment for real data and generation of null distribution.
* `generate_null.py` for collecting results from feature importance experiment (relies on `supervised.py`)
* `supervised_sig.py` extraction of consistent feature sets

`data` directory contains source parallel multi-omic T1D datasets. The directory also contains experimental results from one run of the feature importance experiment.
