# Environment layout

`../environment.yml` is the recommended lightweight environment for the HDR-SC
selection core, examples, and tests.

The files in this directory document optional detector-side dependencies. They
do not redistribute detector code and are not intended to override dependency
files provided by the detector maintainers. Separate environments are
recommended because detector repositories may pin incompatible NumPy, PyTorch,
CUDA, or FAISS versions.

Install the HDR-SC package into an existing detector environment with:

```bash
python -m pip install -e /path/to/HDR-SC
```

Then implement the four-function adapter described in `../docs/ADAPTERS.md`.

