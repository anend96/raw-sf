# raw2sf – uGMRT Raw to Search-Mode PSRFITS Converter

## Overview

`raw2sf` converts uGMRT raw voltage/filterbank data into **search-mode PSRFITS** format for pulsar and FRB analysis.

It is designed as part of the **GMRT-FRB polarization pipeline** and supports:

- Lower / Upper sideband selection
- Circular / Linear feeds
- Parallel block processing
- Automatic PSRFITS header generation
- Redigitization to 8-bit search-mode format

---

## Features

- Reads uGMRT `.raw` files via memory mapping
- Uses external header (`.hdr`) metadata
- Generates fully compliant **PSRFITS search-mode** files
- Parallel chunk processing using `multiprocessing`
- Frequency, beam, and source metadata injection
- Automatic Galactic coordinate conversion
- Configurable block size (`--gulp`)

---

## Requirements

Install the required Python packages:

```bash
pip install numpy tqdm fitsio
