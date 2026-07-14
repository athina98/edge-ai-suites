# Geti exported model

This archive contains a model exported from Geti, along with a couple of
ready-to-run inference demos.

## Contents

| File | Description |
| ---- | ----------- |
| `model.xml` (+ `model.bin` for OpenVINO IR) | The exported model weights. |
| `image.jpg` (optional) | Sample input image from the project's dataset, kept in its original format. |
| `demo.py` | Minimal **synchronous** inference example. |
| `demo_async.py` | Minimal **asynchronous** inference example. |
| `utils.py` | Shared utility functions for loading the model/image and visualising the results. |
| `pyproject.toml` | Python dependencies required by the demos. |
| `README.md` | This file. |

The image may be omitted if no image is available. If `image.jpg` is missing, copy any image into this directory 
and name it `image.jpg` (or edit the demos to point to a different file).
 
## Setup

The recommended way to set up a clean environment is with
[`uv`](https://docs.astral.sh/uv/) - a fast Python package manager.

### Option 1 - one-shot with `uv`

This will create and activate your venv, then run the script immediately.

```bash
# From the directory where this README lives
uv run demo.py
uv run demo_async.py
```

`uv run` will transparently create a virtual environment, install the
dependencies, and execute the script. You will not remain in the virtual 
environment after the script executes.

### Option 2 - create a persistent virtual environment, then activate it

```bash
# Create and activate a virtual environment (Python 3.10+)
uv sync
# Linux / macOS
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

## Running the demos

Once the environment is ready and activated, simply run:

```bash
# Synchronous inference - writes the annotated result to result.jpg
python demo.py

# Asynchronous inference - writes the annotated result to result_async.jpg
python demo_async.py
```

Both scripts load `image.jpg`, run inference on it with OpenVINO Model API and
save an output image with the predicted bounding boxes / labels / masks
overlaid on top.

## Notes

* The demos default to running on CPU. To run on a different device (e.g. an
  Intel GPU), edit the scripts and pass device="GPU" to
  Model.create_model.
* For ONNX models, OpenVINO Model API reads the `.onnx` file directly - no
  additional conversion is required.
* These demos are intentionally minimal. For production deployment, refer to
  the [OpenVINO Model API documentation](https://github.com/open-edge-platform/model_api).


## Licensing

Licensing Information: Ultralytics YOLO models are distributed under the
AGPL-3.0 license, an OSI approved license ideal for open-source research,
academic, and personal projects. For commercial use, enhanced support, and
tailored licensing terms, please explore flexible Ultralytics licensing
options at https://www.ultralytics.com/license.
