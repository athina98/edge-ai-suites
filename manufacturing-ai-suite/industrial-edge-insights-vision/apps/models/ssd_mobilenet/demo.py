# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Synchronous inference demo for a model exported from Geti.

Loads a sample image, runs inference with OpenVINO Model API, then saves an
output image with the overlaid predictions to result.jpg.
"""
from __future__ import annotations

from utils import load_image, load_model, visualise_result


def main() -> None:
    model = load_model()
    image = load_image()

    print("Running synchronous inference...")
    result = model(image)
    print("Predictions:")
    print(result)

    visualise_result(image, result)


if __name__ == "__main__":
    main()

