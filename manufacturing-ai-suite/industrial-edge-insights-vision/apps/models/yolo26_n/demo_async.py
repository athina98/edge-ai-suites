# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Asynchronous inference demo for a model exported from Geti.

Uses OpenVINO Model API's AsyncPipeline to submit the sample image
asynchronously and retrieve the prediction once it is ready. The resulting
image with the overlaid predictions is saved to result.jpg.
"""

from __future__ import annotations

from model_api.pipelines import AsyncPipeline
from utils import load_image, load_model, visualise_result


def main() -> None:
    model = load_model()
    image = load_image()

    print("Running asynchronous inference...")
    pipeline = AsyncPipeline(model)
    pipeline.submit_data(image, id=0)
    pipeline.await_all()
    result, _meta = pipeline.get_result(0)
    print("Predictions:")
    print(result)

    visualise_result(image, result)


if __name__ == "__main__":
    main()

