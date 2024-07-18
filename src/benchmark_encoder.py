import argparse
import time
import warnings

import torch

warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_data():
    """
    Generate random data tensors for model input.
    """
    cube = torch.randn(128, 3, 256, 256).to(DEVICE)
    timestep = torch.randn(128, 4).to(DEVICE)
    latlon = torch.randn(128, 4).to(DEVICE)
    waves = torch.randn(3).to(DEVICE)
    gsd = torch.randn(1).to(DEVICE)
    return cube, timestep, latlon, waves, gsd


def load_exported_model(eager=True):
    """
    Load the exported model from a file.

    Args:
        eager (bool): Flag to decide whether to use eager mode or compiled mode.
    """
    print("Loading exported model")
    ep = torch.export.load("checkpoints/compiled/encoder.pt")
    if eager:
        model = ep.module()
    else:
        model = torch.compile(ep.module(), backend="inductor")
    return model


def benchmark_model(model):
    """
    Benchmark the model by running inference on randomly generated data.

    Args:
        model: The model to benchmark.
    """
    print("Benchmarking model")
    start = time.time()
    for i in range(20):
        cube, timestep, latlon, waves, gsd = get_data()
        with torch.inference_mode():
            out = model(cube, timestep, latlon, waves, gsd)
            print(
                f"Iteration {i}: Output shapes - {out[0].shape}, {out[1].shape}, {out[2].shape}, {out[3].shape}"  # noqa E501
            )
    print("Time taken for inference: ", time.time() - start)


def run(eager=True):
    """
    Run the exported model and benchmark it.

    Args:
        eager (bool): Flag to decide whether to use eager mode or compiled mode.
    """
    print("Running model")
    model = load_exported_model(eager=eager)
    benchmark_model(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run benchmark for the exported model."
    )
    parser.add_argument(
        "--eager", action="store_true", help="Use eager mode for running the model."
    )
    args = parser.parse_args()

    run(args.eager)
