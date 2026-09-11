import argparse
from pathlib import Path
from torchview import draw_graph
from src.utils import init_model, setup_logging

_logger = setup_logging(__name__)


def print_model_layers(
    model_name: str,
    save_folder: Path,
    depth: int = 3,
):
    """Load a model and print its leaf layers formatted as in FeatureDecoder._register_hooks."""
    _logger.info(f"Loading model: <{model_name}>...")
    model = init_model(model_name, pretrained=False).cpu()

    model_graph = draw_graph(
        model,
        input_size=(1, *model.pretrained_cfg['input_size']),  # type: ignore
        expand_nested=True,
        depth=depth,
        save_graph=True,
    )

    path = model_graph.visual_graph.render(
        f"{model_name}_depth-{depth}", directory=save_folder, format="png"
    )
    _logger.info(f"Network plotted at depth {depth}. File saved to {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract and print model leaf layers in the format expected by FeatureDecoder."
    )
    parser.add_argument(
        "--model",
        "--models",
        type=str,
        nargs="+",
        dest="models",
        required=True,
        help="One or more model architecture names (e.g. resnet50s.gluon_in1k)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=None,
        help="Max hierarchy depth to display (e.g. 2 or 3 to inspect macro blocks)",
    )
    parser.add_argument(
        "--save_folder",
        type=str,
        default="plots/architectures",
        help="Folder where to save the architecture plot",
    )
    args = parser.parse_args()
    save_folder = Path(args.save_folder)
    save_folder.mkdir(exist_ok=True)

    for model_name in args.models:
        print_model_layers(model_name=model_name, depth=args.depth, save_folder=save_folder)


if __name__ == "__main__":
    main()

