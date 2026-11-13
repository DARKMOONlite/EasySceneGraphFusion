"""Track an ordered sequence of scene graphs and compress it into one graph."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from PIL import Image

from EasySceneGraphFusion.cli.defaults import _DEFAULT_OBJECT_SYNONYMS, _DEFAULT_PREDICATE_SYNONYMS
from EasySceneGraphFusion.cli.overlay import image_for_graph
from EasySceneGraphFusion.fusion import Standardiser, load_scene_graph_json, save_scene_graph_json
from EasySceneGraphFusion.fusion.io import collect_scene_graph_files
from EasySceneGraphFusion.temporal import TemporalStabaliser


def main(argv: list[str] | None = None) -> None:
    parser = ArgumentParser(
        description="Track entities across consecutive scene graphs and save one compressed graph.",
    )
    parser.add_argument(
        "graphs", type=Path, nargs="+",
        help="JSON files in temporal order, or folders expanded in lexicographic path order. Use one sequence per run.",
    )
    parser.add_argument(
        "--images", type=Path, required=True,
        help="Folder containing matching images, located recursively by graph filename stem.",
    )
    parser.add_argument("-o", "--output", type=Path, default=Path("stabilised_graph.json"))
    parser.add_argument("--visualise", action="store_true", help="Display tracked images and the compressed graph.")
    args = parser.parse_args(argv)

    if not args.images.is_dir():
        parser.error(f"image folder not found: {args.images}")

    graph_paths: list[Path] = []
    for path in args.graphs:
        if path.is_dir():
            graph_paths.extend(collect_scene_graph_files(path))
        elif path.is_file() and path.suffix.lower() == ".json":
            graph_paths.append(path)
        else:
            parser.error(f"expected a scene graph JSON file or folder: {path}")
    if not graph_paths:
        parser.error("no scene graph JSON files found")
    if args.output.resolve() in {path.resolve() for path in graph_paths}:
        parser.error("output must not overwrite an input scene graph")

    try:
        image_paths = [image_for_graph(path, args.images) for path in graph_paths]
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    standardiser = Standardiser(
        object_syn_map=_DEFAULT_OBJECT_SYNONYMS.copy(),
        predicate_syn_map=_DEFAULT_PREDICATE_SYNONYMS.copy(),
    )
    graphs = []
    images = []
    for graph_path, image_path in zip(graph_paths, image_paths):
        graph = load_scene_graph_json(graph_path, source=str(graph_path))
        standardiser.standardise(graph)
        if any(obj.bbox is None or obj.bbox.area <= 0 for obj in graph.objects):
            parser.error(f"tracking requires a non-empty bounding box for every object: {graph_path}")
        graphs.append(graph)
        with Image.open(image_path) as image:
            # BoxMOT expects OpenCV's BGR channel order.
            images.append(np.asarray(image.convert("RGB"))[:, :, ::-1].copy())

    temporal_graph = TemporalStabaliser().mot_tracking(graphs, images, visualise=args.visualise)
    compressed_graph = temporal_graph.compress()
    save_scene_graph_json(compressed_graph, args.output)
    print(
        f"Stabilised {temporal_graph.num_frames} frames with {temporal_graph.num_links} entity tracks "
        f"into {args.output}"
    )
    if args.visualise:
        compressed_graph.visualise()


if __name__ == "__main__":
    main()
