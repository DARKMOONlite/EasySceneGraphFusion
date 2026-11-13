"""Calculate per-graph counts and relationship percentages in a JSON folder tree."""

import json
from argparse import ArgumentParser
from pathlib import Path

from tqdm import tqdm


def calculate_stats(input_root: Path) -> dict:
    if not input_root.is_dir():
        raise NotADirectoryError(f"input folder '{input_root}' not found")

    graph_count = 0
    totals = {"entities": 0, "relationships": 0}
    maxima = dict(totals)
    percentage_total = 0.0
    percentage_minimum = float("inf")
    files = sorted(path for path in input_root.rglob("*.json") if path.is_file())
    for path in tqdm(files, desc="Counting scene graphs", unit="file"):
        try:
            with path.open(encoding="utf-8") as handle:
                data = json.load(handle)
            graphs = [data] if isinstance(data, dict) else data
            if not isinstance(graphs, list):
                raise ValueError("expected a graph object or a list of graphs")
            for graph in graphs:
                if not isinstance(graph, dict) or not (
                    "objects" in graph or "relationships" in graph
                ):
                    raise ValueError("expected a graph with objects or relationships")
                for name, key in (("entities", "objects"), ("relationships", "relationships")):
                    items = graph.get(key, [])
                    if not isinstance(items, list):
                        raise ValueError(f"'{key}' must be a list")
                    totals[name] += len(items)
                    maxima[name] = max(maxima[name], len(items))
                entities = len(graph.get("objects", []))
                possible_relationships = entities * (entities - 1)
                percentage = (
                    100 * len(graph.get("relationships", [])) / possible_relationships
                    if possible_relationships else 1
                )
                percentage_total += percentage
                percentage_minimum = min(percentage_minimum, percentage)
                graph_count += 1
        except (OSError, ValueError) as error:
            raise ValueError(f"{path}: {error}") from error

    return {
        "graphs": graph_count,
        **{
            name: {
                "average": totals[name] / graph_count if graph_count else 0.0,
                "maximum": maxima[name],
            }
            for name in totals
        },
        "relationship_percentage": {
            "average": percentage_total / graph_count if graph_count else 0.0,
            "minimum": percentage_minimum if graph_count else 0.0,
        },
    }


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("input_root", type=Path, help="Root folder containing scene graph JSON files.")
    args = parser.parse_args()
    try:
        stats = calculate_stats(args.input_root)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Scene graphs: {stats['graphs']}")
    for name in ("entities", "relationships"):
        print(f"{name.capitalize()}: average={stats[name]['average']:.2f}, maximum={stats[name]['maximum']}")
    percentage = stats["relationship_percentage"]
    print(f"Relationships as % of possible: average={percentage['average']:.2f}%, minimum={percentage['minimum']:.2f}%")


if __name__ == "__main__":
    main()
