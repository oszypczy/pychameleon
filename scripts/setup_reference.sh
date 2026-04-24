#!/usr/bin/env bash
# Re-clone the Moonpuck CHAMELEON reference and apply pychameleon patches.
#
# Produces external/chameleon_cluster_reference/ in a state identical to the
# one used to generate benchmarks/reference_moonpuck/*. Safe to re-run — it
# refuses to touch an already-patched working tree unless --force is given.
#
# Usage: bash scripts/setup_reference.sh [--force]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="$REPO_ROOT/external/chameleon_cluster_reference"
COMMIT="1c0a65ee6a79706e4d415dd7ca78da5d3c29906d"  # pinned; matches benchmarks meta.json

if [[ -d "$TARGET" && "${1:-}" != "--force" ]]; then
    echo "Reference already exists at $TARGET"
    echo "Pass --force to re-clone."
    exit 0
fi

rm -rf "$TARGET"
echo "Cloning Moonpuck/chameleon_cluster @ $COMMIT ..."
git clone --quiet https://github.com/Moonpuck/chameleon_cluster.git "$TARGET"
git -C "$TARGET" checkout --quiet "$COMMIT"

echo "Applying patches ..."

# Patch 1: metis -> pymetis + CSR adapter
python - "$TARGET/graphtools.py" <<'PY'
import pathlib, re, sys

path = pathlib.Path(sys.argv[1])
text = path.read_text()

text = text.replace("import metis\n", "import pymetis\n")

adapter = '''

def _part_graph(graph, nparts):
    """Adapter: partition a weighted networkx graph using pymetis."""
    nodes = list(graph.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    xadj = [0]
    adjncy = []
    eweights = []
    weight_attr = graph.graph.get("edge_weight_attr", "weight")
    for n in nodes:
        for nb in graph.neighbors(n):
            adjncy.append(node_to_idx[nb])
            eweights.append(int(graph[n][nb].get(weight_attr, 1)))
        xadj.append(len(adjncy))
    n_cuts, parts = pymetis.part_graph(
        nparts, xadj=xadj, adjncy=adjncy, eweights=eweights,
    )
    return n_cuts, parts


'''

text = text.replace("import pymetis\n\n\n", "import pymetis\n" + adapter, 1)

text = text.replace(
    "edgecuts, parts = metis.part_graph(\n        graph, 2, objtype='cut', ufactor=250)",
    "edgecuts, parts = _part_graph(graph, 2)",
)
text = text.replace(
    "edgecuts, parts = metis.part_graph(\n            s_graph, 2, objtype='cut', ufactor=250)",
    "edgecuts, parts = _part_graph(s_graph, 2)",
)
text = text.replace("metis.part_graph(graph, k)", "_part_graph(graph, k)")

path.write_text(text)
print(f"  patched {path.name}")
PY

# Patch 2: networkx >=2.4 API (graph.node -> graph.nodes)
for f in "$TARGET/graphtools.py" "$TARGET/chameleon.py"; do
    python - "$f" <<'PY'
import pathlib, re, sys
p = pathlib.Path(sys.argv[1])
t = p.read_text()
t = re.sub(r"(\bgraph|\bg|\bs_graph)\.node(\[| )", r"\1.nodes\2", t)
p.write_text(t)
print(f"  patched {p.name}")
PY
done

# Patch 3: requirements.txt metis -> pymetis
sed -i '' 's/^metis$/pymetis/' "$TARGET/requirements.txt"
echo "  patched requirements.txt"

# Set up venv
echo "Setting up venv ..."
cd "$TARGET"
uv venv --python 3.11 --quiet
uv pip install --quiet -r requirements.txt

echo ""
echo "Reference ready at: $TARGET"
echo "To verify: cd $TARGET && .venv/bin/python main.py"
