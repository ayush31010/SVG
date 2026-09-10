# New-collection part-grounding holdout v1

100 authored edit instructions over 50 original SVGs, 25 each from
[Feather](https://github.com/feathericons/feather) and
[Tabler](https://github.com/tabler/tabler-icons). Neither collection is used by
the procedural training generator or the previous Hicon/Streetmix evaluation.
Upstream commit IDs are recorded in `upstream.json`; MIT license notices are
included beside the original source SVGs.

Every label specifies editable DOM nodes, not raster segments. Node IDs follow
`svgpatchlab.core.xml.index_tree`, so a path containing two disconnected parts
still counts as one target. The source files are unmodified. Two distinct
instructions share each source; uncertainty estimates must therefore group by
source, not treat all 100 cases as independent.

The agent authored the instructions after inspecting source elements and
reviewed **all 100** target highlights in `review-01.png` through `review-10.png`
before any holdout scoring. This is agent-reviewed annotation, not independent
human annotation. It is a curated part-selection test, not a random sample of
real user edits. The reviewed manifest is frozen by SHA-256 in the experiment
configuration before training and evaluation.

All instructions request changing selected strokes to red. This experiment
scores target selection only. It does not claim operation compilation or
rendered edit accuracy. Some targets exceed the original six-node cardinality
limit; retain those cases and report them rather than silently filtering them.

Regenerate review sheets with `python -m scripts.render_group_holdout_review`
after installing `resvg-py` and Pillow. Gold highlights are red; other nodes are
gray. The sheets contain no model predictions.
