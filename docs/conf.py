"""Sphinx configuration for the chameleon-cluster documentation."""
from __future__ import annotations

from importlib.metadata import version as _dist_version

project = "chameleon-cluster"
author = "Oliwier Szypczyn"
copyright = f"2026, {author}"
release = _dist_version("chameleon-cluster")
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "numpydoc",
    "matplotlib.sphinxext.plot_directive",
]

# The Polish course write-up and local-only materials live next to the Sphinx
# sources but are not part of the published site.
exclude_patterns = ["_build", "documentation", "presentation", "*.pdf"]

autodoc_typehints = "none"  # numpydoc docstrings already document types
numpydoc_show_class_members = False
numpydoc_xref_param_type = True
numpydoc_xref_ignore = {"of", "shape", "default", "or", "optional", "Ignored"}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

plot_include_source = True
plot_html_show_source_link = False
plot_html_show_formats = False
plot_formats = [("png", 110)]

html_theme = "pydata_sphinx_theme"
html_title = f"chameleon-cluster {release}"
html_theme_options = {
    "github_url": "https://github.com/oszypczy/pychameleon",
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/chameleon-cluster/",
            "icon": "fa-brands fa-python",
        }
    ],
    "navbar_align": "left",
    "secondary_sidebar_items": ["page-toc"],
}
html_context = {"default_mode": "auto"}
