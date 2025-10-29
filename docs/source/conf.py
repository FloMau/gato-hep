import sys, os, datetime
from pathlib import Path

# make sure our package is importable
sys.path.insert(0, os.path.abspath("../.."))   # now src/ is on PYTHONPATH
# sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..", "src")))

# load tomllib or tomli
if sys.version_info < (3, 11):
    import tomli as tomllib
else:
    import tomllib

# this file is at <project>/docs/source/conf.py
# so go up three levels: source → docs → project root
pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

project = pyproject["project"]["name"]
author  = pyproject["project"]["authors"][0]["name"]
release = pyproject["project"]["version"]

# (optionally)
python_requires = pyproject["project"]["requires-python"]
rst_epilog = f".. |python_requires| replace:: {python_requires}"


# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
]


html_theme = "pydata_sphinx_theme"
here = Path(__file__).parent
staticfolder = here / "_static"

html_static_path = ["_static", "logos"]
html_theme_options = {
    "logo": {
        "image_light": "logos/gato-hep.png",
        "image_dark": "logos/gato-hep-darkmode.png",
    }
}

project = pyproject["project"]["name"]
author = pyproject["project"]["authors"][0]["name"]
copyright = "{}. Last updated {}".format(
author, datetime.datetime.now().strftime("%d %b %Y %H:%M")
)
python_requires = pyproject["project"]["requires-python"]
rst_epilog = f"""
.. |python_requires| replace:: {python_requires}
"""
