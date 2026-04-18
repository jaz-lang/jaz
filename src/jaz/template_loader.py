from jinja2 import Environment, PackageLoader

_jinja_env = Environment(
    loader=PackageLoader("jaz", "templates"),
    autoescape=False,
    keep_trailing_newline=True,
    trim_blocks=True,
    lstrip_blocks=True,
)

# Pre-compile all templates at import time.
_loader = _jinja_env.loader
if _loader is not None:
    for template_name in _loader.list_templates():
        _jinja_env.get_template(template_name)
