"""
Local proxy for the real `fastapi` package.

This file exists in the workspace and would otherwise shadow the installed
`fastapi` package. Instead of deleting it, we proxy to the installed package
from site-packages so `from fastapi import FastAPI` continues to work.

If you prefer, delete or rename this file and install `fastapi` in your
environment (`pip install fastapi uvicorn`).
"""
import importlib.util
import importlib.machinery
import site
import os

def _load_installed_fastapi():
	# Search common site-packages locations
	candidates = []
	try:
		candidates += site.getsitepackages()
	except Exception:
		pass
	try:
		candidates.append(site.getusersitepackages())
	except Exception:
		pass
	# Also include paths from sysconfig (works for virtualenvs)
	try:
		import sysconfig
		cfg = sysconfig.get_paths()
		for key in ('purelib', 'platlib'):
			p = cfg.get(key)
			if p:
				candidates.append(p)
	except Exception:
		pass
	# And include the venv site-packages using sys.prefix (Windows/Unix)
	try:
		venv_site = os.path.join(sys.prefix, 'Lib', 'site-packages')
		candidates.append(venv_site)
	except Exception:
		pass

	for base in candidates:
		base = os.path.abspath(base)
		pkg_init = os.path.join(base, 'fastapi', '__init__.py')
		if os.path.exists(pkg_init):
			# Load the installed package under the canonical name 'fastapi'
			spec = importlib.util.spec_from_file_location('fastapi', pkg_init)
			module = importlib.util.module_from_spec(spec)
			# Ensure relative imports inside the package resolve to 'fastapi'
			import sys as _sys
			_sys.modules['fastapi'] = module
			spec.loader.exec_module(module)
			return module
	return None

_fastapi_mod = _load_installed_fastapi()
if _fastapi_mod is None:
	raise ImportError('Installed fastapi package not found in site-packages. Install with `pip install fastapi`.')

# Re-export attributes from the installed fastapi package
for _name, _val in _fastapi_mod.__dict__.items():
	if not _name.startswith('__'):
		globals()[_name] = _val
