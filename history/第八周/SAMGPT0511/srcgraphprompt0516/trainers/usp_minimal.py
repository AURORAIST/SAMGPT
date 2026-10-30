import sys


def run_usp_minimal(*args, **kwargs):
    main_module = sys.modules.get('__main__')
    legacy = getattr(main_module, '_run_usp_minimal_legacy', None) if main_module is not None else None
    if legacy is None:
        raise RuntimeError('run_usp_minimal is expected to be called from execute.py as the main module.')
    return legacy(*args, **kwargs)
