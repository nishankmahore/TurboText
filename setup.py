from setuptools import Extension, setup

try:
    from Cython.Build import cythonize

    ext_modules = cythonize(
        [Extension("turbotext._fast", ["src/turbotext/_fast.pyx"])],
        language_level=3,
        compiler_directives={
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
        },
    )
except ImportError:
    # Cython not available — compile from the pre-generated C file bundled in the sdist.
    ext_modules = [Extension("turbotext._fast", ["src/turbotext/_fast.c"])]

setup(ext_modules=ext_modules)
