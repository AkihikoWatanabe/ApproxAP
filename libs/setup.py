from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_module = Extension(
    "update_func_approxap",
    ["update_func_single.pyx"],
    extra_compile_args=['-O3', '-ffast-math', '-march=native', '-fopenmp'],
    extra_link_args=['-fopenmp'],
)

setup(
    name = 'update_func app',
    cmdclass = {'build_ext': build_ext},
    ext_modules = [ext_module],
)
