from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension('update_func_approxap', ['update_func.pyx'])]
setup(
    name = 'update_func_approx_ap',
    cmdclass = {'build_ext':build_ext},
    ext_modules = ext_modules
)
