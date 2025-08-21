# Adapted from https://github.com/pybind/cmake_example/blob/master/setup.py
import os
import re
import sys
import platform
import subprocess
import importlib
from sysconfig import get_paths

import importlib
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install
from distutils.sysconfig import get_config_var
from distutils.version import LooseVersion

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir, build_with_cuda):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)
        self.build_with_cuda = build_with_cuda

class Build(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        # Auto-initialize git submodules if they exist but are empty
        repo_root = os.path.dirname(os.path.abspath(__file__))
        gitmodules_path = os.path.join(repo_root, '.gitmodules')
        if os.path.exists(gitmodules_path):
            pybind11_path = os.path.join(repo_root, 'pybind11', 'CMakeLists.txt')
            if not os.path.exists(pybind11_path):
                print("Initializing git submodules...")
                try:
                    subprocess.check_call(['git', 'submodule', 'update', '--init', '--recursive'], 
                                        cwd=repo_root)
                    print("Submodules initialized successfully")
                except subprocess.CalledProcessError as e:
                    print(f"Warning: Could not initialize submodules: {e}")
                except FileNotFoundError:
                    print("Warning: git not found, skipping submodule initialization")

        super().run()

    def build_extension(self, ext):
        if isinstance(ext, CMakeExtension):
            extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
            info = get_paths()
            include_path = info['include']
            cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                          '-DPYTHON_INCLUDE_PATH=' + include_path]

            cfg = 'Debug' if self.debug else 'Release'
            build_args = ['--config', cfg]

            if platform.system() == "Windows":
                cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir),
                               '-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
                if sys.maxsize > 2**32:
                    cmake_args += ['-A', 'x64']
                build_args += ['--', '/m']
            else:
                cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
                build_args += ['--', '-j8']

            if ext.build_with_cuda:
                cmake_args += ['-DDIFFVG_CUDA=1']
            else:
                cmake_args += ['-DDIFFVG_CUDA=0']

            env = os.environ.copy()
            env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                                  self.distribution.get_version())
            
            # Auto-detect and set environment variables if not already set
            if ext.build_with_cuda and platform.system() != "Windows":
                # Auto-detect CUDA installation if not set
                if 'CUDA_HOME' not in env:
                    cuda_paths = ['/usr/local/cuda-12.6', '/usr/local/cuda-12', '/usr/local/cuda', '/opt/cuda']
                    for cuda_path in cuda_paths:
                        if os.path.exists(os.path.join(cuda_path, 'bin', 'nvcc')):
                            env['CUDA_HOME'] = cuda_path
                            # Add CUDA bin to PATH if not already there
                            cuda_bin = os.path.join(cuda_path, 'bin')
                            if cuda_bin not in env.get('PATH', ''):
                                env['PATH'] = cuda_bin + ':' + env.get('PATH', '')
                            print(f"Auto-detected CUDA at: {cuda_path}")
                            break
                
                # Auto-detect compatible GCC if not set
                if 'CC' not in env and 'CXX' not in env:
                    for gcc_version in ['12', '11', '10']:
                        gcc_path = f'/usr/bin/gcc-{gcc_version}'
                        gxx_path = f'/usr/bin/g++-{gcc_version}'
                        if os.path.exists(gcc_path) and os.path.exists(gxx_path):
                            env['CC'] = gcc_path
                            env['CXX'] = gxx_path
                            print(f"Auto-selected GCC {gcc_version}: {gcc_path}, {gxx_path}")
                            break
            if not os.path.exists(self.build_temp):
                os.makedirs(self.build_temp)
            subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
            subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)
        else:
            super().build_extension(ext)

torch_spec = importlib.util.find_spec("torch")
tf_spec = importlib.util.find_spec("tensorflow")
packages = []
build_with_cuda = False
if torch_spec is not None:
    packages.append('pydiffvg')
    import torch
    if torch.cuda.is_available():
        build_with_cuda = True
if tf_spec is not None and sys.platform != 'win32':
    packages.append('pydiffvg_tensorflow')
    if not build_with_cuda:
        import tensorflow as tf
        if tf.test.is_gpu_available(cuda_only=True, min_cuda_compute_capability=None):
            build_with_cuda = True
if len(packages) == 0:
    print('Error: PyTorch or Tensorflow must be installed. For Windows platform only PyTorch is supported.')
    exit()
# Override build_with_cuda with environment variable
if 'DIFFVG_CUDA' in os.environ:
    build_with_cuda = os.environ['DIFFVG_CUDA'] == '1'

setup(name = 'diffvg',
      version = '0.0.1',
      install_requires = [
          "numpy",
          "svgpathtools",
          "svgwrite", 
          "cssutils",
          "numba",
          "torch-tools",
          "visdom",
          "scikit-image",
          # PyTorch/torchvision should be installed with CUDA support beforehand:
          # pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
          # Or: conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
      ],
      description = 'Differentiable Vector Graphics',
      ext_modules = [CMakeExtension('diffvg', '', build_with_cuda)],
      cmdclass = dict(build_ext=Build, install=install),
      packages = packages,
      zip_safe = False)
