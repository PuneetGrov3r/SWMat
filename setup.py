from distutils.core import setup
setup(
  name = 'SWMat',
  packages = ['SWMat'],
  version = '0.1.1',
  license='Apache License 2.0',
  description = 'A package for making stunning graphs/charts using matplotlib in just few lines of code!',
  author = 'Puneet Grover',
  author_email = 'grover.puneet1995@gmail.com',
  url = 'https://github.com/PuneetGrov3r/SWMat',
  download_url = 'https://github.com/PuneetGrov3r/SWMat/archive/v0.1.1-alpha.tar.gz',
  keywords = ['plot', 'visualization', 'data', 'big data', 'exploration', 'data exploration', 'communication', 'python', 'matplotlib', 'machine learning', 'data science'],
  install_requires=[
          'matplotlib',
          'numpy',
          'pandas'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering :: Visualization',
    'License :: OSI Approved :: Apache Software License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3 :: Only'
  ],
)
