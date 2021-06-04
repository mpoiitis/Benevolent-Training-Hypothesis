from setuptools import setup

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='Steady Learner Hypothesis',
    url='https://anonymous.4open.science/r/steady-learner-hypothesis-00FC',
    author='XXXX',
    author_email='XXXX',
    # Needed to actually package something
    packages=['steadylearner'],
    # Needed for dependencies
    install_requires=['numpy', 'scipy', 'seaborn', 'matplotlib', 'pandas', 'tqdm', 'torch==1.8.1+cu111', 'torchvision==0.9.1+cu111', 'torchaudio===0.8.1'],
    entry_points={
        'console_scripts': ['steadylearner=steadylearner.command_line:main'],
    },
    # *strongly* suggested for sharing
    version='0.1',
    # The license can be anything you like
    license='GNUv3',
    description='Source code of paper: Steady Learner - What trainining reveals about neural network complexity',
    # We will also need a readme eventually (there will be a warning)
    long_description=open('README.rst').read(),
)