from setuptools import setup

setup(
    name='tfx_airflow',
    version='0.1.0',
    packages=[''],
    url='',
    license='',
    author='BASA',
    author_email='basa@octo.com',
    description='',
    install_requires=['tfx>=0.21.1,<0.22',
                      'tensorflow>=2.1,<2.2',
                      'tensorboard>=2.1,<2.2',
                      'Flask==2.3.2'],
)
