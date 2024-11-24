from setuptools import find_packages, setup

package_name = 'camera_interface'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'torch', 'opencv-python'],
    zip_safe=True,
    maintainer='bhavya',
    maintainer_email='bhavya@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolo_model = camera_interface.yolo_interface:main',
                  ],
    },
)
