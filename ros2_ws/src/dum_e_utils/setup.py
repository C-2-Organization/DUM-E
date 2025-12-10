from setuptools import find_packages, setup

package_name = 'dum_e_utils'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rokey',
    maintainer_email='inseonyu7@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'open_gripper = dum_e_utils.gripper_controller:open_gripper',
            'close_gripper = dum_e_utils.gripper_controller:close_gripper',
            'gripper_test = dum_e_utils.gripper_controller:main',
        ],
    },
)
