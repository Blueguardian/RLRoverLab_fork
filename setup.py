import itertools

from setuptools import find_packages, setup  # noqa: F401

INSTALL_REQUIRES = [
    # generic
    "numpy",
    "torch",
    "prettytable==3.3.0",
    "pymeshlab",
    "open3d",
    "gdown",
    # devices
    "hidapi",
    "skrl==1.4.1",
    "wandb",
    "opencv-python",
    "isaaclab==2.0.1",
    "ruamel.yaml",
    "torchvision",
    "stable_baselines3",
    "imitation"
]

# url=EXTENSION_TOML_DATA["package"]["repository"], # add later
# version=EXTENSION_TOML_DATA["package"]["version"],
# description=EXTENSION_TOML_DATA["package"]["description"],
# keywords=EXTENSION_TOML_DATA["package"]["keywords"],
EXTRAS_REQUIRE = {
    "rsl_rl": ["rsl-rl-lib @ git+https://github.com/leggedrobotics/rsl_rl.git"],
}


# cumulation of all extra-requires
EXTRAS_REQUIRE["all"] = list(itertools.chain.from_iterable(EXTRAS_REQUIRE.values()))
setup(
    name="omni.abmoRobotics.RLRoverLab",
    author="Anton Bjørndahl Mortensen",
    maintainer="Anton Bjørndahl Mortensen",
    maintainer_email="abmoRobotics@gmail.com",
    license="BSD-3-Clause",
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    packages=["rover_envs", "rover_il"],
    classifiers=["Natural Language :: English", "Programming Language :: Python :: 3.10"],
    zip_safe=False,
)
