from setuptools import setup, find_packages

setup(
    name             = "torchsummaryX",
    version          = "1.2.0",
    description      = "Improved visualization tool of torchsummary.",
    author           = "Namhyuk Ahn",
    author_email     = "nmhkahn@gmail.com",
    url              = "https://github.com/nmhkahn/torchsummaryX",
    packages         =["torchsummaryX"],
    install_requires = ["torch", "numpy", "pandas"],
)
