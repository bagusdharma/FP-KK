from cx_Freeze import setup, Executable
import os.path
PYTHON_INSTALL_DIR = os.path.dirname(os.path.dirname(os.__file__))

base = None

executables = [Executable("FinalProject.py", base=base)]

packages = ["idna"]
options = {
    'build_exe': {
        'packages':packages,
    },
}

os.environ['TCL_LIBRARY'] = r'C:\Python36\tcl\tcl8.6'
os.environ['TK_LIBRARY'] = r'C:\Python36\tcl\tk8.6'

setup(
    name = "Final Project Kecerdasan Komputasional",
    options = options,
    version = "0.24",
    description = 'Submission for KK subject',
    executables = executables
)