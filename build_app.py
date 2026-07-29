import os
import sys
import subprocess

def build_executable():
    """
    Bundles the MPFragment application into a standalone executable (.app / .exe)
    using PyInstaller.
    """
    print("==================================================")
    print("      BUILDING MPFRAGMENT STANDALONE SOFTWARE     ")
    print("==================================================")

    # PyInstaller command parameters
    cmd = [
        sys.executable,
        "-m", "PyInstaller",
        "--name=MPFragment",
        "--onedir",          # Creates single application folder bundle (.app on macOS)
        "--windowed",        # No terminal console window shown to end user
        "--noconfirm",
        "--clean",
        f"--add-data=static{os.pathsep}static",
        f"--add-data=final_maskrcnn_fragments_model.onnx{os.pathsep}.",
        "app.py"
    ]

    print(f"[Build] Executing command: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print("\n==================================================")
        print("          BUILD COMPLETED SUCCESSFULLY!           ")
        print("==================================================")
        print("Standalone software generated in directory: dist/MPFragment")
        print("Non-technical users can launch the app by double-clicking:")
        if sys.platform == "darwin":
            print(" -> dist/MPFragment.app")
        else:
            print(" -> dist/MPFragment/MPFragment.exe")
        print("==================================================\n")
    else:
        print(f"\n[Build Error] PyInstaller build failed with exit code {result.returncode}")

if __name__ == "__main__":
    build_executable()
