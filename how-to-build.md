# How To Build

## Test environment:
* Windows 10 Pro
* GPU model GTX1080, driver version 432.00
* Anaconda 4.7.12, Python 3.5 (https://www.anaconda.com/distribution/#windows)
* Visual Studio Community 2013 with Update 5 (https://my.visualstudio.com/Downloads?q=visual%20studio%202013&wt.mc_id=o~msft~vscom~older-downloads)
* CMake version 3.16.5 (http://www.cmake.org/cmake/resources/software.html)
* CUDA version 10.1 (https://developer.nvidia.com/cuda-10.1-download-archive-base)
* OptiX version 5.1.1 (https://developer.nvidia.com/designworks/optix/downloads/legacy)

## Instructions:

### Prerequisites
1. Install CUDA 10.1 and verify the installation (check the official installation guide: https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html#verify-installation).
2. Install & build OptiX 5.1.1 and confirm the installation by running "~/Optix SDK 5.1.1/SDK-precompiled-samples/optixDeviceQuery.exe".

### Download and Generate OptaGen
3. Clone the OptaGen repository.
4. Temporarily change "OPTIX_add_sample_executable( OptaGen" to "OPTIX_add_sample_executable( optixPathTracer" in line 33 of "~/OptaGen/src/optixPathTracer/CMakeLists.txt"
5. Startup CMake-GUI.
6. Click "Browse Source..." and select the "src" directory from OptaGen as the source file location.
    (E.g., C:/Users/username/OptaGen/src)
7. Create a build directory manually via Windows File Explorer.
    (E.g., C:/Users/username/OptaGen/build)
8. Press the "Configure" button and select the version of Visual Studio to use (same VS version as building the OptiX SDK). Select the proper compiler version also. Click "Finish." Then the configuration process will start automatically.
    (E.g., Visual Studio 12 2013, x64)
9. While configuring, you might face some interruption. In this case, 
    * Set OptiX_INSTALL_DIR to wherever you installed OptiX.
        (E.g., C:\ProgramData\NVIDIA Corporation\OptiX SDK version)
    * Press "Configure" again.
10. Press "Generate".
11. Recover the change made at the step 0. (i.e., change "optixPathTracer" to "OptaGen")

### CNPY (library to read/write .npy and .npz files in C/C++)
12. Download the cnpy (https://github.com/rogersce/cnpy), generate .sln file via cmake-gui, and build ("F7" on the VS13) it with "Release" option.
13. Link .h file.
	12-1. Open CNPY.sln.
	12-2. Solution Explorer -> cnpy -> right click -> properties -> Configuration Properties -> C/C++ -> General
	12-3. Copy the directory of external dependencies at Additional Include Directories.
	12-4. Open OptaGen.sln which was generated after step 9.
	12-5. Solution Explorer -> OptaGen -> ... -> General -> Additional Include Directories 
	12-6. Add the cnpy repository directory (e.g., C:\Users\username\cnpy) and the directory copied at step 12-3 on Additional Include Directories
14. Let the linker know the library path.
	13-1. OptaGen -> properties -> Configuration Properties -> Linker -> General
	13-2. Add the build directory (e.g., C:\Users\username\cnpy\build) of cnpy on Additional Library Directories.
	13-3. Linker -> Input
	13-4. Add two other directories on Additional Dependencies; C:\Users\user\cnpy\build\Release\cnpy.lib, C:\Users\user\Anaconda3\Library\lib\zlib.lib.

### Build OptaGen
15. Press "F7" (or click "Build Solution" in the "Build" tap).
16. If the Visual Studio asks you to reload your projects, please do so. Since some dependencies are automatically handled for CUDA compilation in VS, it will likely ask you to do this.