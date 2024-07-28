### Install from the source
**Make sure you edit those installation scripts to point to your own conda path before you run them**

#### Minimal installation
**Recommended for:** Running AmadeusGPT without GPUs. This setup is lightweight and is limited to processing movie files and keypoint outputs (.h5) from DeepLabCut.

```bash
# Install the minimal environment
bash install_minimal.sh

# Activate the conda environment
conda activate amadeusgpt-minimal
```

#### GPU installation
**Recommended for:** Users on Linux with GPUs. Support for Windows and MacOS will be added in the future.

```bash
# Install the gpu environment
bash install_gpu.sh

# Activate the conda environment
conda activate amadeusgpt-gpu
```

#### CPU installation
**Recommended for:** MacOS / Linux users working with very small video files.

```bash
# Install the cpu environment
bash install_cpu.sh

# Activate the conda environment
conda activate amadeusgpt-cpu
```