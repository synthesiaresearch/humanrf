## Installation


The simplest way to use the python tools is to include this repository in your `PYTHONPATH`:
```bash
# cd here first
export PYTHONPATH="$PYTHONPATH:`pwd`"
```

We provide cross-platform C++ tools to render alembic `.abc` files and to extract them to wavefront `.obj` files. These tools can be installed as follows on Ubuntu (similar on MacOS):
```bash
# Install system dependencies
sudo apt install libopencv-dev libeigen3-dev

# Get third party dependencies
git submodule update --init --recursive

# Install Pangolin
cd mesh_tools/third_party
cd Pangolin
./scripts/install_prerequisites.sh recommended
cmake -B build
cmake --build build -j11
cd ../..

# Build tools
cmake -B build
cmake --build build -j11
```

### Using alembic files

```bash
# Example 1: Convert alembic file to obj files:
./mesh_tools/build/bin/alembic_extractor \
    --alembic /path/to/alembic/file.abc \
    --output /path/to/output/directory

# Example 2: Render alembic file to depthmaps and masks:
./mesh_tools/build/bin/mesh_renderer \
    --csv /path/to/calibration.csv \
    --alembic /path/to/alembic/file.abc \
    --output /path/to/output/directory \
    --depth \
    --mask
```

## Exporters

We provide example exporters to the following [Blender](https://www.blender.org/), [NGP](https://github.com/NVlabs/instant-ngp/blob/master/docs/nerf_dataset_tips.md) and [Colmap](https://colmap.github.io/format.html#text-format) format, which are detailed below.

### Blender

Note that we already provide a blend file per sequence. However, if you would like to regenerate or modify them you can use the following command:

```bash
blender --background --python ./export_blender.py -- \
--csv /path/to/calibration.csv \
--images /path/to/directory/with/first/frames \
--obj /path/to/obj/or/abc/file/Frame000000.obj \
--blend /tmp/example_output_blend.blend
```

## Converting to ActorsHQ format

If you wish to use the ActorsHQ format and its data loader for another dataset, ```./actorshq/toolbox/import_dfa.py``` shows how it is done for [Dynamic Furry Animal dataset](https://github.com/HaiminLuo/Artemis).

## Trouble shooting

* For Blender related issues (like alembic mesh animation not showing), please first ensure you are running the latest Blender version. We tested with Blender 3.2.1.