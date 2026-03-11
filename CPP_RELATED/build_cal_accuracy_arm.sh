rm -rf build && mkdir build && cd build
cmake .. \
  -DCMAKE_TOOLCHAIN_FILE=$HOME/NDK/android-ndk-r27d/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-28 \
  -DCMAKE_BUILD_TYPE=Release
make
