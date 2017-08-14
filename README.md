# executor

Must be used with a customized tensorflow at Aetf/tensorflow-rpcdev.
Note that these two projects are tightly coupled together. Make sure to use latest commit from both projects.

## Compile
Requires `CMake 3.8` and modern compiler with c++17 support, e.g. `GCC 7.1` is minimum.

### Dependencies
- ZeroMQ with C++ binding & libmq
- Boost 1.64
- [Q](https://github.com/grantila/q)
- [spdlog](https://github.com/gabime/spdlog) (optional, bundled)
- [docopt.cpp](https://github.com/docopt/docopt.cpp) (optional, bundled)

See [toplevel CMakeLists.txt](CMakeLists.txt) for details.
