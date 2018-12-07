# executor

Must be used with a customized tensorflow at Aetf/tensorflow-rpcdev.
Note that these two projects are tightly coupled together. Make sure to use latest commit from both projects.

## Compile
Requires `CMake 3.10` and modern compiler with c++14 support, e.g. `GCC 5.4` is minimum.

### Dependencies
- ZeroMQ with C++ binding & libmq
- Boost 1.64
- [Q](https://github.com/grantila/q)
- [spdlog](https://github.com/gabime/spdlog) (optional, bundled)
- [docopt.cpp](https://github.com/docopt/docopt.cpp) (optional, bundled)

See [toplevel CMakeLists.txt](CMakeLists.txt) for details.
