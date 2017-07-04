# executor

Must be used with a customized tensorflow at Aetf/tensorflow-rpcdev. Note that these two projects are tightly coupled together. Make sure to use latest commit from both projects.

## Compile
Requires `CMake 3.8` and `GCC 5.4` as minimum.

### Dependencies
- ZeroMQ
- Boost 1.64
- Q
- spdlog (optional)

See [toplevel CMakeLists.txt](CMakeLists.txt) for details.
