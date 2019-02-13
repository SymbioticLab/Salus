# Salus: Fine-Grained GPU Sharing Primitives for Deep Learning Applications.

[![pipeline status]][gitlabci]

Implementation of Salus (arXiv paper link coming soon).

Must be used with a customized tensorflow at [SymbioticLab/tensorflow-salus][tf-salus].
Note that these two projects are tightly coupled together. Make sure to use latest commit from both projects.

## Try it out

Using docker. You will need the [nvidia-docker] extension.

```bash
docker run --rm -it registry.gitlab.com/salus/salus
```

This will start a salus server listen at port 5501.

Then when creating tensorflow session, use `zrpc://tcp://localhost:5501` as the session target.

## Compile yourself

Requires `CMake 3.10` and modern compiler with c++14 support, e.g. `GCC 5.4` is minimum.

### Dependencies

- ZeroMQ with C++ binding
- Boost 1.66
- protobuf 3.4.1
- gperftools 2.7 (if build with TCMalloc)
- [nlohmann-json]
- [concurrentqueue]
- [docopt.cpp]
- [easyloggingpp]

See [toplevel CMakeLists.txt](CMakeLists.txt) for details.

[tf-salus]: https://github.com/SymbioticLab/tensorflow-salus
[gitlabci]: https://gitlab.com/Salus/Salus/pipelines
[pipeline status]: https://gitlab.com/Salus/Salus/badges/master/pipeline.svg
[nvidia-docker]: https://github.com/NVIDIA/nvidia-docker
[nlohmann-json]: https://github.com/nlohmann/json
[concurrentqueue]: https://github.com/cameron314/concurrentqueue
[docopt.cpp]: https://github.com/docopt/docopt.cpp
[easyloggingpp]: https://github.com/muflihun/easyloggingpp
