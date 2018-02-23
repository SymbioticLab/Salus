add_executable(executor src/main.cpp)
target_link_libraries(executor
    salus

    docopt_s
)

# We need to disable copy relocation
# see https://stackoverflow.com/questions/34073051/when-we-are-supposed-to-use-rtld-deepbind
# This has to be achieved by using -fPIC alone.
# -fPIC is propogated through dependent libraries but POSITION_INDEPENDENT_CODE adds -fPIE which
# re-enables copy relocation.
set_property(TARGET executor PROPERTY POSITION_INDEPENDENT_CODE OFF)
