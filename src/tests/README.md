Marian tests
============

Unit tests and application tests are enabled with CMake option
`-DCOMPILE_TESTS=ON`, e.g.:

    cd build
    cmake .. -DCOMPILE_TESTS=ON
    make -j8

Running all unit tests:

    make test

Running a single unit test is also possible:

    ./src/tests/run_graph_tests

We use [Catch framework](https://github.com/philsquared/Catch) for unit
testing.
