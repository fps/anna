# ANNA - Artificial Neural Networks for Audio

This is just a playground for trying out some things. This code is not usable in any form.

# Build benchmarks and tests

meson setup --buildtype=release build
meson compile -vC build

## Run tests

meson test -vC build
