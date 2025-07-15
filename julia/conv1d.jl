import Flux
import JSON

weight = randn(3, 16, 16)
bias = randn(16)

input = randn(100, 16, 1)

m = Flux.Conv(weight, bias, dilation = 13)

output = m(input)

data = Dict(
    "weights" => permutedims(weight, (3, 2, 1))[:],
    "bias" => bias,
    "input" => input[:],
    "output" => output[:]
)

Base.write("conv1d.json", JSON.json(data, 2))
