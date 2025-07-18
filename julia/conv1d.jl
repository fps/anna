import Flux
import JSON

weight = randn(Float32, 3, 16, 16)
bias = randn(Float32, 16)

input = randn(Float32, 100, 16, 1)

m = Flux.Conv(weight, bias, dilation = 13)

output = m(input)

data = Dict(
    "parameters" => vcat(permutedims(weight, (1, 2, 3))[:], bias),
    "input" => permutedims(input, (3, 2, 1))[:],
    "output" => permutedims(output, (3, 2, 1))[:]
)

Base.write("conv1d.json", JSON.json(data, 2))
