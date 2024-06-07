include("../src/routing.jl")
include("../src/gr4h.jl")

using ComponentArrays
using CSV
using DataFrames
using CairoMakie

params = ComponentVector(x1=521.1, x2=-2.9, x3=218.0, x4=4.12, β=21 / 4)
u0 = ComponentVector(soilwater=0.3, rtgstore=0.5)

file_path = "../data/airGR_sim.csv"
data = CSV.File(file_path);
df = DataFrame(data);
data_len = length(df[!, :prec])
pas = ComponentVector(params=params, u0=u0)
flow, fluxes = base_gr4h_run([df[!, :prec], df[!, :pet]], pas, collect(1:data_len))

fluxes_df = DataFrame(fluxes)

timeidx = collect(1:10000)
fig = Figure(size=(400, 300))
ax = CairoMakie.Axis(fig[1, 1], title="predict results", xlabel="time", ylabel="flow(mm)")
lines!(ax, timeidx, df[timeidx, :qsim], color=:red)
lines!(ax, timeidx, fluxes_df[timeidx, :flow], color=:blue)
fig
