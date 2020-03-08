using CSV, DataFrames, Gadfly

Gadfly.push_theme(:dark)

# Load Data
k2com = CSV.read("double_pend/experiments/results/k2/best_com.dat", header=["com"], transpose=true, delim=" ")
k2co1 = CSV.read("double_pend/experiments/results/k2/best_co1.dat", header=["co1"], transpose=true, delim=" ")
k2out = CSV.read("double_pend/experiments/results/k2/best_out.dat", header=["out"], transpose=true, delim=" ")
k2 = hcat(k2com, k2co1, k2out)

k3com = CSV.read("double_pend/experiments/results/k3/best_com.dat", header=["com"], transpose=true, delim=" ")
k3co1 = CSV.read("double_pend/experiments/results/k3/best_co1.dat", header=["co1"], transpose=true, delim=" ")
k3out = CSV.read("double_pend/experiments/results/k3/best_out.dat", header=["out"], transpose=true, delim=" ")
k3 = hcat(k3com, k3co1, k3out)

k4com = CSV.read("double_pend/experiments/results/k4/best_com.dat", header=["com"], transpose=true, delim=" ")
k4co1 = CSV.read("double_pend/experiments/results/k4/best_co1.dat", header=["co1"], transpose=true, delim=" ")
k4out = CSV.read("double_pend/experiments/results/k4/best_out.dat", header=["out"], transpose=true, delim=" ")
k4 = hcat(k4com, k4co1, k4out)

k5com = CSV.read("double_pend/experiments/results/k5/best_com.dat", header=["com"], transpose=true, delim=" ")
k5co1 = CSV.read("double_pend/experiments/results/k5/best_co1.dat", header=["co1"], transpose=true, delim=" ")
k5out = CSV.read("double_pend/experiments/results/k5/best_out.dat", header=["out"], transpose=true, delim=" ")
k5 = hcat(k5com, k5co1, k5out)

k6com = CSV.read("double_pend/experiments/results/k6/best_com.dat", header=["com"], transpose=true, delim=" ")
k6co1 = CSV.read("double_pend/experiments/results/k6/best_co1.dat", header=["co1"], transpose=true, delim=" ")
k6out = CSV.read("double_pend/experiments/results/k6/best_out.dat", header=["out"], transpose=true, delim=" ")
k6 = hcat(k6com, k6co1, k6out)

k7com = CSV.read("double_pend/experiments/results/k7/best_com.dat", header=["com"], transpose=true, delim=" ")
k7co1 = CSV.read("double_pend/experiments/results/k7/best_co1.dat", header=["co1"], transpose=true, delim=" ")
k7out = CSV.read("double_pend/experiments/results/k7/best_out.dat", header=["out"], transpose=true, delim=" ")
k7 = hcat(k7com, k7co1, k7out)

k8com = CSV.read("double_pend/experiments/results/k8/best_com.dat", header=["com"], transpose=true, delim=" ")
k8co1 = CSV.read("double_pend/experiments/results/k8/best_co1.dat", header=["co1"], transpose=true, delim=" ")
k8out = CSV.read("double_pend/experiments/results/k8/best_out.dat", header=["out"], transpose=true, delim=" ")
k8 = hcat(k8com, k8co1, k8out)

# Statistics
@show describe(k2)
@show describe(k3)
@show describe(k4)
@show describe(k5)
@show describe(k6)
@show describe(k7)
@show describe(k8)

k2stats = stack(rename!(describe(k2), :variable => :type), [:mean, :median, :min, :max])
k3stats = stack(rename!(describe(k3), :variable => :type), [:mean, :median, :min, :max])
k4stats = stack(rename!(describe(k4), :variable => :type), [:mean, :median, :min, :max])
k5stats = stack(rename!(describe(k5), :variable => :type), [:mean, :median, :min, :max])
k6stats = stack(rename!(describe(k6), :variable => :type), [:mean, :median, :min, :max])
k7stats = stack(rename!(describe(k7), :variable => :type), [:mean, :median, :min, :max])
k8stats = stack(rename!(describe(k8), :variable => :type), [:mean, :median, :min, :max])

k2plot = plot(k2stats, x=:variable, y=:value, color=:type, Geom.bar(position=:dodge), Guide.xlabel("k2"), Coord.cartesian(ymin=0, ymax=625))
k3plot = plot(k3stats, x=:variable, y=:value, color=:type, Geom.bar(position=:dodge), Guide.xlabel("k3"), Coord.cartesian(ymin=0, ymax=625))
k4plot = plot(k4stats, x=:variable, y=:value, color=:type, Geom.bar(position=:dodge), Guide.xlabel("k4"), Coord.cartesian(ymin=0, ymax=625))
k5plot = plot(k5stats, x=:variable, y=:value, color=:type, Geom.bar(position=:dodge), Guide.xlabel("k5"), Coord.cartesian(ymin=0, ymax=625))
k6plot = plot(k6stats, x=:variable, y=:value, color=:type, Geom.bar(position=:dodge), Guide.xlabel("k6"), Coord.cartesian(ymin=0, ymax=625))
k7plot = plot(k7stats, x=:variable, y=:value, color=:type, Geom.bar(position=:dodge), Guide.xlabel("k7"), Coord.cartesian(ymin=0, ymax=625))
k8plot = plot(k8stats, x=:variable, y=:value, color=:type, Geom.bar(position=:dodge), Guide.xlabel("k8"), Coord.cartesian(ymin=0, ymax=625))

set_default_plot_size(500mm, 100mm)
hstack(k2plot, k3plot, k4plot, k5plot, k6plot, k7plot, k8plot)

# Gen Test Results
k2res = CSV.read("double_pend/experiments/results/k2/res_test.dat", header=["test"], transpose=true, delim=" ")
k3res = CSV.read("double_pend/experiments/results/k3/res_test.dat", header=["test"], transpose=true, delim=" ")
k4res = CSV.read("double_pend/experiments/results/k4/res_test.dat", header=["test"], transpose=true, delim=" ")
k5res = CSV.read("double_pend/experiments/results/k5/res_test.dat", header=["test"], transpose=true, delim=" ")
k6res = CSV.read("double_pend/experiments/results/k6/res_test.dat", header=["test"], transpose=true, delim=" ")
k7res = CSV.read("double_pend/experiments/results/k7/res_test.dat", header=["test"], transpose=true, delim=" ")
k8res = CSV.read("double_pend/experiments/results/k8/res_test.dat", header=["test"], transpose=true, delim=" ")

k2tests = []
k3tests = []
k4tests = []
k5tests = []
k6tests = []
k7tests = []
k8tests = []

for i in 0:624
  push!(k2tests, k2res[1][(i*100)+1:(i*100)+100])
  push!(k3tests, k3res[1][(i*100)+1:(i*100)+100])
  push!(k4tests, k4res[1][(i*100)+1:(i*100)+100])
  push!(k5tests, k5res[1][(i*100)+1:(i*100)+100])
  push!(k6tests, k6res[1][(i*100)+1:(i*100)+100])
  push!(k7tests, k7res[1][(i*100)+1:(i*100)+100])
  push!(k8tests, k8res[1][(i*100)+1:(i*100)+100])
end

k2tests = DataFrame(k2tests)
k3tests = DataFrame(k3tests)
k4tests = DataFrame(k4tests)
k5tests = DataFrame(k5tests)
k6tests = DataFrame(k6tests)
k7tests = DataFrame(k7tests)
k8tests = DataFrame(k8tests)

@show sum.(eachcol(k2tests))
@show sum.(eachcol(k3tests))
@show sum.(eachcol(k4tests))
@show sum.(eachcol(k5tests))
@show sum.(eachcol(k6tests))
@show sum.(eachcol(k7tests))
@show sum.(eachcol(k8tests))

set_default_plot_size(500mm, 250mm)

a = plot(sum.(eachcol(k2tests)), y=Col.value(), Geom.bar, Guide.ylabel("k2"))
b = plot(sum.(eachcol(k3tests)), y=Col.value(), Geom.bar, Guide.ylabel("k3"))
c = plot(sum.(eachcol(k4tests)), y=Col.value(), Geom.bar, Guide.ylabel("k4"))
d = plot(sum.(eachcol(k5tests)), y=Col.value(), Geom.bar, Guide.ylabel("k5"))
e = plot(sum.(eachcol(k6tests)), y=Col.value(), Geom.bar, Guide.ylabel("k6"))
f = plot(sum.(eachcol(k7tests)), y=Col.value(), Geom.bar, Guide.ylabel("k7"))
g = plot(sum.(eachcol(k8tests)), y=Col.value(), Geom.bar, Guide.ylabel("k8"))

vstack(a, b, c, d, e, f, g)