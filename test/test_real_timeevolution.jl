#=
test_timeevolution:
- Julia version: 1.5.2
- Author: qcancam
- Date: 2021-01-03
=#

include("../src/MatrixProductStates.jl")

using .MatrixProductStates, SparseArrays, Arpack, PyCall, PyPlot

#np=pyimport("numpy")
#@pyimport matplotlib.pyplot as plt

g = 1.0; L = 7
T=10.0
N=1000.0
τ = T/N

function H_TFIM(g, L)
    H = Array{Complex{Float64}, 2}[]
    id = [1  0;
          0  1]
    σˣ = [0  1;
          1  0]
    σᶻ = [1  0;
          0 -1]
    for i in 1:L-2
        push!(H,-σᶻ⊗σᶻ-g*σˣ⊗ id)
    end
    #=L-1=#
    push!(H,-σᶻ⊗σᶻ-g*(σˣ⊗ id+ id⊗ σˣ))

    return H
end

function H_TFIM_MPO(g, L)
    id = [1  0;
          0  1]
    σˣ = [0  1;
          1  0]
    σᶻ = [1  0;
          0 -1]
    W_tnsr = zeros(Complex{Float64}, 3, 3, 2, 2)
    W_tnsr[1, 1, :, :] = id
    W_tnsr[2, 1, :, :] = -σᶻ*0
    W_tnsr[3, 1, :, :] = -σˣ
    W_tnsr[3, 2, :, :] = σᶻ*0
    W_tnsr[3, 3, :, :] = id

    return MPO(W_tnsr, L)
end


H = H_TFIM(g, L)
ψ = randn(MPS{L, Complex{Float64}}, 100, 2)

ϕ,time_list,expect_list= real_time_evolution(ψ, H[1],H[2],H[L-1],H_TFIM_MPO(g,L), T,N,50)
#ϕ= real_time_evolution(ψ, H[1],H[2],H[L-1],H_TFIM_MPO(g,L), T,N,50)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(time_list, expect_list, "-", c="b", ms=10)
ax.set_xlabel("t")
ax.set_ylabel("⟨-∑σˣ(t)⟩")
#ax.set_title("2D Line Plot Sample")
#ax.grid(true)
plt.show()