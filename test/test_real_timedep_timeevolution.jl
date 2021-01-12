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

g = 1.0; L = 3
T=100.0
N=1000.0
τ = T/N

function H_LZ(g, L)
    H = Array{Complex{Float64}, 2}[]
    id = [1  0;
          0  1]
    σˣ = [0  1;
          1  0]
    σᶻ = [1  0;
          0 -1]
    for i in 1:L-2
        push!(H,g*σᶻ⊗id+σˣ⊗id)
    end
    #=L-1=#
    push!(H,g*σᶻ⊗id+σˣ⊗id+g*id⊗σᶻ+id⊗σˣ)

    return H
end

function H_Z_MPO(L)
    id = [1  0;
          0  1]
    σˣ = [0  1;
          1  0]
    σᶻ = [1  0;
          0 -1]
    W_tnsr = zeros(Complex{Float64}, 2, 2, 2, 2)
    W_tnsr[1, 1, :, :] = id
    W_tnsr[2, 1, :, :] = σᶻ
    W_tnsr[2, 2, :, :] = id

    return MPO(W_tnsr, L)
end


#H = H_LZ(g, L)
ψ = randn(MPS{L, Complex{Float64}}, 100, 2)

glist=[-T/2+T/N*i for i in 1:N]
H1=[H_LZ(g, L)[1] for g in glist]
H2=[H_LZ(g, L)[2] for g in glist]
HLm1=[H_LZ(g, L)[L-1] for g in glist]


ϕ,time_list,expect_list= real_timedep_time_evolution(ψ, H1,H2,HLm1,H_Z_MPO(L), T,N,50)

#ϕ= real_time_evolution(ψ, H[1],H[2],H[L-1],H_TFIM_MPO(g,L), T,N,50)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(time_list, expect_list, "-", c="b", ms=10)
ax.set_xlabel("t")
ax.set_ylabel("⟨-∑σᶻ⟩")
#ax.set_title("2D Line Plot Sample")
#ax.grid(true)
plt.show()