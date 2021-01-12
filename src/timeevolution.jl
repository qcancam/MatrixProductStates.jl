# Imaginary Time Evolution
# I don't think this works!
# #+HTML: <details><summary>Source</summary>
# #+HTML: <p>

# [[file:~/.julia/dev/MatrixProductStates/README.org::*Imaginary%20Time%20Evolution][Imaginary Time Evolution:1]]
# Fixme! this does not appear to find ground states!

function _MPO_handed_time_evolver_odd(hs::Vector{Matrix{T}}, τ, L, d, reim) where {T}
    tensors = Array{T, 4}[]

    if iseven(L)
        I=reshape([1 0; 0 1],(1,1,2,2))
        push!(tensors,I)
    end

    for h in hs
        if reim =="imag"
            O = exp(-τ*h)
        elseif reim=="real"
            O = exp(-1.0im*τ*h)
        end
        @cast P[(σⁱ, σⁱ′), (σⁱ⁺¹, σⁱ⁺¹′)] |= O[(σⁱ, σⁱ⁺¹), (σⁱ′, σⁱ⁺¹′)] (σⁱ:d, σⁱ′:d)
        U, S, V = svd(P)

        @cast U[1, k, σⁱ, σⁱ′]     := U[(σⁱ, σⁱ′), k] * √(S[k])      (σⁱ:d)
        @cast Ū[k, 1, σⁱ⁺¹, σⁱ⁺¹′] := √(S[k]) * V'[k, (σⁱ⁺¹, σⁱ⁺¹′)] (σⁱ⁺¹:d)
        push!(tensors, U, Ū)
    end

    if isodd(L)
        I=reshape([1 0; 0 1],(1,1,2,2))
        push!(tensors,I)
    end

    MPO{L, T}(tensors)
end

function _MPO_handed_time_evolver_even(hs::Vector{Matrix{T}}, τ, L, d, reim) where {T}
    tensors = Array{T, 4}[]

    if isodd(L)
        I=reshape([1 0; 0 1],(1,1,2,2))
        push!(tensors,I)
    end

    for h in hs
        if reim =="imag"
            O = exp(-τ*h)
        elseif reim=="real"
            O = exp(-1.0im*τ*h)
        end
        @cast P[(σⁱ, σⁱ′), (σⁱ⁺¹, σⁱ⁺¹′)] |= O[(σⁱ, σⁱ⁺¹), (σⁱ′, σⁱ⁺¹′)] (σⁱ:d, σⁱ′:d)
        U, S, V = svd(P)

        @cast U[1, k, σⁱ, σⁱ′]     := U[(σⁱ, σⁱ′), k] * √(S[k])      (σⁱ:d)
        @cast Ū[k, 1, σⁱ⁺¹, σⁱ⁺¹′] := √(S[k]) * V'[k, (σⁱ⁺¹, σⁱ⁺¹′)] (σⁱ⁺¹:d)
        push!(tensors, U, Ū)
    end

    if iseven(L)
        I=reshape([1 0; 0 1],(1,1,2,2))
        push!(tensors,I)
    end

    MPO{L, T}(tensors)
end

function MPO_time_evolvers(h1::Matrix, hi::Matrix, hL::Matrix, τ, L, d, reim)

    if iseven(L)
        odd_hs  = [h1, [hi for _ in 3:2:(L-2)]...,hL]
        even_hs = [hi for i in 2:2:(L-2)]
    else
        odd_hs  = [h1, [hi for _ in 3:2:(L-2)]...] #size=3
        even_hs = [[hi for i in 2:2:(L-2)]...,hL] #size=3
    end

    Uodd  = _MPO_handed_time_evolver_odd(odd_hs, τ, L, d, reim)
    Ueven = _MPO_handed_time_evolver_even(even_hs, τ, L, d, reim)
    Uodd, Ueven
end

function imag_time_evolution(ψ::MPS{L, T}, h1::Matrix{T}, hi::Matrix{T}, hL::Matrix{T}, H::MPO{L,T}, β, N, Dcut) where {L, T}
    @warn "This probably still doesn't work!"
    τ = β/N
    d = length(ψ[1][1, 1, :])
    ϕ = ψ  # Ground state guess
    dir = left
    Uodd, Ueven = MPO_time_evolvers(h1, hi, hL, τ, L, d, "imag")
    for i in 1:N
        println("inverse temperature = ",i*τ)
        println((ϕ'*(H*ϕ))/ϕ'ϕ)

        ϕ1, dir = compress(Uodd  * ϕ,  dir, Dcut=Dcut)
        ϕ,  dir = compress(Ueven * ϕ1, dir, Dcut=Dcut)
        #ϕ,  dir = compress(Uodd  * ϕ2, dir, Dcut=Dcut)
    end
    ϕ
end

function real_time_evolution(ψ::MPS{L, T}, h1::Matrix{T}, hi::Matrix{T}, hL::Matrix{T}, H::MPO{L,T}, Time, N, Dcut) where {L, T}
    @warn "This probably still doesn't work!"
    τ = Time/N
    d = length(ψ[1][1, 1, :])
    ϕb = ψ
    dir = left
    Uodd, Ueven = MPO_time_evolvers(h1, hi, hL, τ, L, d, "real")
    time_list=Any[]
    expect_list=Any[]
    for i in 1:N
        t = i * τ
        expect = (ϕb' * (H * ϕb)) / ϕb'ϕb
        println("time = ",t)
        println(expect)
        push!(time_list,t)
        push!(expect_list,expect)

        Uoddϕ = Uodd * ϕb
        ϕa, dir = compress(Uoddϕ,  dir, Dcut=Dcut)
        #ϕb, dir = variational_compress(ϕa, dir, Uoddϕ)
        #ϕa, dir = variational_compress(ϕb, dir, Uoddϕ)
        Uevenϕ = Ueven * ϕa
        ϕb,  dir = compress(Uevenϕ, dir, Dcut=Dcut)
        #ϕa, dir = variational_compress(ϕb, dir, Uoddϕ)
        #ϕb, dir = variational_compress(ϕa, dir, Uoddϕ)
    end
    ϕb, time_list, expect_list
end

# Time Evolution:1 ends here