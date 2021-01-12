# Compression
# #+HTML: <details><summary>Source</summary>
# #+HTML: <p>

# [[file:~/.julia/dev/MatrixProductStates/README.org::*Compression][Compression:1]]
function compress(ψ::MPS{L, T}, to_the::Right; Dcut::Int=typemax(Int)) where {L, T}
    tensors = Array{T, 3}[]
    
    B = ψ[1]
    d = length(B[1, 1, :])
    
    @cast Bm[(σ¹, a⁰), a¹] |= B[a⁰, a¹, σ¹]
    U, S, V = psvd(Bm, rank=Dcut)
    #S = S/√sum(S .^ 2)

    @cast A[a⁰, a¹, σ¹] |= U[(σ¹, a⁰), a¹] (σ¹:d)
    push!(tensors, A)
    
    for i ∈ 2:L
        B = ψ[i]
        d = length(B[1, 1, :])

        @tensor M[aⁱ⁻¹, aⁱ, σⁱ] := (Diagonal(S)*V')[aⁱ⁻¹, aⁱ⁻¹′] * B[aⁱ⁻¹′, aⁱ, σⁱ]
        @cast   Mm[(σⁱ, aⁱ⁻¹), aⁱ] |= M[aⁱ⁻¹, aⁱ, σⁱ]
        
        U, S, V = psvd(Mm, rank=Dcut)
        #S = S/√sum(S .^ 2)

        @cast A[aⁱ⁻¹, aⁱ, σⁱ] |= U[(σⁱ, aⁱ⁻¹), aⁱ] (σⁱ:d)
        push!(tensors, A)
    end
    MPS{L, T}(tensors), Left()
end

leftcanonical(ψ) = compress(ψ, right)[1]

function compress(ψ::MPS{L, T}, to_the::Left; Dcut::Int=typemax(Int)) where {L, T}
    tensors = Array{T, 3}[]
    
    A = ψ[L]
    d = length(A[1, 1, :])
    @cast Am[aᴸ⁻¹, (σᴸ, aᴸ)] |= A[aᴸ⁻¹, aᴸ, σᴸ]
    
    U, S, V = psvd(Am, rank=Dcut)
    #S = S/√sum(S .^ 2)    

    @cast B[aᴸ⁻¹, aᴸ, σᴸ] |= V'[aᴸ⁻¹, (σᴸ, aᴸ)] (σᴸ:d)
    push!(tensors, B)
    
    for i ∈ (L-1):-1:1
        A = ψ[i]
        d = length(A[1, 1, :])
        @tensor M[aⁱ⁻¹, aⁱ, σⁱ]    := A[aⁱ⁻¹, aⁱ′, σⁱ] * (U * Diagonal(S))[aⁱ′, aⁱ]
        @cast   Mm[aⁱ⁻¹, (σⁱ, aⁱ)] |= M[aⁱ⁻¹, aⁱ, σⁱ]
        
        U, S, V = psvd(Mm, rank=Dcut)
        #S = S/√sum(S .^ 2)

        @cast B[aⁱ⁻¹, aⁱ, σⁱ] |= V'[aⁱ⁻¹, (σⁱ, aⁱ)] (σⁱ:d)
        push!(tensors, B)
    end
    MPS{L, T}(reverse(tensors)), Right()
end

rightcanonical(ψ) = compress(ψ, left)[1]

#compress(ψ; Dcut) = compress(ψ, left, Dcut=Dcut)[1]

function variational_compress(ψ::MPS{L, T}, to_the::Left, ϕ::MPS{L, T}) where {L, T}
    btensors = Array{T, 3}[]

    A=ψ[L]
    d = length(A[1, 1, :])
    @cast Am[aᴸ⁻¹, (σᴸ, aᴸ)] |= A[aᴸ⁻¹, aᴸ, σᴸ]
    U, S, V = svd(Am)
    @cast B[aᴸ⁻¹, aᴸ, σᴸ] |= V'[aᴸ⁻¹, (σᴸ, aᴸ)] (σᴸ:d)
    push!(btensors, B)

    for i in L-1:-1:2
        prevM=1
        for j in 1:i-1
            M = ϕ[j]
            M̃ = ψ[j]
            prevM=sum([M̃[:,:,k]'*prevM*M[:,:,k] for k in 1:d])
        end
        Lm=prevM

        prevM=1
        temp=0
        for j in L:-1:i+1
            temp += 1
            M = ϕ[j]
            M̃ = btensors[temp]
            prevM = sum([M[:,:,k]*prevM*M̃[:,:,k]' for k in 1:d])
        end
        Rm=prevM

        M = ϕ[i]
        @tensor M̃[aⁱ⁻¹, aⁱ, σⁱ] := Lm[aⁱ⁻¹,aⁱ⁻¹′] * Rm[aⁱ′,aⁱ] * M[aⁱ⁻¹′, aⁱ′, σⁱ]
        #println("error rate : ",round(1.0-real(tr(sum(M̃[:,:,k]*M̃[:,:,k]' for k in 1:2))),digits=10))

        @cast   Mm[aⁱ⁻¹, (σⁱ, aⁱ)] |= M̃[aⁱ⁻¹, aⁱ, σⁱ]
        U, S, V = svd(Mm)
        @cast B[aⁱ⁻¹, aⁱ, σⁱ] |= V'[aⁱ⁻¹, (σⁱ, aⁱ)] (σⁱ:d)

        push!(btensors, B)
    end


    prevM=1
    temp=0
    for j in L:-1:2
        temp += 1
        M = ϕ[j]
        M̃ = btensors[temp]
        prevM=sum([M[:,:,k]*prevM*M̃[:,:,k]' for k in 1:d])
    end
    Rm=prevM
    M = ϕ[1]
    @tensor M̃[aⁱ⁻¹, aⁱ, σⁱ] := Rm[aⁱ′,aⁱ] * M[aⁱ⁻¹, aⁱ′, σⁱ]
    #println("error rate : ",round(1.0-real(tr(sum(M̃[:,:,k]*M̃[:,:,k]' for k in 1:2))),digits=10))
    @cast   Mm[aⁱ⁻¹, (σⁱ, aⁱ)] |= M̃[aⁱ⁻¹, aⁱ, σⁱ]
    U, S, V = svd(Mm)
    @cast B[aⁱ⁻¹, aⁱ, σⁱ] |= V'[aⁱ⁻¹, (σⁱ, aⁱ)] (σⁱ:d)

    push!(btensors, B)

    MPS{L, T}(reverse(btensors)), Right()
end


function variational_compress(ψ::MPS{L, T}, to_the::Right, ϕ::MPS{L, T}) where {L, T}
    atensors = Array{T, 3}[]

    B=ψ[1]
    d = length(B[1, 1, :])
    @cast   Bm[(σⁱ, aⁱ⁻¹), aⁱ] |= B[aⁱ⁻¹, aⁱ, σⁱ]
    U, S, V = svd(Bm)
    @cast A[aⁱ⁻¹, aⁱ, σⁱ] |= U[(σⁱ, aⁱ⁻¹), aⁱ] (σⁱ:d)
    push!(atensors, A)

    for i in 2:L-1
        prevM=1
        temp=0
        for j in 1:i-1
            temp+=1
            M = ϕ[j]
            M̃ = atensors[temp]
            prevM=sum([M̃[:,:,k]'*prevM*M[:,:,k] for k in 1:d])
        end
        Lm=prevM

        prevM=1
        for j in L:-1:i+1
            M = ϕ[j]
            M̃ = ψ[j]
            prevM=sum([M[:,:,k]*prevM*M̃[:,:,k]' for k in 1:d])
        end
        Rm=prevM

        M = ϕ[i]
        @tensor M̃[aⁱ⁻¹, aⁱ, σⁱ] := Lm[aⁱ⁻¹,aⁱ⁻¹′] * Rm[aⁱ′,aⁱ] * M[aⁱ⁻¹′, aⁱ′, σⁱ]
        #println("error rate : ",round(1.0-real(tr(sum(M̃[:,:,k]*M̃[:,:,k]' for k in 1:2))),digits=10))

        @cast   Mm[(σⁱ, aⁱ⁻¹), aⁱ] |= M̃[aⁱ⁻¹, aⁱ, σⁱ]
        U, S, V = svd(Mm)
        @cast A[aⁱ⁻¹, aⁱ, σⁱ] |= U[(σⁱ, aⁱ⁻¹), aⁱ] (σⁱ:d)

        push!(atensors, A)
    end

    prevM=1
    temp=0
    for j in 1:L-1
        temp+=1
        M = ϕ[j]
        M̃ = atensors[temp]
        prevM = sum([M̃[:,:,k]'*prevM*M[:,:,k] for k in 1:d])
    end
    Lm=prevM
    M = ϕ[L]
    @tensor M̃[aⁱ⁻¹, aⁱ, σⁱ] := Lm[aⁱ⁻¹,aⁱ⁻¹'] * M[aⁱ⁻¹', aⁱ, σⁱ]
    #println("error rate : ",round(1.0-real(tr(sum(M̃[:,:,k]*M̃[:,:,k]' for k in 1:2))),digits=10))
    @cast   Mm[(σⁱ, aⁱ⁻¹), aⁱ] |= M̃[aⁱ⁻¹, aⁱ, σⁱ]
    U, S, V = svd(Mm)
    @cast A[aⁱ⁻¹, aⁱ, σⁱ] |= U[(σⁱ, aⁱ⁻¹), aⁱ] (σⁱ:d)

    push!(atensors, A)

    MPS{L, T}(atensors), Left()
end

# Compression:1 ends here
