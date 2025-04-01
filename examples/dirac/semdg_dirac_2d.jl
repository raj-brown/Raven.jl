#------------------------------Markdown Language Header-----------------------
# # Dirac Equation
#
# ```
## @MISC{Bal2018,
#    author = {Guillaume Bal},
#    title = {Topological Protection Perturbed of Edge States},
#    year = {2018},
#    url = {arXiv:1709.00605v2},
# }
# ```
#
#jl #FIXME: add convergence study
#jl #FIXME: optimize kernels
#jl #FIXME: add introduction to dirac equation latex
#--------------------------------Markdown Language Header-----------------------
using Base: sign_mask
using Adapt
using MPI
using CUDA
using KernelAbstractions
using KernelAbstractions.Extras: @unroll
using LinearAlgebra
using Printf
using Raven
using StaticArrays
using WriteVTK

using MAT
using SpecialFunctions

import Base.abs2

const convergetest = true
const outputvtk = false
const m = -10.0
const matpar = :half # :const :half :waveguide
const BC = :crbc # options: :crbc :forbc :reflect
const K = 4
const delta_x = 1.0
const delta_y = 0.2
const timeend = 20.0
#const ħ = 1

function abs2(v::SVector{2,ComplexF64})
    return abs2.(v)
end

function initialCondition(x::SVector{2})
    FT = eltype(x)
    CT = Complex{FT}
    return exp(-(x[1] / 0.1)^2) * exp(-((x[2]) / 0.1)^2) .* SVector{2,CT}(1, 1)
end

# ∂ₜψ = -σ₁∂ₓψ-im(y)σ₃ψ
@kernel function crbc_tangent_kernel!(
    dq,
    q,
    dRdX,
    wJ,
    invwJ,
    DT,
    m,
    ::Val{N},
    ::Val{C},
) where {N,C}
    i, j, c1 = @index(Local, NTuple) # (i,j) dof in cell c1 local cell index within workgroup
    _, _, c = @index(Global, NTuple) # global cell numbering 

    # C compile time data corrisponding to cells per workgroup
    lDT1 = @localmem eltype(eltype(q)) (N[1], N[1])
    lDT2 = @localmem eltype(eltype(q)) (N[2], N[2])
    lU = @localmem eltype(q) (N..., C) # local solution at prod(N) dofs in C cells of domain

    # Pauli Matrices: SA denotes StaticArrays
    σ₁ = SA[0.0 1.0; 1.0 0.0]
    σ₂ = SA[0.0 -im; im 0.0]
    σ₃ = SA[1.0 0.0; 0.0 -1.0]

    @inbounds begin
        for sj = 0x0:N[2]:(N[1]-0x1) # Sets stride in j
            if j + sj <= N[1] && c1 == 0x1 # localmemory is shared amongst a workgroup so only c1 needs to work
                lDT1[i, j+sj] = DT[1][i, j+sj]
            end
        end

        for si = 0x0:N[1]:(N[2]-0x1)
            if i + si <= N[2] && c1 == 0x1
                lDT2[i+si, j] = DT[2][i+si, j]
            end
        end

        # data (i, j) in global index cell c
        qijc = q[i, j, c]
        lU[i, j, c1] = qijc
    end

    @synchronize

    σ₁ = SA[0.0 1.0; 1.0 0.0]
    σ₂ = SA[0.0 -im; im 0.0]
    σ₃ = SA[1.0 0.0; 0.0 -1.0]
    # Here a thread will compute the (i,j)th component of the dq matrix where i,j is the dof and c is the global cell index
    @inbounds begin
        dqijc_update = -zero(eltype(dq))
        invwJijc = invwJ[i, j, c]

        dRdXijc = dRdX[i, j, c]
        wJijc = wJ[i, j, c]
        wJdRdXijc = wJijc * dRdXijc

        @unroll for m = 0x1:N[1]
            dqijc_update -= wJdRdXijc[1] * lDT1[i, m] * σ₁ * lU[m, j, c1] # -rₓDᵣσ₁u
        end

        @unroll for n = 0x1:N[2]
            dqijc_update -= wJdRdXijc[2] * lDT2[j, n] * σ₁ * lU[i, n, c1] # -sₓDₛσ₁u 
        end

        dq[i, j, c] = invwJijc * dqijc_update - im * m * σ₃ * lU[i, j, c1]
    end
end

# ∂ₜψ = dqdtan-σ₂∂ᵥψ
@kernel function crbc_volume_kernel_top!(
    dq,
    q,
    dqdtan,
    dwdn,
    m,
    ::Val{N},
    ::Val{C},
) where {N,C}
    i, j, c1 = @index(Local, NTuple) # (i,j) dof in cell c1 local cell index within workgroup
    _, _, c = @index(Global, NTuple) # global cell numbering 

    σ₁ = SA[0.0 1.0; 1.0 0.0]
    σ₂ = SA[0.0 -im; im 0.0]
    σ₃ = SA[1.0 0.0; 0.0 -1.0]
    @inbounds begin
        P = SA[0.5 0.5; 0.5im -0.5im]

        dq[i, j, c] += dqdtan[i, j, c] - σ₂ * P * dwdn[i, j, c]
    end
end

@kernel function crbc_volume_kernel_bottom!(
    dq,
    q,
    dqdtan,
    dwdn,
    m,
    ::Val{N},
    ::Val{C},
) where {N,C}
    i, j, c1 = @index(Local, NTuple) # (i,j) dof in cell c1 local cell index within workgroup
    _, _, c = @index(Global, NTuple) # global cell numbering 

    σ₁ = SA[0.0 1.0; 1.0 0.0]
    σ₂ = SA[0.0 -im; im 0.0]
    σ₃ = SA[1.0 0.0; 0.0 -1.0]
    @inbounds begin
        P = SA[0.5 0.5; -0.5im 0.5im]

        dq[i, j, c] += dqdtan[i, j, c] - σ₂ * P * dwdn[i, j, c]
    end
end


# ∂ₜψ = -σ₁∂ₓψ-σ₂∂ᵥψ-im(y)σ₃ψ
@kernel function rhs_volume_kernel!(
    dq,
    q,
    dRdX,
    wJ,
    invwJ,
    DT,
    materialparams,
    ::Val{N},
    ::Val{C},
) where {N,C}
    i, j, c1 = @index(Local, NTuple) # (i,j) dof in cell c1 local cell index within workgroup
    _, _, c = @index(Global, NTuple) # global cell numbering 

    # C compile time data corrisponding to cells per workgroup
    lDT1 = @localmem eltype(eltype(q)) (N[1], N[1])
    lDT2 = @localmem eltype(eltype(q)) (N[2], N[2])
    lU = @localmem eltype(q) (N..., C) # local solution at prod(N) dofs in C cells of domain

    # Pauli Matrices: SA denotes StaticArrays
    σ₁ = SA[0.0 1.0; 1.0 0.0]
    σ₂ = SA[0.0 -im; im 0.0]
    σ₃ = SA[1.0 0.0; 0.0 -1.0]

    @inbounds begin
        for sj = 0x0:N[2]:(N[1]-0x1) # Sets stride in j
            if j + sj <= N[1] && c1 == 0x1 # localmemory is shared amongst a workgroup so only c1 needs to work
                lDT1[i, j+sj] = DT[1][i, j+sj]
            end
        end

        for si = 0x0:N[1]:(N[2]-0x1)
            if i + si <= N[2] && c1 == 0x1
                lDT2[i+si, j] = DT[2][i+si, j]
            end
        end

        # data (i, j) in global index cell c
        qijc = q[i, j, c]
        lU[i, j, c1] = qijc
    end

    @synchronize

    σ₁ = SA[0.0 1.0; 1.0 0.0]
    σ₂ = SA[0.0 -im; im 0.0]
    σ₃ = SA[1.0 0.0; 0.0 -1.0]
    # Here a thread will compute the (i,j)th component of the dq matrix where i,j is the dof and c is the global cell index
    @inbounds begin
        dqijc_update = -zero(eltype(dq))
        invwJijc = invwJ[i, j, c]

        dRdXijc = dRdX[i, j, c]
        wJijc = wJ[i, j, c]
        wJdRdXijc = wJijc * dRdXijc
        matparamijc = materialparams[i, j, c]


        @unroll for m = 0x1:N[1]
            dqijc_update -= wJdRdXijc[1] * lDT1[i, m] * σ₁ * lU[m, j, c1] # -rₓDᵣσ₁u
            dqijc_update -= wJdRdXijc[3] * lDT1[i, m] * σ₂ * lU[m, j, c1] # -rᵥDᵣσ₂u
        end

        @unroll for n = 0x1:N[2]
            dqijc_update -= wJdRdXijc[2] * lDT2[j, n] * σ₁ * lU[i, n, c1] # -sₓDₛσ₁u 
            dqijc_update -= wJdRdXijc[4] * lDT2[j, n] * σ₂ * lU[i, n, c1] # -sᵥDₛσ₂u
        end
        dq[i, j, c] += invwJijc * dqijc_update - im * matparamijc * σ₃ * lU[i, j, c1]
    end
end

@kernel function crbc_surface_kernel!(
    dq,
    q,
    vmapM,
    vmapP,
    n,
    wsJ,
    invwJ,
    ::Val{N},
    ::Val{C},
) where {N,C}
    ij, c1 = @index(Local, NTuple)
    _, c = @index(Global, NTuple)
    lqflux = @localmem eltype(dq) (N..., C)

    σ₁ = SA[0.0 1.0; 1.0 0.0]
    σ₂ = SA[0.0 -im; im 0.0]

    @inbounds if ij <= N[1]
        i = ij
        @unroll for j = 1:N[2]
            lqflux[i, j, c1] = zero(eltype(lqflux))
        end
    end

    @synchronize

    σ₁ = SA[0.0 1.0; 1.0 0.0]
    σ₂ = SA[0.0 -im; im 0.0]
    @inbounds if ij <= N[2]
        # Faces with r=-1 : West
        i = 1
        j = ij
        fid = j

        idM = vmapM[fid, c]
        idP = vmapP[fid, c]

        nf = n[fid, c]
        wsJf = wsJ[fid, c]

        qM = q[idM]
        qP = q[idP]

        invwJijc = invwJ[idM]

        #fscale = invwJijc * wsJf / 2

        #lqflux[i, j, c1] += fscale * (nf[1] * σ₁ * (qM - qP))

        fscale = invwJijc * wsJf
        A = SA[0 nf[1]-im*nf[2]; nf[1]+im*nf[2] 0]
        numflux = (qM - qP + A * (qM + qP)) / 2
        lqflux[i, j, c1] += fscale * (nf[1] * σ₁ * qM + nf[2] * σ₂ * qM - numflux)

        # Faces with r=1 : East 
        i = N[1]
        j = ij
        fid = N[2] + j

        idM = vmapM[fid, c]
        idP = vmapP[fid, c]

        nf = n[fid, c]
        wsJf = wsJ[fid, c]

        qM = q[idM]
        qP = q[idP]


        invwJijc = invwJ[idM]

        #fscale = invwJijc * wsJf / 2

        #lqflux[i, j, c1] += fscale * (nf[1] * σ₁ * (qM - qP))

        fscale = invwJijc * wsJf
        A = SA[0 nf[1]-im*nf[2]; nf[1]+im*nf[2] 0]
        numflux = (qM - qP + A * (qM + qP)) / 2
        lqflux[i, j, c1] += fscale * (nf[1] * σ₁ * qM + nf[2] * σ₂ * qM - numflux)
    end

    @synchronize

    # RHS update
    i = ij
    @inbounds if i <= N[1]
        @unroll for j = 1:N[2]
            dq[i, j, c] += lqflux[i, j, c1]
        end
    end
end

@kernel function rhs_surface_kernel!(
    dq,
    q,
    crbc_data,
    vmapM,
    vmapP,
    bc,
    n,
    wsJ,
    invwJ,
    ::Val{N},
    ::Val{C},
) where {N,C}
    ij, c1 = @index(Local, NTuple)
    _, c = @index(Global, NTuple)
    lqflux = @localmem eltype(dq) (N..., C)

    σ₁ = SA[0.0 1.0; 1.0 0.0]
    σ₂ = SA[0.0 -im; im 0.0]

    @inbounds if ij <= N[1]
        i = ij
        @unroll for j = 1:N[2]
            lqflux[i, j, c1] = zero(eltype(lqflux))
        end
    end

    @synchronize

    σ₁ = SA[0.0 1.0; 1.0 0.0]
    σ₂ = SA[0.0 -im; im 0.0]
    @inbounds if ij <= N[2]
        # Faces with r=-1 : West
        i = 1
        j = ij
        fid = j

        idM = vmapM[fid, c]
        idP = vmapP[fid, c]

        nf = n[fid, c]
        wsJf = wsJ[fid, c]

        qM = q[idM]
        qP = q[idP]

        invwJijc = invwJ[idM]

        fscale = invwJijc * wsJf
        A = SA[0 nf[1]-im*nf[2]; nf[1]+im*nf[2] 0]
        numflux = (qM - qP + A * (qM + qP)) / 2
        lqflux[i, j, c1] += fscale * (nf[1] * σ₁ * qM + nf[2] * σ₂ * qM - numflux)

        # Faces with r=1 : East 
        i = N[1]
        j = ij
        fid = N[2] + j

        idM = vmapM[fid, c]
        idP = vmapP[fid, c]

        nf = n[fid, c]
        wsJf = wsJ[fid, c]

        qM = q[idM]
        qP = q[idP]

        invwJijc = invwJ[idM]

        fscale = invwJijc * wsJf

        A = SA[0 nf[1]-im*nf[2]; nf[1]+im*nf[2] 0]
        numflux = (qM - qP + A * (qM + qP)) / 2
        lqflux[i, j, c1] += fscale * (nf[1] * σ₁ * qM + nf[2] * σ₂ * qM - numflux)
    end

    @synchronize

    σ₁ = SA[0.0 1.0; 1.0 0.0]
    σ₂ = SA[0.0 -im; im 0.0]
    @inbounds if ij <= N[1]
        # Faces with s=-1 : South 
        i = ij
        j = 1
        fid = 2N[2] + i

        idM = vmapM[fid, c]
        idP = vmapP[fid, c]
        idB = bc[3, c]

        nf = n[fid, c]
        wsJf = wsJ[fid, c]

        qM = q[idM]
        qP = q[idP]

        if idB == 2
            #qP = (qM - σ₂ * qM) / 2
            if BC == :crbc
                qP = crbc_data[idP]
            elseif BC == :forbc
                qP = (qM - σ₂ * qM) / 2
            elseif BC == :reflect
                qP = -qM
            end
        end

        invwJijc = invwJ[idM]

        σ₁ = SA[0.0 1.0; 1.0 0.0]
        σ₂ = SA[0.0 -im; im 0.0]

        fscale = invwJijc * wsJf
        # nx σ₁ (u_- u*) + ny σ₂ (u_- u*)
        A = SA[0 nf[1]-im*nf[2]; nf[1]+im*nf[2] 0]
        numflux = (qM - qP + A * (qM + qP)) / 2
        lqflux[i, j, c1] += fscale * (nf[1] * σ₁ * qM + nf[2] * σ₂ * qM - numflux)

        # Faces with s=1 : North
        i = ij
        j = N[2]
        fid = 2N[2] + N[1] + i

        idM = vmapM[fid, c]
        idP = vmapP[fid, c]
        idB = bc[4, c]

        nf = n[fid, c]
        wsJf = wsJ[fid, c]

        qM = q[idM]
        qP = q[idP]

        if idB == 2
            if BC == :crbc
                qP = crbc_data[idP]
            elseif BC == :forbc
                qP = (qM + σ₂ * qM) / 2
            elseif BC == :reflect
                qP = -qM
            end
        end

        invwJijc = invwJ[idM]

        fscale = invwJijc * wsJf
        A = SA[0 nf[1]-im*nf[2]; nf[1]+im*nf[2] 0]
        numflux = (qM - qP + A * (qM + qP)) / 2
        lqflux[i, j, c1] += fscale * (nf[1] * σ₁ * qM + nf[2] * σ₂ * qM - numflux)
    end

    @synchronize

    # RHS update
    i = ij
    @inbounds if i <= N[1]
        @unroll for j = 1:N[2]
            dq[i, j, c] += lqflux[i, j, c1]
        end
    end
end

function coefmap!(materialparams, gridpoints, m)
    iMax, jMax, eMax = size(gridpoints)
    if matpar == :half
        for e = 1:eMax
            center = last.(gridpoints[[1, end], [1, end], e])[1, :]
            avg = (center[1] + center[2]) / 2
            mtemp = avg > 0 ? m : -m
            for i = 1:iMax, j = 1:jMax
                materialparams[i, j, e] = mtemp
            end
        end
    elseif matpar == :waveguide
        tempmin = 100000.0
        for e = 1:eMax
            center = last.(gridpoints[[1, end], [1, end], e])[1, :]
            avg = (center[1] + center[2]) / 2
            if abs(avg) < tempmin
                tempmin = abs(avg)
            end
        end

        for e = 1:eMax
            center = last.(gridpoints[[1, end], [1, end], e])[1, :]
            avg = (center[1] + center[2]) / 2
            mtemp = abs(avg) ≈ tempmin ? -m : m
            for i = 1:iMax, j = 1:jMax
                materialparams[i, j, e] = mtemp
            end
        end
    elseif matpar == :const
        for e = 1:eMax
            for i = 1:iMax, j = 1:jMax
                materialparams[i, j, e] = m
            end
        end
    end
end

# ∂ₜψ = -σ₁∂ₓψ-im(y)σ₃ψ
#
# P∂ₜψ = -Pσ₁∂ₓψ-im(y)Pσ₃ψ              : Top
#      = -σ₂∂ₓω-im(y)σ₁ω                : Top
# P∂ₜψ = -Pσ₁∂ₓψ-im(y)Pσ₃ψ              : Bottom
#      = -Pσ₁PinvP ∂ₓψ-im(y)Pσ₃PinvPψ   : Bottom
#      =  σ₂∂ₓω-im(y)σ₁ω                : Bottom
function crbc_recursion_bottom!(dwdn, dqdtan, q, grid, m, a, sig)
    cell = referencecell(grid)
    Q = size(cell, 2) - 1
    S = size(points(grid))

    q1, q2 = components(q)
    dw1dn, dw2dn = components(dwdn)
    dq1dtan, dq2dtan = components(dqdtan)

    for e = 1:S[3]
        for x = 1:size(cell, 1)
            for (j, i) in zip(2:Q+1, Q:1)
                dw1dn[x, i, e] = (a[2*(j-1)-1] - 1) * dw1dn[x, i+1, e]

                dw1dn[x, i, e] -= a[2*(j-1)] * (dq1dtan[x, i, e] + im * dq2dtan[x, i, e])
                dw1dn[x, i, e] +=
                    a[2*(j-1)-1] * (dq1dtan[x, i+1, e] + im * dq2dtan[x, i+1, e])
                #dw1dn[x, i, e] -= im*m*a[2*(j-1)]*(q1[x, i, e] - im*q2[x, i, e])
                #dw1dn[x, i, e] += im*m*a[2*(j-1)-1]*(q1[x, i+1, e] - im*q2[x, i+1, e])

                dw1dn[x, i, e] += sig[2*(j-1)] * (q1[x, i, e] + im * q2[x, i, e])
                dw1dn[x, i, e] -= sig[2*(j-1)-1] * (q1[x, i+1, e] + im * q2[x, i+1, e])

                dw1dn[x, i, e] /= (a[2*(j-1)] + 1)
            end

            dw2dn[x, 1, e] = zero(eltype(dw2dn))

            for (j, i) in zip(Q+1:-1:2, 1:Q)
                dw2dn[x, i+1, e] = (a[2*(j-1)] - 1) * dw2dn[x, i, e]

                dw2dn[x, i+1, e] -= a[2*(j-1)] * (dq1dtan[x, i, e] - im * dq2dtan[x, i, e])
                dw2dn[x, i+1, e] +=
                    a[2*(j-1)-1] * (dq1dtan[x, i+1, e] - im * dq2dtan[x, i+1, e])
                #dw2dn[x, i+1, e] -= im*m*a[2*(j-1)]*(q1[x, i, e] + im*q2[x, i, e])
                #dw2dn[x, i+1, e] += im*m*a[2*(j-1)-1]*(q1[x, i+1, e] + im*q2[x, i+1, e])

                dw2dn[x, i+1, e] += sig[2*(j-1)] * (q1[x, i, e] - im * q2[x, i, e])
                dw2dn[x, i+1, e] -= sig[2*(j-1)-1] * (q1[x, i+1, e] - im * q2[x, i+1, e])

                dw2dn[x, i+1, e] /= (a[2*(j-1)-1] + 1)
            end
        end
    end

    return dwdn
end



function crbc_recursion_top!(dwdn, dqdtan, q, grid, m, a, sig)
    cell = referencecell(grid)
    Q = size(cell, 2) - 1
    S = size(points(grid))

    q1, q2 = components(q)
    dw1dn, dw2dn = components(dwdn)
    dq1dtan, dq2dtan = components(dqdtan)

    for e = 1:S[3]
        for x = 1:size(cell, 1)
            for j = 2:Q+1
                dw1dn[x, j, e] = (a[2*(j-1)-1] - 1) * dw1dn[x, j-1, e]

                dw1dn[x, j, e] += a[2*(j-1)] * (dq1dtan[x, j, e] - im * dq2dtan[x, j, e])
                dw1dn[x, j, e] -=
                    a[2*(j-1)-1] * (dq1dtan[x, j-1, e] - im * dq2dtan[x, j-1, e])
                #dw1dn[x, j, e] -= im*m*a[2*(j-1)]*(q1[x, j, e] + im*q2[x, j, e])
                #dw1dn[x, j, e] += im*m*a[2*(j-1)-1]*(q1[x, j-1, e] + im*q2[x, j-1, e])

                dw1dn[x, j, e] += sig[2*(j-1)] * (q1[x, j, e] - im * q2[x, j, e])
                dw1dn[x, j, e] -= sig[2*(j-1)-1] * (q1[x, j-1, e] - im * q2[x, j-1, e])

                dw1dn[x, j, e] /= (a[2*(j-1)] + 1)
            end

            dw2dn[x, Q+1, e] = zero(eltype(dw2dn))

            for j = Q+1:-1:2
                dw2dn[x, j-1, e] = (a[2*(j-1)] - 1) * dw2dn[x, j, e]

                dw2dn[x, j-1, e] += a[2*(j-1)] * (dq1dtan[x, j, e] + im * dq2dtan[x, j, e])
                dw2dn[x, j-1, e] -=
                    a[2*(j-1)-1] * (dq1dtan[x, j-1, e] + im * dq2dtan[x, j-1, e])
                #dw2dn[x, j-1, e] -= im*m*a[2*(j-1)]*(q1[x, j, e] - im*q2[x, j, e])
                #dw2dn[x, j-1, e] += im*m*a[2*(j-1)-1]*(q1[x, j-1, e] - im*q2[x, j-1, e])

                dw2dn[x, j-1, e] += sig[2*(j-1)] * (q1[x, j, e] + im * q2[x, j, e])
                dw2dn[x, j-1, e] -= sig[2*(j-1)-1] * (q1[x, j-1, e] + im * q2[x, j-1, e])

                dw2dn[x, j-1, e] /= (a[2*(j-1)-1] + 1)
            end
        end
    end

    return dwdn
end

function crbc_rhs!(dq, q, dwdn, dqdtan, grid, invwJ, DT, m, a, sig, cm; orient = "top")
    backend = Raven.get_backend(dq)
    cell = referencecell(grid)
    dRdX, _, wJ = components(first(volumemetrics(grid)))
    n, _, wsJ = components(first(surfacemetrics(grid)))
    fm = facemaps(grid)

    C = 1#max(512 ÷ prod(size(cell)), 1)
    crbc_tangent_kernel!(backend, (size(cell)..., C))(
        dqdtan,
        q,
        dRdX,
        wJ,
        invwJ,
        DT,
        m,
        Val(size(cell)),
        Val(C);
        ndrange = size(dq),
    )

    J = maximum(size(cell))
    C = 1#max(128 ÷ J, 1)
    crbc_surface_kernel!(backend, (J, C))(
        dqdtan,
        q,
        fm.vmapM,
        fm.vmapP,
        n,
        wsJ,
        invwJ,
        Val(size(cell)),
        Val(C);
        ndrange = (J, last(size(dq))),
    )

    if orient == "top"
        dwdn = crbc_recursion_top!(dwdn, dqdtan, q, grid, m, a, sig)

        C = 1#max(512 ÷ prod(size(cell)), 1)
        crbc_volume_kernel_top!(backend, (size(cell)..., C))(
            dq,
            q,
            dqdtan,
            dwdn,
            m,
            Val(size(cell)),
            Val(C);
            ndrange = size(dq),
        )
    elseif orient == "bottom"
        dwdn = crbc_recursion_bottom!(dwdn, dqdtan, q, grid, m, a, sig)

        C = 1#max(512 ÷ prod(size(cell)), 1)
        crbc_volume_kernel_bottom!(backend, (size(cell)..., C))(
            dq,
            q,
            dqdtan,
            dwdn,
            m,
            Val(size(cell)),
            Val(C);
            ndrange = size(dq),
        )
    end
end


function rhs!(dq, q, grid, crbc_data, invwJ, DT, materialparams, bc, cm)
    backend = Raven.get_backend(dq)
    cell = referencecell(grid)
    dRdX, _, wJ = components(first(volumemetrics(grid)))
    n, _, wsJ = components(first(surfacemetrics(grid)))
    fm = facemaps(grid)
    start!(q, cm)

    C = max(512 ÷ prod(size(cell)), 1)
    rhs_volume_kernel!(backend, (size(cell)..., C))(
        dq,
        q,
        dRdX,
        wJ,
        invwJ,
        DT,
        materialparams,
        Val(size(cell)),
        Val(C);
        ndrange = size(dq),
    )


    finish!(q, cm)

    J = maximum(size(cell))
    C = max(128 ÷ J, 1)
    # J  x C workgroup sizes to evaluate  multiple elements on one wg.
    rhs_surface_kernel!(backend, (J, C))(
        dq,
        viewwithghosts(q),
        crbc_data,
        fm.vmapM,
        fm.vmapP,
        bc,
        n,
        wsJ,
        invwJ,
        Val(size(cell)),
        Val(C);
        ndrange = (J, last(size(dq))),
    )
end

function run(
    ic,
    FT,
    AT,
    N,
    K,
    L;
    outputvtk = false,
    vtkdir = "output",
    comm = MPI.COMM_WORLD,
)
    rank = MPI.Comm_rank(comm)
    cell = LobattoCell{FT,AT}((N .+ 1)...)
    coordinates = (
        range(FT(-delta_x), stop = FT(delta_x), length = K + 1),
        range(FT(-delta_y), stop = FT(delta_y), length = K + 1),
    )

    periodicity = (true, false)
    gm = GridManager(cell, brick(coordinates, periodicity); comm = comm, min_level = L)
    grid = generate(gm)


    #jl # crude dt estimate
    cfl = 1 // 20
    dx = Base.step(first(coordinates))
    dt = cfl * dx / (11)^2

    numberofsteps = ceil(Int, timeend / dt)
    dt = timeend / numberofsteps

    RKA = (
        FT(0),
        FT(-567301805773 // 1357537059087),
        FT(-2404267990393 // 2016746695238),
        FT(-3550918686646 // 2091501179385),
        FT(-1275806237668 // 842570457699),
    )
    RKB = (
        FT(1432997174477 // 9575080441755),
        FT(5161836677717 // 13612068292357),
        FT(1720146321549 // 2090206949498),
        FT(3134564353537 // 4481467310338),
        FT(2277821191437 // 14882151754819),
    )

    if outputvtk
        rank == 0 && mkpath(vtkdir)
        pvd = rank == 0 ? paraview_collection("timesteps") : nothing
    end

    do_output = function (step, time, q, param, grid)
        if outputvtk && step % ceil(Int, timeend / 100 / dt) == 0
            par = GridArray{eltype(param)}(undef, grid)
            for i in eachindex(grid.points)
                par[i] = param[i]
            end
            cell = referencecell(grid)
            cd(vtkdir) do
                filename = "step$(lpad(step, 6, '0'))"
                vtkfile = vtk_grid(filename, grid)
                P = toequallyspaced(cell)
                vtkfile["|q|²"] = Adapt.adapt(Array, P * abs2.(q))
                vtkfile["m"] = Adapt.adapt(Array, P * par)
                vtk_save(vtkfile)
                if rank == 0
                    pvd[time] = vtkfile
                end
            end
        end
    end

    #jl # initialize state
    q = ic.(points(grid))

    #jl # storage for RHS
    dq = similar(q)
    dq .= Ref(zero(eltype(q)))

    #jl # precompute inverse of weights × Jacobian
    _, _, wJ = components(first(volumemetrics(grid)))
    invwJ = inv.(wJ)
    #jl # precompute derivative transpose
    DT = derivatives_1d(cell)

    materialparams_H = zeros(size(wJ))
    gridpoints_H = adapt(Array, grid.points)
    coefmap!(materialparams_H, gridpoints_H, m)
    materialparams = adapt(AT, materialparams_H)

    # adjust boundary code
    bc_H = adapt(Array, boundarycodes(grid))
    for j = 1:size(bc_H, 2)
        if last(gridpoints_H[end, end, j]) == delta_y  #North
            bc_H[4, j] = 2
        end

        if last(gridpoints_H[1, 1, j]) == -delta_y # South
            bc_H[3, j] = 2
        end
    end

    bc = adapt(AT, bc_H)
    cm = commmanager(eltype(q), nodecommpattern(grid); comm)

    #%%%%%%%%%%%%%#
    #  CRBC DATA  #
    #%%%%%%%%%%%%%#
    Q = 10
    crbc_cell = LobattoCell{FT,AT}(N[1] + 1, Q + 1)

    crbc_data = similar(q)
    crbc_data .= Ref(zero(eltype(q)))

    # TOP
    crbc_coordinates_top = (
        range(FT(-delta_x), stop = FT(delta_x), length = K * 2^L + 1),
        range(FT(delta_y), stop = FT(delta_y + 0.1), length = 2),
    )
    crbc_gm_top = GridManager(
        crbc_cell,
        brick(crbc_coordinates_top, (true, false));
        comm = comm,
        min_level = 0,
    )
    crbc_grid_top = generate(crbc_gm_top)
    crbc_q_top = GridArray{eltype(q)}(undef, crbc_grid_top)
    crbc_q_top .= Ref(zero(eltype(q)))
    crbc_dq_top = similar(crbc_q_top)
    crbc_dwdn_top = similar(crbc_q_top)
    crbc_dqdtan_top = similar(crbc_q_top)
    crbc_dq_top .= Ref(zero(eltype(q)))
    crbc_dwdn_top .= Ref(zero(eltype(q)))
    crbc_dqdtan_top .= Ref(zero(eltype(q)))

    _, _, crbc_wJ_top = components(first(volumemetrics(crbc_grid_top)))
    crbc_invwJ_top = inv.(crbc_wJ_top)
    crbc_DT_top = derivatives_1d(crbc_cell)

    crbc_interface_top = findall(≈(delta_y), last.(points(crbc_grid_top)))
    bulk_interface_top = findall(≈(delta_y), last.(points(grid)))
    @assert points(grid)[bulk_interface_top] ≈ points(crbc_grid_top)[crbc_interface_top]

    # BOTTOM
    crbc_coordinates_bottom = (
        range(FT(-delta_x), stop = FT(delta_x), length = K * 2^L + 1),
        range(FT(-delta_y - 0.1), stop = FT(-delta_y), length = 2),
    )
    crbc_gm_bottom = GridManager(
        crbc_cell,
        brick(crbc_coordinates_bottom, (true, false));
        comm = comm,
        min_level = 0,
    )
    crbc_grid_bottom = generate(crbc_gm_bottom)
    crbc_q_bottom = GridArray{eltype(q)}(undef, crbc_grid_bottom)
    crbc_q_bottom .= Ref(zero(eltype(q)))
    crbc_dq_bottom = copy(crbc_q_bottom)
    crbc_dwdn_bottom = copy(crbc_q_bottom)
    crbc_dqdtan_bottom = copy(crbc_q_bottom)
    _, _, crbc_wJ_bottom = components(first(volumemetrics(crbc_grid_bottom)))
    crbc_invwJ_bottom = inv.(crbc_wJ_bottom)
    crbc_DT_bottom = derivatives_1d(crbc_cell)

    crbc_interface_bottom = findall(≈(-delta_y), last.(points(crbc_grid_bottom)))
    bulk_interface_bottom = findall(≈(-delta_y), last.(points(grid)))
    @assert points(grid)[bulk_interface_bottom] ≈
            points(crbc_grid_bottom)[crbc_interface_bottom]

    #jl # initial output
    step = 0
    time = FT(0)

    data = MAT.matread("optimal_cosines_data.mat")
    a = data["a"]
    sig = data["sig"]

    do_output(step, time, q, materialparams_H, grid)

    for step in 1:numberofsteps
        if time + dt > timeend
            dt = timeend - time
        end
        crbc_dw1dn_top, _ = components(crbc_dwdn_top)
        crbc_dw1dn_bottom, _ = components(crbc_dwdn_bottom)
        q1, q2 = components(q)

        for stage in eachindex(RKA, RKB)
            @. dq *= RKA[stage]
            if BC == :crbc
                @. crbc_dq_top *= RKA[stage]
                @. crbc_dq_bottom *= RKA[stage]

                crbc_dw1dn_top[crbc_interface_top] .=
                    q1[bulk_interface_top] - im * q2[bulk_interface_top]
                crbc_dw1dn_bottom[crbc_interface_bottom] .=
                    -q1[bulk_interface_bottom] - im * q2[bulk_interface_bottom]

                crbc_rhs!(
                    crbc_dq_top,
                    crbc_q_top,
                    crbc_dwdn_top,
                    crbc_dqdtan_top,
                    crbc_grid_top,
                    crbc_invwJ_top,
                    crbc_DT_top,
                    m,
                    a,
                    sig,
                    cm;
                    orient = "top",
                )

                crbc_rhs!(
                    crbc_dq_bottom,
                    crbc_q_bottom,
                    crbc_dwdn_bottom,
                    crbc_dqdtan_bottom,
                    crbc_grid_bottom,
                    crbc_invwJ_bottom,
                    crbc_DT_bottom,
                    matpar == :half ? -m : m,
                    a,
                    sig,
                    cm;
                    orient = "bottom",
                )

                Pinv = SA[0.5 0.5; 0.5*im -0.5*im]
                crbc_data[bulk_interface_top] .=
                    [Pinv * i for i in crbc_dwdn_top[crbc_interface_top]]

                Pinv = SA[-0.5 -0.5; 0.5*im -0.5*im]
                crbc_data[bulk_interface_bottom] .=
                    [Pinv * i for i in crbc_dwdn_bottom[crbc_interface_bottom]]
            end

            rhs!(dq, q, grid, crbc_data, invwJ, DT, materialparams, bc, cm)
            @. q += RKB[stage] * dt * dq

            if BC == :crbc
                @. crbc_q_top += RKB[stage] * dt * crbc_dq_top
                @. crbc_q_bottom += RKB[stage] * dt * crbc_dq_bottom
            end
        end
        time += dt

        do_output(step, time, q, materialparams_H, grid)
    end

    do_output(numberofsteps, timeend, q, materialparams_H, grid)

    if outputvtk && rank == 0
        cd(vtkdir) do
            vtk_save(pvd)
        end
    end

    return q, wJ, time
end

let
    FT = Float64
    AT = CUDA.functional() && CUDA.has_cuda_gpu() ? CuArray : Array
    N = (4, 4)
    L = 2

    if !MPI.Initialized()
        MPI.Init()
    end

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)

    if CUDA.functional()
        CUDA.device!(MPI.Comm_rank(comm) % length(CUDA.devices()))
        CUDA.allowscalar(false)
    end

    if rank == 0
        @info """Configuration:
            precision        = $FT
            array type       = $AT

            Convergence test = $convergetest
            Outputvtk        = $outputvtk

            Base             = $K
            Level refine     = $L
            polynomial order = $N
            Material param   = $(String(matpar))
            m                = $m
            BC Type          = Periodic x $(String(BC))
            domain width     = $(2*delta_x)
            domain height    = $(2*delta_y)
            time end         = $(timeend)
        """
    end

    #jl # visualize solution of dirac wave 
    vtkdir = "vtk_semidg_dirac_2d$(K)x$(K)_L$(L)_$(String(BC))_$(String(matpar))_$(m)"
    if rank == 0
        @info """Starting Dirac solver with:
            ($K, $K) coarse grid
            $L refinement level
        """
    end

    _, _, tf = run(initialCondition, FT, AT, N, K, L; outputvtk = outputvtk, vtkdir, comm)
    outputvtk == true && rank == 0 && @info "Finished, vtk output written to $vtkdir"
    outputvtk == false && rank == 0 && @info "Finished initial sim w/ no output"

    if convergetest
        numlevels = 4
        N = (4, 4)

        cell = LobattoCell{FT,AT}((N .+ 1)...)
        (a, b) = tohalves_1d(cell)

        tohalves_q1 = Raven.Kron((b[1], a[1]))
        tohalves_q2 = Raven.Kron((b[1], a[2]))
        tohalves_q3 = Raven.Kron((b[2], a[1]))
        tohalves_q4 = Raven.Kron((b[2], a[2]))

        rank == 0 && @info "Starting Emperical convergence study h-refinement"
        err = zeros(FT, numlevels)

        for l = 1:numlevels
            L = l - 1
            totalcells = (K * 2^L, K * 2^L)
            dofs = prod(totalcells) * prod(1 .+ N)

            final_course, _, t1 =
                run(initialCondition, FT, AT, N, K, L; outputvtk = false, comm)
            final_fine, wJ, t2 =
                run(initialCondition, FT, AT, N, K, L + 1; outputvtk = false, comm)
            @assert t1 == t2 == tf
            fp_h = similar(final_fine)

            c = size(final_fine, 3)
            fp_h[:, :, 1:4:c] = tohalves_q1 * final_course
            fp_h[:, :, 2:4:c] = tohalves_q2 * final_course
            fp_h[:, :, 3:4:c] = tohalves_q3 * final_course
            fp_h[:, :, 4:4:c] = tohalves_q4 * final_course

            e1 = sqrt(
                sum(Adapt.adapt(Array, wJ .* getindex.(real.(fp_h .- final_fine), 1) .^ 2)),
            )
            e2 = sqrt(
                sum(Adapt.adapt(Array, wJ .* getindex.(real.(fp_h .- final_fine), 2) .^ 2)),
            )
            e3 = sqrt(
                sum(Adapt.adapt(Array, wJ .* getindex.(imag.(fp_h .- final_fine), 1) .^ 2)),
            )
            e4 = sqrt(
                sum(Adapt.adapt(Array, wJ .* getindex.(imag.(fp_h .- final_fine), 2) .^ 2)),
            )

            err[l] = maximum([e1, e2, e3, e4])

            if rank == 0
                @info @sprintf(
                    "Level %d, cells = (%2d, %2d), dof = %d, error = %.16e",
                    l,
                    totalcells...,
                    dofs,
                    err[l]
                )
            end
        end
        rates = log2.(err[1:numlevels-1] ./ err[2:numlevels])
        if rank == 0 && numlevels > 1
            @info "Convergence rates:\n" * join(
                ["rate for levels $l → $(l + 1) = $(rates[l])" for l = 1:(numlevels-1)],
                "\n",
            )
        end
    end
end
