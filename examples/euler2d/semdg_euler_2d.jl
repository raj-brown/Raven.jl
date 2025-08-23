#--------------------------------Markdown Language Header-----------------------
# # 2D Euler Equation
#FIXME: add convergence study
#FIXME: add to docs
#FIXME: optimize kernels
#--------------------------------Markdown Language Header-----------------------
using Base: sign_mask
using Adapt
using MPI
using CUDA
using LinearAlgebra
using Printf
using Raven
using StaticArrays
using WriteVTK
using KernelAbstractions
using KernelAbstractions.Extras: @unroll


const outputvtk = false

const γ = 1.4

function initialcondition(x::SVector{2}, time)
    FT = eltype(x)
    xₒ = 5.0
    yₒ = 0.0
    β = 5.0
    γ = 1.4
    ρ = 1.0
    u = 1.0
    v = 0.0 
    p=1
    xmut = x - u*time
    ymvt = y - v*time
    r = sqrt((xmut -xₒ)^2 + (ymvt-yₒ)^2)
    u = u - β*exp(1-r^2)*(ymvt - yₒ)/(2*π)
    v = v + β*exp(1-r^2)*(xmut - xₒ)/(2*π)
    ρ₁ = (1 - ((γ-1)/(16*γ*pi^2))*β^2*exp(2*(1-r^2)))
    p₁ = ρ₁^gamma
    ρ = ρ₁
    ρu = ρ₁*u
    ρv = ρ₁*v
    e = p₁/(γ-1) + 0.5*ρ₁*(u^2 + v^2)
    return SVector{4,FT}(ρ, ρu, ρv, e)


function rhs_volume_kernel!(dq, q, dRdX, wJ, invwJ, DT, ::Val{N}, ::Val{C}) where {N,C}
    i, j, cl = @index(Local, NTuple)
    _, _, c =  @index(Global,NTuple)

    lDT₁ = @localmem eltype(q) (N[1], N[1])
    lDT₂ = @localmem eltype(q) (N[2], N[2])

    lf₁ = @localmem eltype(q) (N..., C)
    lf₂ = @localmem eltype(q) (N..., C)
    lf₃ = @localmem eltype(q) (N..., C)
    lf₄ = @localmem eltype(q) (N..., C)

    lg₁ = @localmem eltype(q) (N..., C)
    lg₂ = @localmem eltype(q) (N..., C)
    lg₃ = @localmem eltype(q) (N..., C)
    lg₄ = @localmem eltype(q) (N..., C)

    @inbounds begin
        for sj = 0x0:N[2]:(N[1]-0x1)
            if j + sj <= N[1] && cl == 0x1
                lDT1[i, j+sj] = DT[1][i, j+sj]
            end
        end

        for si = 0x0:N[1]:(N[2]-0x1)
            if i + si <= N[2] && cl == 0x1
                lDT2[i+si, j] = DT[2][i+si, j]
            end
        end

        ρ = q[i, j, 0x1, c]
        ρu = q[i, j, 0x2, c]
        ρv = q[i, j, 0x3, c]
        u = ρu/ρ
        v = ρv/ρ
        e = q[i, j, 0x4, c]
        p = (γ -1)*(e - 0.5*(ρu*u + ρv*v))

        lf₁[i, j, cl] = ρu
        lf₂[i, j, cl] = ρu*u + p        
        lf₃[i, j, cl] = ρu*v
        lf₄[i, j, cl] = u*(e + p)


        lg₁[i, j, cl] = ρv
        lg₂[i, j, cl] = ρ*u*v 
        lg₃[i, j, cl] = ρv*v + p
        lg₃[i, j, cl] = v*(e + p)

    end

    sync_threads()

    @inbounds begin
        df₁_update = zero(eltype(dq))
        df₂_update = zero(eltype(dq))
        df₃_update = zero(eltype(dq))
        df₄_update = zero(eltype(dq))
        invwJijc = invwJ[i, j, 0x1, c]
        wJijc = wJ[i, j, 0x1, c]

        #dρdt = rₓDᵣ(ρu) + sₓD(ρu)
        
        
        #dpdt = -c ((r_x D_r + s_x D_s) ux + (r_y D_r + s_y D_s) uy)
        #duxdt = -c p_x = -c (r_x D_r + s_x D_s) p
        #duydt = -c p_y = -c (r_y D_r + s_y D_s) p

        for m = 0x1:N[1]
            df₁_update += dRdX[i, j, 0x1, c] * lDT1[i, m] * lf₁[m, j, cl] # rₓ(Dᵣρu)
            df₁_update += dRdX[i, j, 0x3, c] * lDT1[i, m] * lg₁[m, j, cl] # rᵥ(Dᵣρv)

            df₂_update += dRdX[i, j, 0x1, c] * lDT1[i, m] * lf₂[m, j, cl] # rₓDᵣ(ρu²+p) 
            df₂_update += dRdX[i, j, 0x3, c] * lDT1[i, m] * lg₂[m, j, cl] # rᵥDᵣ(ρuv) 

            df₃_update += dRdX[i, j, 0x1, c] * lDT1[i, m] * lf₃[m, j, cl] # rₓDᵣ(ρuv)
            df₃_update += dRdX[i, j, 0x3, c] * lDT1[i, m] * lg₃[m, j, cl] # rᵥDᵣ(ρv²+p)

            df₄_update += dRdX[i, j, 0x1, c] * lDT1[i, m] * lf₄[m, j, cl] # rₓDᵣ(u[E+p])
            df₄_update += dRdX[i, j, 0x3, c] * lDT1[i, m] * lg₄[m, j, cl] # rᵥDᵣ(v[E+p])
        end

        for n = 0x1:N[2]
            df₁_update += dRdX[i, j, 0x2, c] * lDT2[j, n] * lf₁[i, n, cl] #  sₓDₛ(ρu)
            df₁_update += dRdX[i, j, 0x4, c] * lDT2[j, n] * lg₁[i, n, cl] #  sᵥDₛ(ρv)

            df₂_update += dRdX[i, j, 0x2, c] * lDT2[j, n] * lf₂[i, n, cl] #  sₓDₛ(ρu²+p)
            df₂_update += dRdX[i, j, 0x4, c] * lDT2[j, n] * lg₂[i, n, cl] #  sᵥDₛ(ρuv)

            df₃_update += dRdX[i, j, 0x2, c] * lDT2[j, n] * lf₃[i, n, cl] #  sₓDₛ(ρuv)
            df₃_update += dRdX[i, j, 0x4, c] * lDT2[j, n] * lg₃[i, n, cl] #  sᵥDₛ(ρv²+p)

            df₄_update += dRdX[i, j, 0x2, c] * lDT2[j, n] * lf₄[i, n, cl] #  sₓDₛ(u[E+p])
            df₄_update += dRdX[i, j, 0x4, c] * lDT2[j, n] * lg₄[i, n, cl] #  sᵥDₛ(v[E+p])         
        end

        dq[i, j, 0x1, c] += wJijc * invwJijc * df₁_update
        dq[i, j, 0x2, c] += wJijc * invwJijc * df₂_update
        dq[i, j, 0x3, c] += wJijc * invwJijc * df₃_update
        dq[i, j, 0x4, c] += wJijc * invwJijc * df₄_update

    end

    return nothing
end

function rhs_surface_kernel!(
    dq,
    q,
    vmapM,
    vmapP,
    mapB,
    n,
    wsJ,
    invwJ,
    ::Val{N},
    ::Val{C},
) where {N,C}
    ij, cl = threadIdx()
    c = (blockIdx().x - 1) * blockDim().y + cl

    lpflux = CuStaticSharedArray(eltype(dq), (N..., C))
    luxflux = CuStaticSharedArray(eltype(dq), (N..., C))
    luyflux = CuStaticSharedArray(eltype(dq), (N..., C))

    @inbounds if ij <= N[1]
        i = ij
        for j = 1:N[2]
            lpflux[i, j, cl] = zero(eltype(lpflux))
            luxflux[i, j, cl] = zero(eltype(lpflux))
            luyflux[i, j, cl] = zero(eltype(lpflux))
        end
    end

    sync_threads()

    @inbounds if ij <= N[2]
        # Faces with r=-1 : West
        i = 1
        j = ij
        fid = j

        idM = vmapM[fid, c]
        idP = vmapP[fid, c]
        idB = mapB[1, c]

        nx = n[0x1, fid, c]
        ny = n[0x2, fid, c]

        wsJf = wsJ[0x1, fid, c]

        if idB == 1
            dp = zero(eltype(lpflux)) # pM == pP
            ndotu = nx * q[i, j, 0x2, c] + ny * q[i, j, 0x3, c]
            dux = ndotu * nx
            duy = ndotu * ny
        elseif idB == 0
            Pc, Pij = fldmod1(idP, N[1] * N[2])
            Pj, Pi = fldmod1(Pij, N[1])

            dp = q[i, j, 0x1, c] - q[Pi, Pj, 0x1, Pc]
            dux = q[i, j, 0x2, c] - q[Pi, Pj, 0x2, Pc]
            duy = q[i, j, 0x3, c] - q[Pi, Pj, 0x3, Pc]
        end

        invwJijc = invwJ[i, j, 0x1, c]

        fscale = invwJijc * wsJf / 2

        lpflux[i, j, cl] += fscale * (nx * dux + ny * duy)
        luxflux[i, j, cl] += fscale * nx * dp
        luyflux[i, j, cl] += fscale * ny * dp

        # Faces with r=1 : East 
        i = N[1]
        j = ij
        fid = N[2] + j

        idM = vmapM[fid, c]
        idP = vmapP[fid, c]

        nf = n[fid, c]
        wsJf = wsJ[fid, c]

        idB = mapB[2, c]

        nx = n[0x1, fid, c]
        ny = n[0x2, fid, c]

        wsJf = wsJ[0x1, fid, c]

        if idB == 1
            dp = zero(eltype(lpflux)) # pM == pP
            ndotu = nx * q[i, j, 0x2, c] + ny * q[i, j, 0x3, c]
            dux = ndotu * nx
            duy = ndotu * ny
        elseif idB == 0
            Pc, Pij = fldmod1(idP, N[1] * N[2])
            Pj, Pi = fldmod1(Pij, N[1])

            dp = q[i, j, 0x1, c] - q[Pi, Pj, 0x1, Pc]
            dux = q[i, j, 0x2, c] - q[Pi, Pj, 0x2, Pc]
            duy = q[i, j, 0x3, c] - q[Pi, Pj, 0x3, Pc]
        end

        invwJijc = invwJ[i, j, 0x1, c]

        fscale = invwJijc * wsJf / 2

        lpflux[i, j, cl] += fscale * (nx * dux + ny * duy)
        luxflux[i, j, cl] += fscale * nx * dp
        luyflux[i, j, cl] += fscale * ny * dp
    end

    sync_threads()

    @inbounds if ij <= N[1]
        # Faces with s=-1 : South 
        i = ij
        j = 1
        fid = 2N[2] + i

        idM = vmapM[fid, c]
        idP = vmapP[fid, c]
        idB = mapB[3, c]

        nx = n[0x1, fid, c]
        ny = n[0x2, fid, c]

        wsJf = wsJ[0x1, fid, c]

        if idB == 1
            dp = zero(eltype(lpflux)) # pM == pP
            ndotu = nx * q[i, j, 0x2, c] + ny * q[i, j, 0x3, c]
            dux = ndotu * nx
            duy = ndotu * ny
        elseif idB == 0
            Pc, Pij = fldmod1(idP, N[1] * N[2])
            Pj, Pi = fldmod1(Pij, N[1])

            dp = q[i, j, 0x1, c] - q[Pi, Pj, 0x1, Pc]
            dux = q[i, j, 0x2, c] - q[Pi, Pj, 0x2, Pc]
            duy = q[i, j, 0x3, c] - q[Pi, Pj, 0x3, Pc]
        end

        invwJijc = invwJ[i, j, 0x1, c]

        fscale = invwJijc * wsJf / 2

        lpflux[i, j, cl] += fscale * (nx * dux + ny * duy)
        luxflux[i, j, cl] += fscale * nx * dp
        luyflux[i, j, cl] += fscale * ny * dp

        i = ij
        j = N[2]
        fid = 2N[2] + N[1] + i

        idM = vmapM[fid, c]
        idP = vmapP[fid, c]
        idB = mapB[0x4, c]

        nx = n[0x1, fid, c]
        ny = n[0x2, fid, c]

        wsJf = wsJ[0x1, fid, c]

        if idB == 1
            dp = zero(eltype(lpflux)) # pM == pP
            ndotu = nx * q[i, j, 0x2, c] + ny * q[i, j, 0x3, c]
            dux = ndotu * nx
            duy = ndotu * ny
        elseif idB == 0
            Pc, Pij = fldmod1(idP, N[1] * N[2])
            Pj, Pi = fldmod1(Pij, N[1])

            dp = q[i, j, 0x1, c] - q[Pi, Pj, 0x1, Pc]
            dux = q[i, j, 0x2, c] - q[Pi, Pj, 0x2, Pc]
            duy = q[i, j, 0x3, c] - q[Pi, Pj, 0x3, Pc]
        end

        invwJijc = invwJ[i, j, 0x1, c]

        fscale = invwJijc * wsJf / 2

        lpflux[i, j, cl] += fscale * (nx * dux + ny * duy)
        luxflux[i, j, cl] += fscale * nx * dp
        luyflux[i, j, cl] += fscale * ny * dp
    end

    sync_threads()

    # RHS update
    i = ij
    @inbounds if i <= N[1]
        for j = 1:N[2]
            dq[i, j, 0x1, c] += lpflux[i, j, cl]
            dq[i, j, 0x2, c] += luxflux[i, j, cl]
            dq[i, j, 0x3, c] += luyflux[i, j, cl]
        end
    end
    return nothing
end

function rhs!(dq, q, grid, invwJ, DT, mapB, cm)
    backend = Raven.get_backend(dq)
    cell = referencecell(grid)
    dRdX, _, wJ = components(first(volumemetrics(grid)))
    n, _, wsJ = components(first(surfacemetrics(grid)))
    fm = facemaps(grid)

    start!(q, cm)
    C = max(512 ÷ prod(size(cell)), 1)
    b = cld(last(size(dq)), C)


    @cuda threads = (size(cell)..., C) blocks = b rhs_volume_kernel!(
        parent(dq),
        parent(q),
        parent(dRdX),
        parent(wJ),
        parent(invwJ),
        DT,
        Val(size(cell)),
        Val(C),
    )


    finish!(q, cm)

    J = maximum(size(cell))
    C = max(128 ÷ J, 1)
    # J  x C workgroup sizes to evaluate  multiple elements on one wg.
    @cuda threads = (J, C) blocks = cld(last(size(dq)), C) rhs_surface_kernel!(
        parent(dq),
        parent(viewwithghosts(q)),
        fm.vmapM,
        fm.vmapP,
        mapB,
        parent(n),
        parent(wsJ),
        parent(invwJ),
        Val(size(cell)),
        Val(C),
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
    coordinates = ntuple(_ -> range(FT(0), stop = FT(10), length = K + 1), 2)
    periodicity = (false, false)
    gm = GridManager(cell, brick(coordinates, periodicity); comm = comm, min_level = L)
    grid = generate(gm)

    timeend = FT(2π)

    #jl # crude dt estimate
    cfl = 1 // 20
    dx = Base.step(first(coordinates))
    dt = cfl * dx / (maximum(N))^2

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

    do_output = function (step, time, q)
        if outputvtk && step % ceil(Int, timeend / 100 / dt) == 0
            cd(vtkdir) do
                filename = "step$(lpad(step, 6, '0'))"
                vtkfile = vtk_grid(filename, grid)
                P = toequallyspaced(cell)
                vtkfile["q"] = Adapt.adapt(Array, P * q)
                vtk_save(vtkfile)
                if rank == 0
                    pvd[time] = vtkfile
                end
            end
        end
    end

    #jl # initialize state
    q = ic.(points(grid))
    @show size(q)
    dq = similar(q)
    dq .= Ref(zero(eltype(q)))

    #jl # precompute inverse of weights × Jacobian
    _, _, wJ = components(first(volumemetrics(grid)))
    invwJ = inv.(wJ)

    #jl # precompute derivative transpose
    DT = derivatives_1d(cell)

    @show size(DT[1])

    gridpoints_H = adapt(Array, grid.points)

    # adjust boundary code
    mapB_H = adapt(Array, boundarycodes(grid))
    for j = 1:size(mapB_H, 2)
        if last(gridpoints_H[end, end, j]) == 1  #North
            mapB_H[4, j] = 1
        end

        if last(gridpoints_H[1, 1, j]) == -1 # South
            mapB_H[3, j] = 1
        end

        if first(gridpoints_H[end, end, j]) == 1 # East 
            mapB_H[2, j] = 1
        end

        if first(gridpoints_H[1, 1, j]) == -1 # West 
            mapB_H[1, j] = 1
        end
    end

    mapB = adapt(AT, mapB_H)

    cm = commmanager(eltype(q), nodecommpattern(grid); comm)

    #jl # initial output
    step = 0
    time = FT(0)

    do_output(step, time, q)

    for step = 1:numberofsteps
        if time + dt > timeend
            dt = timeend - time
        end

        for stage in eachindex(RKA, RKB)
            @. dq *= RKA[stage]
            rhs!(dq, q, grid, invwJ, DT, mapB, cm)
            @. q += RKB[stage] * dt * dq
        end
        time += dt

        do_output(step, time, q)
    end

    #jl # final output
    do_output(numberofsteps, timeend, q)
    if outputvtk && rank == 0
        cd(vtkdir) do
            vtk_save(pvd)
        end
    end
end

let
    FT = Float64
    @assert CUDA.functional() && CUDA.has_cuda_gpu() "Nvidia GPU not available"
    AT = CuArray
    N = (7, 7)
    K = 4
    L = 4

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
            polynomial order = $N
            array type       = $AT
        """
    end

    #jl # visualize solution
    vtkdir = "vtk_semidg_acoustic_2d$(K)x$(K)_L$(L)"
    if rank == 0
        @info """Starting Gaussian advection with:
            ($K, $K) coarse grid
            $L refinement level
        """
    end

    run(initialcondition, FT, AT, N, K, L; outputvtk = outputvtk, vtkdir, comm)
    rank == 0 && @info "Finished, vtk output written to $vtkdir"
end
