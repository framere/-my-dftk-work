using Pkg
Pkg.activate(".")

using DFTK
using PsiTK
using PseudoPotentialData
using LinearAlgebra
using Printf

function main(method::String)
    pd_pbe_family = PseudoFamily("dojo.nc.sr.pbe.v0_5.stringent.upf") 

    He = ElementPsp(:He, pd_pbe_family)
    atoms = [He]
    box_length = 12.0
    lattice = [[ box_length  0.000000  0.00000]; 
               [ 0.00000  box_length-0.1  0.00000];
               [ 0.00000  0.000000  box_length-0.2]]
    positions = [[0.500000, 0.500000, 0.500000]] 
    Ecut = 40

    # start with a DFT-PBE 
    model  = model_PBE(lattice, atoms, positions)
    basis  = PlaneWaveBasis(model; Ecut=Ecut, kgrid=[1, 1, 1])
    println("run PBE")
    scfres_pbe = self_consistent_field(
        basis; 
        is_converged=ScfConvergenceEnergy(1e-7)
    )

    # use the PBE solution as initial guess for the HF solver
    model = model_HF(
        lattice, 
        atoms, 
        positions; 
        exx_kernel=Coulomb(ProbeCharge())
    )
    basis  = PlaneWaveBasis(model; Ecut=Ecut, kgrid=[1, 1, 1])
    println("run HF")
    scfres_hf = self_consistent_field(
                    basis;
                    solver=DFTK.scf_damping_solver(damping=1.0),
                    is_converged = ScfConvergenceEnergy(1e-7),
                    tol=1e-5, 
                    ρ=scfres_pbe.ρ, 
                    ψ=scfres_pbe.ψ, 
                    occupation = scfres_pbe.occupation,
                    maxiter=100, 
                    diagtolalg=DFTK.AdaptiveDiagtol(; ratio_ρdiff=5e-4),
                    exxalg=AceExx()
                )

    σ = 1  # non spin-polarized 
    ik = 1 # only one single k-point
    kpt = basis.kpoints[ik]
    orbitalType = eltype(scfres_hf.ψ[1]) # this is usually ComplexF64

    # we are interested in N*N_occ virtual orbitals
    N = 200

    # stochastic initial guess
    ϕk = construct_stochastic_orbitals(N, kpt, orbitalType)
    
    println("Compute DSVs")
    model  = model_HF(
        lattice, 
        atoms, 
        positions; 
        exx_kernel=Coulomb(ProbeCharge())
    )
    basis  = PlaneWaveBasis(model; Ecut=Ecut, kgrid=[1, 1, 1])
    ExactExchangeTerm = only([term for term in basis.terms if term isa DFTK.TermExactExchange])
    _, K = DFTK.ene_ops(ExactExchangeTerm, basis, scfres_hf.ψ, scfres_hf.occupation) # K = exchange operator
    Kk = K[ik] # we only look at our single k-point
     
    # We also need the occupied orbitals for level shifting
    ψocc, occupation_occ = DFTK.select_occupied_orbitals(basis, scfres_hf.ψ, scfres_hf.occupation; threshold=1e-8)
    ψocck = ψocc[ik] # we only look at ik=1

    Nfull = length(kpt.G_vectors)

    # set up Kk_virt
    ε_homo = maximum(scfres_hf.eigenvalues[ik][scfres_hf.occupation[ik] .> 1e-6])
    ham_hf_levelshifted = LevelShiftedOperator(scfres_hf.ham[ik], ψocck, ε_homo, 1e-5, 2*Ecut)
    shift = abs(minimum(minimum.(scfres_hf.eigenvalues))) + 2.0 # lowest HF eigenvalue + 2.0 Ha
    Kk_virt = ProjectedShiftedOperator(Kk, ψocck, shift)
    kinetic_preconditioner = PreconditionerTPA(scfres_hf.ham[ik].basis, kpt)
    # D_real = DFTK.precondprep!(kinetic_preconditioner, nothing)
    
    # run LOBPCG for DSV's
    # this solves the equation Kk_virt * f = ham_hf_levelshifted * λ * f 
    if method == "LOBPCG"
        @time dsv = DFTK.LOBPCG(
            Kk_virt, 
            ϕk, 
            ham_hf_levelshifted, 
            kinetic_preconditioner, 
            1e-5, 
            500, 
            callback=DFTK.DefaultLobpcgCallback()
        )  
        X_dsv = dsv.X
        qr_decomp = qr(X_dsv)
        X_ortho = Matrix(qr_decomp.Q)
        h_dsv = Hermitian(X_ortho' * (scfres_hf.ham[ik] * X_ortho))
        canonical_dsv_res = eigen(h_dsv)
        N_occ = size(ψocck,2)
        N_dsv = size(canonical_dsv_res.vectors,2)
        ψvirtk = X_ortho * canonical_dsv_res.vectors
        
    elseif method == "Davidson"
        println("Run Davidson for DSVs")
        @time Σ_dsv, X_dsv = davidson(Kk_virt, ϕk, ψocck, N*6, 1e-5)

        # we finally re-canonicalize the virtual DSV orbitals
        println("Recanonicalize DSVs.")
        h_dsv = scfres_hf.ham[ik] * X_dsv
        h_dsv = X_dsv' * h_dsv
        h_dsv = Hermitian(h_dsv)
        canonical_dsv_res = eigen(h_dsv)

        N_occ = size(ψocck,2)
        N_dsv = size(canonical_dsv_res.vectors,2)

        ψvirtk = X_dsv * canonical_dsv_res.vectors
    end 
    ψ_cc4s = hcat(ψocck, ψvirtk)
    ε_cc4s = vcat(scfres_hf.eigenvalues[ik][1:N_occ], canonical_dsv_res.values)
    occupation_cc4s = vcat(occupation_occ[ik], zeros(N_dsv))

    println("prepare and dump Cc4s files")

    res_cc4s = (
        X = ψ_cc4s,
        λ = ε_cc4s,
        converged = true,
        n_iter = 1
    )

    cc4s_bands = (;
        basis = basis,
        ψ = [res_cc4s.X],
        eigenvalues = [res_cc4s.λ],
        ρ = scfres_hf.ρ,
        εF = scfres_hf.εF,
        occupation = [occupation_cc4s],
        diagonalization = [res_cc4s],
        n_bands_converge = N_occ + N_dsv
    )

    dump_cc4s_files(cc4s_bands, "."; force=true, auxfield_thresh=1e-6)
    println("done")
end

function construct_stochastic_orbitals(N, kpt, orbitalType)
    NG = length(kpt.G_vectors)
    radius = rand(NG,N)
    phase = cis.(2π .* rand(NG,N))
    ϕk = zeros(orbitalType, length(kpt.G_vectors), N)
    ϕk = radius .* phase
    for a in 1:N
        ϕk[:,a] ./= norm(ϕk[:,a]) # normalize
    end
    # orthogonalize
    qr_decomp = qr(ϕk)
    ϕk = Matrix(qr_decomp.Q)
end


# Level shifted Operator (Shift + Penalty)
struct LevelShiftedOperator{TH, TV}
    base_op::TH                      # the operator
    V::TV                            # basis for penalty projector
    ε_homo::Float64                  # energy of HOMO           
    safe_shift::Float64              # avoid zero gap (~ 1e-5 Ha)
    penalty::Float64                 # should be 2*Ecut
end
function LevelShiftedOperator(base_op, V, ε_homo, safe_shift, penalty)
    return LevelShiftedOperator{typeof(base_op), typeof(V)}(
        base_op, V, ε_homo, safe_shift, penalty
    )
end
function LinearAlgebra.mul!(Y, op::LevelShiftedOperator, X)
    mul!(Y, op.base_op, X)
    Y .+= (-op.ε_homo + op.safe_shift) .* X
    
    # Projector penalty
    coeffs = op.V' * X   
    mul!(Y, op.V, coeffs, op.penalty, 1.0)
    return Y
end
function Base.:*(op::LevelShiftedOperator, X::AbstractMatrix)
    Y = similar(X)
    mul!(Y, op, X) 
    return Y
end
Base.size(op::LevelShiftedOperator, args...) = size(op.base_op, args...)
Base.eltype(op::LevelShiftedOperator) = eltype(op.base_op)
LinearAlgebra.ishermitian(op::LevelShiftedOperator) = ishermitian(op.base_op)
 

# Projected and shifted operator (using (1-VV') as projector)
struct ProjectedShiftedOperator{TOp, TV}
    base_op::TOp       # the operator
    V::TV              # basis for projector
    shift::Float64     # should be a lilttle larger than the lower epsilon_hf
end
function ProjectedShiftedOperator(base_op, V, shift)
    return ProjectedShiftedOperator{typeof(base_op), typeof(V)}(
        base_op, V, shift
    )
end
function LinearAlgebra.mul!(Y, op::ProjectedShiftedOperator, X)
    workX = similar(X)
    copy!(workX, X)
    coeffs_in = op.V' * X
    mul!(workX, op.V, coeffs_in, -1.0, 1.0) # workX -= op.V * coeffs_in
    mul!(Y, op.base_op, workX)              #     Y  = op.base_op * workX
    coeffs_out = op.V' * Y
    mul!(Y, op.V, coeffs_out, -1.0, 1.0)    #     Y -= op.V * coeffs_out
    mul!(Y, op.V, coeffs_in, op.shift, 1.0) #     Y += shift*coeffs_in
    return Y
end
function Base.:*(op::ProjectedShiftedOperator, X::AbstractMatrix)
    Y = similar(X)
    mul!(Y, op, X) 
    return Y
end
Base.size(op::ProjectedShiftedOperator, args...) = size(op.base_op, args...)
Base.eltype(op::ProjectedShiftedOperator) = eltype(op.base_op)
LinearAlgebra.ishermitian(op::ProjectedShiftedOperator) = ishermitian(op.base_op)

function davidson(
    A::ProjectedShiftedOperator,
    # D_real::AbstractArray{<:Real, 3},
    V::AbstractMatrix{T},
    ψocck::AbstractMatrix{T},
    Naux::Integer,
    thresh::Float64
)::Tuple{Vector{T},Matrix{T}} where T<:Number


    Nlow = size(V, 2)
    if Naux < Nlow
        println("ERROR: auxiliary basis must not be smaller than number of target eigenvalues")
    end
    basis = A.base_op.basis
    kpt   = A.base_op.kpoint

    iter = 0
    while true
        iter += 1

        qr_decomp = qr(V)
        V = Matrix(qr_decomp.Q)

        H = V' * (A * V)
        H = Hermitian(H)
        Σ, U = eigen(H, 1:Nlow)
        X = V * U
        R = X .* Σ' - A * X
        Rnorm = norm(R, 2)

        output = @sprintf("iter=%6d  Rnorm=%11.3e  size(V,2)=%6d\n", iter, Rnorm, size(V, 2))
        print(output)

        if Rnorm < thresh
            println("converged!")
            return (Σ, X)
        end

        # Preconditioner (currently identity; uncomment block below to activate)
        # t = zero(similar(R)) 
        # for i = 1:size(t,2)
        #    R_real = ifft(basis, kpt, R[:,i]) # FFT to real space
        #    C = -1.0 ./ (D_real .- Σ[i])
        #    t_real = C .* R_real # apply C
        #    t[:,i] = fft(basis, kpt, t_real) # FFT back to reciprocal space
        # end
        
        t = R # no preconditioner

        # Expand or restart the search space
        if size(V, 2) <= Naux - Nlow
            V = hcat(V, t)
        else
            V = hcat(X, t)
        end
    end
end

main("LOBPCG")
