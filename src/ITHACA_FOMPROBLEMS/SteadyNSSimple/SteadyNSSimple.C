/*---------------------------------------------------------------------------*\
     ██╗████████╗██╗  ██╗ █████╗  ██████╗ █████╗       ███████╗██╗   ██╗
     ██║╚══██╔══╝██║  ██║██╔══██╗██╔════╝██╔══██╗      ██╔════╝██║   ██║
     ██║   ██║   ███████║███████║██║     ███████║█████╗█████╗  ██║   ██║
     ██║   ██║   ██╔══██║██╔══██║██║     ██╔══██║╚════╝██╔══╝  ╚██╗ ██╔╝
     ██║   ██║   ██║  ██║██║  ██║╚██████╗██║  ██║      ██║      ╚████╔╝
     ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝      ╚═╝       ╚═══╝

 * In real Time Highly Advanced Computational Applications for Finite Volumes
 * Copyright (C) 2017 by the ITHACA-FV authors
-------------------------------------------------------------------------------

  License
  This file is part of ITHACA-FV

  ITHACA-FV is free software: you can redistribute it and/or modify
  it under the terms of the GNU Lesser General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  ITHACA-FV is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
  GNU Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public License
  along with ITHACA-FV. If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/


/// \file
/// Source file of the steadyNS class.

#include "SteadyNSSimple.H"
#include "viscosityModel.H"

// * * * * * * * * * * * * * * * Constructors * * * * * * * * * * * * * * * * //

// Constructor
SteadyNSSimple::SteadyNSSimple() {}

SteadyNSSimple::SteadyNSSimple(int argc, char* argv[])
    :
    steadyNS(argc, argv)
{
    Info << offline << endl;
}

// fvVectorMatrix SteadyNSSimple::get_Umatrix(volVectorField& U,
//         volScalarField& p)
// {
//     surfaceScalarField& phi = _phi();
//     fv::options& fvOptions = _fvOptions();
//     fvVectorMatrix Ueqn
//     (
//         fvm::div(phi, U)
//         + turbulence->divDevReff(U)
//     );
//     Ueqn.relax();
//     Ueqn_global = &Ueqn;
//     return Ueqn;
// }

// fvScalarMatrix SteadyNSSimple::get_Pmatrix(volVectorField& U,
//         volScalarField& p, scalar& presidual)
// {
//     surfaceScalarField& phi = _phi();
//     simpleControl& simple = _simple();
//     fvMesh& mesh = _mesh();
//     int i = 0;

//     while (simple.correctNonOrthogonal())
//     {
//         fvScalarMatrix pEqn
//         (
//             fvm::laplacian(rAtU(), p) == fvc::div(phiHbyA)
//         );
//         pEqn.setReference(pRefCell, pRefValue);

//         if (i == 0)
//         {
//             presidual = pEqn.solve().initialResidual();
//         }

//         else
//         {
//             pEqn.solve().initialResidual();
//         }

//         if (simple.finalNonOrthogonalIter())
//         {
//             phi = phiHbyA - pEqn.flux();
//         }

//         i++;
//     }

//     //p.storePrevIter(); // Perché ho dovuto metterlo se nel solver non c'è???
//     p.relax();
//     U = HbyA - rAtU() * fvc::grad(p);
//     U.correctBoundaryConditions();
//     fvOptions.correct(U);
//     fvScalarMatrix pEqn
//     (
//         fvm::laplacian(rAtU(), p) == fvc::div(phiHbyA)
//     );
//     return pEqn;
// }

void SteadyNSSimple::truthSolve2(List<scalar> mu_now, word Folder)
{
    Time& runTime = _runTime();
    volScalarField& p = _p();
    volVectorField& U = _U();
    fvMesh& mesh = _mesh();
    surfaceScalarField& phi = _phi();
    fv::options& fvOptions = _fvOptions();
    simpleControl& simple = _simple();
    singlePhaseTransportModel& laminarTransport = _laminarTransport();
    scalar residual = 1;
    scalar uresidual = 1;
    Vector<double> uresidual_v(0, 0, 0);
    scalar presidual = 1;
    scalar csolve = 0;
    // Variable that can be changed
    turbulence->read();
    std::ofstream res_os;
    res_os.open("./ITHACAoutput/Offline/residuals", std::ios_base::app);
#if OFVER == 6

    while (simple.loop(runTime) && residual > tolerance && csolve < maxIter )
#else
    while (simple.loop() && residual > tolerance && csolve < maxIter )
#endif
    {
        Info << "Time = " << runTime.timeName() << nl << endl;
        // --- Pressure-velocity SIMPLE corrector
        // Momentum predictor
        fvVectorMatrix UEqn
        (
            fvm::div(phi, U)
            + turbulence->divDevReff(U)
            ==
            -fvc::grad(p)
        );
        UEqn.relax();
        uresidual_v = UEqn.solve().initialResidual();
        phi = fvc::flux(1 / UEqn.A() * UEqn.H());
        int i = 0;

        // Non-orthogonal pressure corrector loop
        while (simple.correctNonOrthogonal())
        {
            fvScalarMatrix pEqn
            (
                fvm::laplacian(1 / UEqn.A(), p) == fvc::div(phi)
            );
            pEqn.setReference(pRefCell, pRefValue);

            if (i == 0)
            {
                presidual = pEqn.solve().initialResidual();
            }

            else
            {
                pEqn.solve();
            }

            if (simple.finalNonOrthogonalIter())
            {
                phi -= pEqn.flux();
            }

            i++;
        }

        //#include "continuityErrs.H"
        // Explicitly relax pressure for momentum corrector
        p.relax();
        scalar C = 0;

        for (label i = 0; i < 3; i++)
        {
            if (C < uresidual_v[i])
            {
                C = uresidual_v[i];
            }
        }

        uresidual = C;
        residual = max(presidual, uresidual);
        Info << "\nResidual: " << residual << endl << endl;
        // Momentum corrector
        //U.correctBoundaryConditions();
        Info << "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
             << "  ClockTime = " << runTime.elapsedClockTime() << " s"
             << nl << endl;
    }

    ITHACAstream::exportSolution(U, name(counter), Folder);
    ITHACAstream::exportSolution(p, name(counter), Folder);
    Ufield.append(U);
    Pfield.append(p);
    counter++;
    writeMu(mu_now);
    // --- Fill in the mu_samples with parameters (mu) to be used for the POD sample points
    mu_samples.conservativeResize(mu_samples.rows() + 1, mu_now.size());

    for (int i = 0; i < mu_now.size(); i++)
    {
        mu_samples(mu_samples.rows() - 1, i) = mu_now[i];
    }

    // Resize to Unitary if not initialized by user (i.e. non-parametric problem)
    if (mu.cols() == 0)
    {
        mu.resize(1, 1);
    }

    if (mu_samples.rows() == mu.cols())
    {
        ITHACAstream::exportMatrix(mu_samples, "mu_samples", "eigen",
                                   Folder);
    }
}