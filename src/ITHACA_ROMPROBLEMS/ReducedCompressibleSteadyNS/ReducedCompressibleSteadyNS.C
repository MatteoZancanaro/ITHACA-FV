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
/// Source file of the reducedSteadyNS class

#include "ReducedCompressibleSteadyNS.H"

// * * * * * * * * * * * * * * * Constructors * * * * * * * * * * * * * * * * //

// Constructor
ReducedCompressibleSteadyNS::ReducedCompressibleSteadyNS()
{
}

ReducedCompressibleSteadyNS::ReducedCompressibleSteadyNS(
    CompressibleSteadyNS& FOMproblem)
    :
    problem(&FOMproblem)
{
    // Create a new Umodes set where the first ones are the lift functions

    for (label i = 0; i < problem->liftfield.size(); i++)
    {
        ULmodes.append(problem->liftfield[i]);
    }

    for (label i = 0; i < problem->Umodes.size(); i++)
    {
        ULmodes.append(problem->Umodes.toPtrList()[i]);
    }
}

void ReducedCompressibleSteadyNS::setOnlineVelocity(Eigen::MatrixXd vel)
{
    M_Assert(problem->inletIndex.rows() == vel.size(),
             "Imposed boundary conditions dimensions do not match given values matrix dimensions");
    Eigen::MatrixXd vel_scal;
    vel_scal.resize(vel.rows(), vel.cols());

    for (int k = 0; k < problem->inletIndex.rows(); k++)
    {
        label p = problem->inletIndex(k, 0);
        label l = problem->inletIndex(k, 1);
        scalar area = gSum(problem->liftfield[0].mesh().magSf().boundaryField()[p]);
        scalar u_lf = gSum(problem->liftfield[k].mesh().magSf().boundaryField()[p] *
                           problem->liftfield[k].boundaryField()[p]).component(l) / area;
        vel_scal(k, 0) = vel(k, 0) / u_lf;
    }
    vel_now = vel_scal;
}

// * * * * * * * * * * * * * * * Solve Functions  * * * * * * * * * * * * * //

void ReducedCompressibleSteadyNS::solveOnlineCompressible(scalar mu_now,
        int NmodesUproj, int NmodesPproj, int NmodesEproj)
{
    // Reisuals initialization
    scalar residualNorm(1);
    scalar residualJump(1);
    Eigen::MatrixXd uResidual(1, NmodesUproj);
    Eigen::MatrixXd eResidual(1, NmodesEproj);
    Eigen::MatrixXd pResidual(1, NmodesPproj);
    Eigen::MatrixXd uResidualOld(1, NmodesUproj);
    Eigen::MatrixXd eResidualOld(1, NmodesEproj);
    Eigen::MatrixXd pResidualOld(1, NmodesPproj);
    scalar uNormRes(1);
    scalar eNormRes(1);
    scalar pNormRes(1);

    // Parameters definition
    ITHACAparameters para;
    float residualJumpLim =
        para.ITHACAdict->lookupOrDefault<float>("residualJumpLim", 1e-5);
    float normalizedResidualLim =
        para.ITHACAdict->lookupOrDefault<float>("normalizedResidualLim", 1e-5);
    int maxIter =
        para.ITHACAdict->lookupOrDefault<float>("maxIter", 2000);
    bool closedVolume = false;
    label csolve = 0;

    Vector<double> uresidual_v(0, 0, 0); // Only for temp compilying -> to be removed as soon as reduced systems are used.
    scalar uresidual = 1; // Only for temp compilying -> to be removed as soon as reduced systems are used.
    scalar eresidual = 1; // Only for temp compilying -> to be removed as soon as reduced systems are used.
    scalar presidual = 1; // Only for temp compilying -> to be removed as soon as reduced systems are used.
    scalar residual = 1; // Only for temp compilying -> to be removed as soon as reduced systems are used.

    // Full variables initialization
    volVectorField& U = problem->_U();
    volScalarField& P = problem->_p();
    volScalarField& E = problem->_E();
    volScalarField& rho = problem->_rho();
    volScalarField& psi = problem->_psi();
    surfaceScalarField& phi = problem->_phi();

    // Reduced variables initialization
    Eigen::MatrixXd u(1, NmodesUproj);
    Eigen::MatrixXd e(1, NmodesEproj);
    Eigen::MatrixXd p(1, NmodesPproj);

    fv::options& fvOptions = problem->_fvOptions();
    fluidThermo& thermo = problem->pThermo();

    while ((residualJump > residualJumpLim
            || residual > normalizedResidualLim) && csolve < maxIter)
    {
        csolve++;

        uResidualOld = uResidual;
        eResidualOld = eResidual;
        pResidualOld = pResidual;

        U = ULmodes.reconstruct(u, "Ur");
        E = problem->Emodes.reconstruct(e, "Er");
        P = problem->Pmodes.reconstruct(p, "Pr");


        problem->getUmatrix(U);

//if (simple.momentumPredictor())
//{
        uresidual_v = solve(problem->Ueqn_global() == - problem->getGradP(P)).initialResidual(); //Working
        U = ULmodes.reconstruct(u, "Ur");

        fvOptions.correct(U);
//}

//Energy equation phase
        problem->getEmatrix(U, P);
        eresidual = problem->Eeqn_global().solve().initialResidual();
        E = problem->Emodes.reconstruct(e, "Er");
        fvOptions.correct(thermo.he());
        thermo.correct(); // Here are calculated both temperature and density based on P,U and he.

// Pressure equation phase
        constrainPressure(P, rho, U, problem->getPhiHbyA(problem->Ueqn_global(), U, P), problem->getRhorAUf(problem->Ueqn_global()));// Update the pressure BCs to ensure flux consistency

        closedVolume = adjustPhi(problem->phiHbyA(), U, P);
        while (problem->_simple().correctNonOrthogonal())
        {
            problem->getPmatrix(P);

            presidual = problem->Peqn_global().solve().initialResidual();
            P = problem->Pmodes.reconstruct(p, "Pr");

            if (problem->_simple().finalNonOrthogonalIter())
            {
                phi = problem->phiHbyA() + problem->Peqn_global().flux();
            }
        }

//#include "incompressible/continuityErrs.H"
        P.relax();// Explicitly relax pressure for momentum corrector
        U = problem->HbyA() - (1.0 / problem->Ueqn_global().A()) * problem->getGradP(P);//rAU * fvc::grad(p);
        U.correctBoundaryConditions();
        fvOptions.correct(U);
        bool pLimited = problem->_pressureControl().limit(P);

// For closed-volume cases adjust the pressure and density levels to obey overall mass continuity
        if (closedVolume)
        {
            P += (problem->_initialMass() - fvc::domainIntegrate(psi * P))
                 / fvc::domainIntegrate(psi);
        }

        if (pLimited || closedVolume)
        {
            P.correctBoundaryConditions();
        }

        rho = thermo.rho(); // Here rho is calculated as p*psi = p/(R*T)
        rho.relax();

        uresidual = max(max(uresidual_v[0], uresidual_v[1]), uresidual_v[2]);
        residual = max(max(presidual, uresidual), eresidual);
        Info << "\nResidual: " << residual << endl;
        problem->turbulence->correct();

        residualJump = max(max(((uResidualOld-uResidual).cwiseAbs()).sum(),((pResidualOld-pResidual).cwiseAbs()).sum()),((eResidualOld-eResidual).cwiseAbs()).sum());
    }
}