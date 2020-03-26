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


void ReducedCompressibleSteadyNS::projectReducedOperators(int NmodesUproj, int NmodesPproj, int NmodesEproj)
{
    PtrList<volVectorField> gradModP;
    for (label i = 0; i < NmodesPproj; i++)
    {
        gradModP.append(fvc::grad(problem->Pmodes[i]));
    }
    projGradModP = ULmodes.project(gradModP, NmodesUproj);
}

// * * * * * * * * * * * * * * * Solve Functions  * * * * * * * * * * * * * //

void ReducedCompressibleSteadyNS::solveOnlineCompressible(scalar mu_now,
        int NmodesUproj, int NmodesPproj, int NmodesEproj)
{
    // Residuals initialization
    scalar residualNorm(1);
    scalar residualJump(1);
    Eigen::MatrixXd uResidualOld = Eigen::MatrixXd::Zero(1, NmodesUproj);
    Eigen::MatrixXd eResidualOld = Eigen::MatrixXd::Zero(1, NmodesEproj);
    Eigen::MatrixXd pResidualOld = Eigen::MatrixXd::Zero(1, NmodesPproj);
    Eigen::VectorXd uResidual(Eigen::Map<Eigen::VectorXd>(uResidualOld.data(),
                              NmodesUproj));
    Eigen::VectorXd eResidual(Eigen::Map<Eigen::VectorXd>(eResidualOld.data(),
                              NmodesEproj));
    Eigen::VectorXd pResidual(Eigen::Map<Eigen::VectorXd>(pResidualOld.data(),
                              NmodesPproj));
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
    // Full variables initialization
    fluidThermo& thermo = problem->pThermo();
    volVectorField& U = problem->_U();
    volScalarField& P = problem->pThermo->p();
    //volScalarField& P = thermo.p();
    volScalarField& E = problem->pThermo->he();
    //volScalarField& E = problem->_E();
    volScalarField& rho = problem->_rho();
    volScalarField& psi = problem->_psi();
    surfaceScalarField& phi = problem->_phi();
    Time& runTime = problem->_runTime();
    fvMesh& mesh = problem->_mesh();
    fv::options& fvOptions = problem->_fvOptions();
    scalar cumulativeContErr = problem->cumulativeContErr;
    // Reduced variables initialization
    Eigen::MatrixXd u = Eigen::MatrixXd::Zero(NmodesUproj, 1);
    Eigen::MatrixXd e = Eigen::MatrixXd::Zero(NmodesEproj, 1);
    Eigen::MatrixXd p = Eigen::MatrixXd::Zero(NmodesPproj, 1);

    while ((residualJump > residualJumpLim
            || residualNorm > normalizedResidualLim) && csolve < maxIter)
    {
        csolve++;
        Info << "csolve:" << csolve << endl;

        #if OFVER == 6
        problem->_simple().loop(runTime);
        #else
        problem->_simple().loop();
        #endif

        uResidualOld = uResidual;
        eResidualOld = eResidual;
        pResidualOld = pResidual;
        //Momentum equation phase
        List<Eigen::MatrixXd> RedLinSysU;

        problem->getUmatrix(U);

        //if (problem->_simple().momentumPredictor())
        //{
            RedLinSysU = ULmodes.project(problem->Ueqn_global(),
                                         NmodesUproj);
            Eigen::MatrixXd projGradP = projGradModP * p;
            RedLinSysU[1] = RedLinSysU[1] - projGradP;
            u = reducedProblem::solveLinearSys(RedLinSysU, u, uResidual, vel_now, "bdcSvd");
            U = ULmodes.reconstruct(u, "Ur");
            //solve(problem->Ueqn_global() == -problem->getGradP(P)); //For debug purposes only, second part only useful when using uEqn_global==-getGradP
            fvOptions.correct(U);
        //}
        //Energy equation phase
        problem->getEmatrix(U, P);
        List<Eigen::MatrixXd> RedLinSysE = problem->Emodes.project(
                                               problem->Eeqn_global(), NmodesEproj);
        e = reducedProblem::solveLinearSys(RedLinSysE, e, eResidual);
        //E = problem->Emodes.reconstruct(e, "Er");
        problem->Eeqn_global().solve(); //For debug purposes only
        //fvOptions.correct(thermo.he());
        fvOptions.correct(E);
        thermo.correct(); // Here are calculated both temperature and density based on P,U and he.
        // Pressure equation phase
        constrainPressure(P, rho, U, problem->getPhiHbyA(problem->Ueqn_global(), U, P),
                          problem->getRhorAUf(
                              problem->Ueqn_global()));// Update the pressure BCs to ensure flux consistency
        //closedVolume = adjustPhi(problem->phiHbyA(), U, P);
        //closedVolume = adjustPhi(problem->getPhiHbyA(problem->Ueqn_global(), U, P), U, P);
        surfaceScalarField phiHbyACalculated = problem->getPhiHbyA(problem->Ueqn_global(), U, P);
        closedVolume = adjustPhi(phiHbyACalculated, U, P);

        List<Eigen::MatrixXd> RedLinSysP;

        while (problem->_simple().correctNonOrthogonal())
        {
            problem->getPmatrix(problem->Ueqn_global(), U, P);
            RedLinSysP = problem->Pmodes.project(problem->Peqn_global(), NmodesPproj);
            p = reducedProblem::solveLinearSys(RedLinSysP, p, pResidual);
            //P.storePrevIter();
            //P = problem->Pmodes.reconstruct(p, "Pr");
            problem->Peqn_global().solve(); //For debug purposes only

            if (problem->_simple().finalNonOrthogonalIter())
            {
                //phi = problem->phiHbyA() + problem->Peqn_global().flux();
                phi = problem->getPhiHbyA(problem->Ueqn_global(), U, P) + problem->Peqn_global().flux();
            }
        }

#include "incompressible/continuityErrs.H"
        P.relax();// Explicitly relax pressure for momentum corrector
        U = problem->HbyA() - (1.0 / problem->Ueqn_global().A()) * problem->getGradP(P);
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

        //rho.storePrevIter();
        rho = thermo.rho(); // Here rho is calculated as p*psi = p/(R*T)
        rho.relax();
        std::cout << "Ures = " << (uResidual.cwiseAbs()).sum() / (RedLinSysU[1].cwiseAbs()).sum() << std::endl;
        std::cout << "Eres = " << (eResidual.cwiseAbs()).sum() / (RedLinSysE[1].cwiseAbs()).sum() << std::endl;
        std::cout << "Pres = " << (pResidual.cwiseAbs()).sum() / (RedLinSysP[1].cwiseAbs()).sum() << std::endl;
        std::cout << "U = " << u << std::endl;
        std::cout << "E = " << e << std::endl;
        std::cout << "P = " << p << std::endl;
        residualNorm = max(max((uResidual.cwiseAbs()).sum() /
                               (RedLinSysU[1].cwiseAbs()).sum(),
                               (pResidual.cwiseAbs()).sum() / (RedLinSysP[1].cwiseAbs()).sum()),
                           (eResidual.cwiseAbs()).sum() / (RedLinSysE[1].cwiseAbs()).sum());
        residualJump = max(max(((uResidual - uResidualOld).cwiseAbs()).sum() / (RedLinSysU[1].cwiseAbs()).sum(),
                               ((pResidual - pResidualOld).cwiseAbs()).sum() / (RedLinSysP[1].cwiseAbs()).sum()),
                           ((eResidual - eResidualOld).cwiseAbs()).sum() / (RedLinSysE[1].cwiseAbs()).sum());
        std::cout << residualNorm << std::endl;
        std::cout << residualJump << std::endl;
        problem->turbulence->correct();
    }
    label k = 1;
    ITHACAstream::exportSolution(U, "1", "./ITHACAoutput/Online/");
    ITHACAstream::exportSolution(P, "1", "./ITHACAoutput/Online/");
    ITHACAstream::exportSolution(E, "1", "./ITHACAoutput/Online/");
}