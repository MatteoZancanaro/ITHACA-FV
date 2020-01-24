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

// void reducedSimpleSteadyNS::solveOnline_Simple(scalar mu_now)
// {
//     counter++;
//     Eigen::VectorXd uresidualOld;
//     Eigen::VectorXd presidualOld;
//     uresidualOld.resize(ULmodes.size());
//     presidualOld.resize(problem->Pmodes.size());
//     Eigen::VectorXd uresidual;
//     Eigen::VectorXd presidual;
//     scalar residual_jump(1);
//     scalar U_norm_res(1);
//     scalar P_norm_res(1);
//     Eigen::MatrixXd a = Eigen::VectorXd::Zero(ULmodes.size());
//     Eigen::MatrixXd b = Eigen::VectorXd::Zero(problem->Pmodes.size());
//     ITHACAparameters para;
//     float residualJumpLim =
//         para.ITHACAdict->lookupOrDefault<float>("residualJumpLim", 1e-5);
//     float normalizedResidualLim =
//         para.ITHACAdict->lookupOrDefault<float>("normalizedResidualLim", 1e-5);
//     volVectorField Uaux("Uaux", problem->_U());
//     volScalarField Paux("Paux", problem->_p());

//     while (residual_jump > residualJumpLim
//             || std::max(U_norm_res, P_norm_res) > normalizedResidualLim)
//     {
//         Uaux = ULmodes.reconstruct(a, "Uaux");
//         Paux = problem->Pmodes.reconstruct(b, "Paux");
//         simpleControl& simple = problem->_simple();
//         setRefCell(Paux, simple.dict(), problem->pRefCell, problem->pRefValue);
//         problem->_phi() = linearInterpolate(Uaux) & problem->_U().mesh().Sf();
//         fvVectorMatrix Au(get_Umatrix_Online(Uaux, Paux));
//         List<Eigen::MatrixXd> RedLinSysU = ULmodes.project(Au);
//         a = reducedProblem::solveLinearSys(RedLinSysU, a, uresidual, vel_now);
//         Info << "res for a" << endl;
//         Info << uresidual.norm() << endl;
//         Uaux = ULmodes.reconstruct(a, "Uaux");
//         problem->_phi() = linearInterpolate(Uaux) & problem->_U().mesh().Sf();
//         fvScalarMatrix Ap(get_Pmatrix_Online(Uaux, Paux));
//         List<Eigen::MatrixXd> RedLinSysP = problem->Pmodes.project(Ap);
//         b = reducedProblem::solveLinearSys(RedLinSysP, b, presidual);
//         Info << "res for b" << endl;
//         Info << presidual.norm() << endl;
//         uresidualOld = uresidualOld - uresidual;
//         presidualOld = presidualOld - presidual;
//         uresidualOld = uresidualOld.cwiseAbs();
//         presidualOld = presidualOld.cwiseAbs();
//         residual_jump = std::max(uresidualOld.sum(), presidualOld.sum());
//         uresidualOld = uresidual;
//         presidualOld = presidual;
//         uresidual = uresidual.cwiseAbs();
//         presidual = presidual.cwiseAbs();
//         U_norm_res = uresidual.sum() / (RedLinSysU[1].cwiseAbs()).sum();
//         P_norm_res = presidual.sum() / (RedLinSysP[1].cwiseAbs()).sum();
//         // std::cout << "Residual jump = " << residual_jump << std::endl;
//         // std::cout << "Normalized residual = " << std::max(U_norm_res,P_norm_res) << std::endl;
//     }

//     Uaux = ULmodes.reconstruct(a, "Uaux");
//     Paux = problem->Pmodes.reconstruct(b, "Paux");
//     ITHACAstream::exportSolution(Uaux, name(counter),
//                                  "./ITHACAoutput/Reconstruct/");
//     ITHACAstream::exportSolution(Paux, name(counter),
//                                  "./ITHACAoutput/Reconstruct/");
// }

// void reducedSimpleSteadyNS::setOnlineVelocity(Eigen::MatrixXd vel)
// {
//     assert(problem->inletIndex.rows() == vel.size()
//            && "Imposed boundary conditions dimensions do not match given values matrix dimensions");
//     vel_now = vel;
// }