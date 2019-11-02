/*---------------------------------------------------------------------------*\
Copyright (C) 2017 by the ITHACA-FV authors

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

Description
    Example of NS-Stokes Reduction Problem

\*---------------------------------------------------------------------------*/

#include "argList.H"
#include "Time.H"
#include "fvMesh.H"
#include "RBFMotionSolver.H"
#include "dictionary.H"
#include <iostream>
#include "fvCFD.H"
#include "IOmanip.H"
#include "SteadyNSSimple.H"
#include "ReducedSimpleSteadyNS.H"
#include "ReducedSteadyNS.H"
#include "ITHACAPOD.H"
#include "ITHACAutilities.H"
#include <Eigen/Dense>
#include "DEIM.H"
#include "EigenFunctions.H"
#define _USE_MATH_DEFINES
#include <cmath>
#include "pointMesh.H" //Perhaps not needed..?
#include "pointFields.H" //Perhaps not needed..?
#include "pointPatchField.H"

class DEIM_functionU : public DEIM<fvVectorMatrix>
{
    public:
        using DEIM::DEIM;

        Eigen::MatrixXd onlineCoeffsA(fvVectorMatrix Ueqn)
        {
            Eigen::MatrixXd theta(magicPointsA.size(), 1);

            for (int i = 0; i < magicPointsA.size(); i++)
            {
                Eigen::SparseMatrix<double> Mr;
                Foam2Eigen::fvMatrix2EigenM(Ueqn, Mr);
                int ind_row = magicPointsA[i][0] + xyz_A[i].first() *
                              sizeM;;
                int ind_col = magicPointsA[i][1] + xyz_A[i].second() *
                              sizeM;
                theta(i) = Mr.coeffRef(ind_row, ind_col);
            }

            return theta;
        }

        Eigen::MatrixXd onlineCoeffsB(fvVectorMatrix Ueqn)
        {
            Eigen::MatrixXd theta(magicPointsB.size(), 1);

            for (int i = 0; i < magicPointsB.size(); i++)
            {
                Eigen::VectorXd br;
                Foam2Eigen::fvMatrix2EigenV(Ueqn, br);
                int ind_row = magicPointsB[i] + xyz_B[i] * sizeM;
                theta(i) = br(ind_row);
            }

            return theta;
        }


};

class DEIM_functionP : public DEIM<fvScalarMatrix>
{
    public:
        using DEIM::DEIM;

        Eigen::MatrixXd onlineCoeffsA(fvScalarMatrix Peqn)
        {
            Eigen::MatrixXd theta(magicPointsA.size(), 1);

            for (int i = 0; i < magicPointsA.size(); i++)
            {
                Eigen::SparseMatrix<double> Mr;
                Foam2Eigen::fvMatrix2EigenM(Peqn, Mr);
                int ind_row = magicPointsA[i][0] + xyz_A[i].first() *
                              sizeM;;
                int ind_col = magicPointsA[i][1] + xyz_A[i].second() *
                              sizeM;
                theta(i) = Mr.coeffRef(ind_row, ind_col);
            }

            return theta;
        }

        Eigen::MatrixXd onlineCoeffsB(fvScalarMatrix Peqn)
        {
            Eigen::MatrixXd theta(magicPointsA.size(), 1);

            for (int i = 0; i < magicPointsA.size(); i++)
            {
                Eigen::VectorXd br;
                Foam2Eigen::fvMatrix2EigenV(Peqn, br);
                int ind_row = magicPointsB[i] + xyz_B[i] * sizeM;
                theta(i) = br(ind_row);
            }

            return theta;
        }


};


class NS_geom_par : public SteadyNSSimple
{
    public:
        explicit NS_geom_par(int argc, char* argv[])
            :
            SteadyNSSimple(argc, argv),
            U(_U()),
            p(_p()),
            phi(_phi())
        {
            fvMesh& mesh = _mesh();
            dyndict = new IOdictionary
            (
                IOobject
                (
                    "dynamicMeshDictRBF",
                    "./constant",
                    mesh,
                    IOobject::MUST_READ,
                    IOobject::NO_WRITE
                )
            );
#include "createFields_aux.H"
            //#include "prepareRestart.H"
            ITHACAutilities::getPointsFromPatch(mesh, 0, wingTop, wingTop_ind);
            ITHACAutilities::getPointsFromPatch(mesh, 1, wingBottom, wingBottom_ind);
            ms = new RBFMotionSolver(mesh, *dyndict);
            vectorField motion(ms->movingPoints().size(), vector::zero);
            movingIDs = ms->movingIDs();
            x0 = ms->movingPoints();
            curX = x0;
            point0 = ms->curPoints();
            NmodesU = readInt(ITHACAdict->lookup("N_modes_U"));
            NmodesP = readInt(ITHACAdict->lookup("N_modes_P"));
            NmodesSUP = readInt(ITHACAdict->lookup("N_modes_SUP"));
            NmodesDEIMAU = readInt(ITHACAdict->lookup("N_modes_DEIM_A_U"));
            NmodesDEIMBU = readInt(ITHACAdict->lookup("N_modes_DEIM_B_U"));
            NmodesDEIMAP = readInt(ITHACAdict->lookup("N_modes_DEIM_A_P"));
            NmodesDEIMBP = readInt(ITHACAdict->lookup("N_modes_DEIM_B_P"));
            axis = Foam::vector(0, 0, 1);
            ITHACAutilities::createSymLink("./ITHACAoutput/supfield");
        }

        int NmodesU;
        int NmodesP;
        int NmodesSUP;
        int NmodesDEIMAU;
        int NmodesDEIMBU;
        int NmodesDEIMAP;
        int NmodesDEIMBP;

        /// Lifted velocity modes.
        Modes<vector> ULmodes;
        Modes<scalar> PLmodes;



        volScalarField& p;
        volVectorField& U;
        surfaceScalarField& phi;
        vector axis;

        autoPtr<volScalarField> _nut0;
        autoPtr<volScalarField> _nuTilda0;

        autoPtr<volScalarField> _nut;
        autoPtr<volScalarField> _nuTilda;

        DEIM_functionU* DEIMU;
        DEIM_functionP* DEIMP;

        PtrList<volVectorField> fieldsUA_U;
        PtrList<volVectorField> fieldsUB_U;
        PtrList<volVectorField> fieldsUA_P;
        PtrList<volVectorField> fieldsUB_P;

        PtrList<volScalarField> fieldsPA_U;
        PtrList<volScalarField> fieldsPB_U;
        PtrList<volScalarField> fieldsPA_P;
        PtrList<volScalarField> fieldsPB_P;

        PtrList<surfaceScalarField> fieldsphiA_U;
        PtrList<surfaceScalarField> fieldsphiB_U;
        PtrList<surfaceScalarField> fieldsphiA_P;
        PtrList<surfaceScalarField> fieldsphiB_P;

        std::chrono::duration<double> elapsed;
        std::chrono::duration<double> elapsedON;
        std::chrono::duration<double> elapsedOFF;

        autoPtr<RBFMotionSolver> RBFmotionPtr;
        autoPtr<volScalarField> _cv;
        autoPtr<volScalarField> _NonOrtho;

        /// dictionary to store input output infos
        IOdictionary* dyndict;

        RBFMotionSolver* ms;

        List<vector> wingTop;
        List<vector> wingBottom;
        vectorField point0;

        vectorField pointTop;
        vectorField pointBottom;

        labelList movingIDs;
        List<vector> x0;
        List<vector> curX;

        labelList wingTop_ind;
        labelList wingBottom_ind;

        // Reduced Matrices DEIM
        std::vector<Eigen::MatrixXd> ReducedMatricesAU;
        Eigen::MatrixXd ReducedVectorsBU;
        std::vector<Eigen::MatrixXd> ReducedMatricesAP;
        Eigen::MatrixXd ReducedVectorsBP;
        Eigen::MatrixXd ModesUEig;
        Eigen::MatrixXd ModesPEig;

        PtrList<volScalarField> Volumes;
        PtrList<volScalarField> Tfield_new;
        PtrList<volScalarField> Volumes_new;




        // DEIM_function* DEIMmatrice;

        void OfflineSolve(Eigen::VectorXd pars, word Folder)
        {
            fvMesh& mesh = _mesh();
            Time& runTime = _runTime();
            volScalarField& cv = _cv();
            surfaceScalarField& phi = _phi();

            if (offline)
            {
                ITHACAstream::read_fields(Ufield, U, "./ITHACAoutput/Offline/");
                ITHACAstream::read_fields(Pfield, p, "./ITHACAoutput/Offline/");
                ITHACAstream::read_fields(Volumes, cv, "./ITHACAoutput/Offline/");
                //ITHACAstream::read_fields(phiField, phi, "./ITHACAoutput/Offline/");
                //volVectorField Usup("Usup", U);
                //ITHACAstream::read_fields(supfield, Usup, "./ITHACAoutput/supfield/");
            }
            else
            {
                for (int k = 0; k < pars.rows(); k++)
                {
                    List<scalar> par(1);
                    updateMesh(pars[k]);
                    ITHACAstream::writePoints(mesh.points(), Folder, name(k + 1) + "/polyMesh/");
                    truthSolve2(par, Folder);
                    cv.ref() = mesh.V();
                    Volumes.append(cv);
                    ITHACAstream::exportSolution(U, name(k + 1), Folder);
                    ITHACAstream::exportSolution(cv, name(k + 1), Folder);
                    ITHACAstream::exportSolution(p, name(k + 1), Folder);
                    //solveOneSup(p, k);
                    restart();
                }
            }
        };

        void solveOneSup(volScalarField p, int k)
        {
            volVectorField Usup
            (
                IOobject
                (
                    "Usup",
                    U.time().timeName(),
                    U.mesh(),
                    IOobject::NO_READ,
                    IOobject::AUTO_WRITE
                ),
                U.mesh(),
                dimensionedVector("zero", U.dimensions(), vector::zero)
            );
            dimensionedScalar nu_fake
            (
                "nu_fake",
                dimensionSet(0, 2, -1, 0, 0, 0, 0),
                scalar(1)
            );
            Vector<double> v(0, 0, 0);

            for (label i = 0; i < Usup.boundaryField().size(); i++)
            {
                if (Usup.boundaryField()[i].type() != "processor")
                {
                    ITHACAutilities::changeBCtype(Usup, "fixedValue", i);
                    assignBC(Usup, i, v);
                    assignIF(Usup, v);
                }
            }

            fvVectorMatrix u_sup_eqn
            (
                - fvm::laplacian(nu_fake, Usup)
            );
            solve
            (
                u_sup_eqn == fvc::grad(p)
            );
            supfield.append(Usup);
            ITHACAstream::exportSolution(Usup, name(k + 1), "./ITHACAoutput/supfield/");
        }

        void restart()
        {
            fvMesh& mesh = _mesh();
            mesh.movePoints(point0);
            volScalarField& p = _p();
            volVectorField& U = _U();
            surfaceScalarField& phi = _phi();
            p = _p0();
            U = _U0();
            phi = _phi0();
        }

        void updateMesh(double par=0)
        {
            fvMesh& mesh = _mesh();
            mesh.movePoints(point0);
            List<vector> wingTop_cur = ITHACAutilities::rotatePoints(wingTop, axis, par);
            List<vector> wingBottom_cur = ITHACAutilities::rotatePoints(wingBottom, axis, par);
            ITHACAutilities::setIndices2Value(wingTop_ind, wingTop_cur, movingIDs, curX);
            ITHACAutilities::setIndices2Value(wingBottom_ind, wingBottom_cur, movingIDs, curX);
            ms->setMotion(curX - x0);
            mesh.movePoints(ms->curPoints());
        }

        void PODDEIM()
        {
            PODDEIM(NmodesU, NmodesP, NmodesSUP, NmodesDEIMAU, NmodesDEIMBU, NmodesDEIMAP,
                    NmodesDEIMBP);
        }

        void PODDEIM(int NmodesU, int NmodesP, int NmodesSUP, int NmodesDEIMAU,
                     int NmodesDEIMBU, int NmodesDEIMAP, int NmodesDEIMBP)
        {
            volVectorField& U = _U();
            volScalarField& p = _p();
            surfaceScalarField& phi = _phi();
            // DEIMU = new DEIM_functionU(UEqnList, NmodesDEIMAU, NmodesDEIMBU, "U_matrix");
            // DEIMP = new DEIM_functionP(PEqnList, NmodesDEIMAP, NmodesDEIMBP, "P_matrix");
            ULmodes.resize(0);
            PLmodes.resize(0);

            for (int i = 0; i < inletIndex.rows(); i++)
            {
                ULmodes.append(liftfield[i]);
            }

            for (int i = 0; i < NmodesU; i++)
            {
                ULmodes.append(Umodes.toPtrList()[i]);
            }

            for (int i = 0; i < NmodesSUP; i++)
            {
                ULmodes.append(supmodes.toPtrList()[i]);
            }

            for (int i = 0; i < NmodesP; i++)
            {
                PLmodes.append(Pmodes.toPtrList()[i]);
            }

            // ReducedMatricesAU.resize(NmodesDEIMAU);
            // ReducedMatricesAP.resize(NmodesDEIMAP);
            // ULmodes.toEigen();
            // PLmodes.toEigen();
            // for (int i = 0; i < NmodesDEIMAU; i++)
            // {
            //     ReducedMatricesAU[i] = ULmodes.EigenModes[0].transpose() *
            //                            DEIMU->MatrixOnlineA[i] *
            //                            ULmodes.EigenModes[0];
            // }
            // ReducedVectorsBU = ULmodes.EigenModes[0].transpose() * DEIMU->MatrixOnlineB;
            // for (int i = 0; i < NmodesDEIMAP; i++)
            // {
            //     ReducedMatricesAP[i] = PLmodes.EigenModes[0].transpose() *
            //                            DEIMP->MatrixOnlineA[i] *
            //                            PLmodes.EigenModes[0];
            // }
            // ReducedVectorsBP = PLmodes.EigenModes[0].transpose() * DEIMP->MatrixOnlineB;
        }

};

class reducedSimpleSteadyNSGeo : public reducedSimpleSteadyNS
{
    public:
        reducedSimpleSteadyNSGeo(NS_geom_par& problem)
            :
            reducedSimpleSteadyNS(problem),
            problem(&problem)
        {}

        /// Full problem.
        NS_geom_par* problem;

        void solveOnline_Simple(scalar mu_now,
                                int NmodesUproj, int NmodesPproj, int NmodesSup, word Folder)
        {
            counter++;
            int UprojN;
            int PprojN;

            if (NmodesUproj == 0)
            {
                UprojN = ULmodes.size();
            }
            else
            {
                UprojN = NmodesUproj + NmodesSup + problem->inletIndex.rows();
            }

            if (NmodesPproj == 0)
            {
                PprojN = problem->Pmodes.size();
            }
            else
            {
                PprojN = NmodesPproj;
            }

            Eigen::VectorXd uresidualOld = Eigen::VectorXd::Zero(UprojN);
            Eigen::VectorXd presidualOld = Eigen::VectorXd::Zero(PprojN);
            Eigen::VectorXd uresidual;
            Eigen::VectorXd presidual;
            scalar U_norm_res(1);
            scalar P_norm_res(1);
            Eigen::MatrixXd a = Eigen::VectorXd::Zero(UprojN);
            Eigen::MatrixXd b = Eigen::VectorXd::Zero(PprojN);
            a(0) = vel_now(0, 0);
            ITHACAparameters para;
            float residualJumpLim =
                para.ITHACAdict->lookupOrDefault<float>("residualJumpLim", 1e-5);
            float normalizedResidualLim =
                para.ITHACAdict->lookupOrDefault<float>("normalizedResidualLim", 1e-5);
            scalar residual_jump(1 + residualJumpLim);
            //problem->restart();
            volVectorField Uaux("U", problem->_U());
            volScalarField Paux("p", problem->_p());
            volScalarField& p = problem->_p();
            volVectorField& U = problem->_U();
            surfaceScalarField phi(problem->_phi());
            phi = fvc::interpolate(U) & U.mesh().Sf();
            int iter = 0;
            simpleControl& simple = problem->_simple();
            setRefCell(p, simple.dict(), problem->pRefCell, problem->pRefValue);

            while (residual_jump > residualJumpLim
                    || std::max(U_norm_res, P_norm_res) > normalizedResidualLim)
            {
                iter++;
                Paux.storePrevIter();
                fvVectorMatrix UEqn
                (
                    fvm::div(phi, Uaux)
                    + problem->turbulence->divDevReff(Uaux)
                    == -fvc::grad(Paux)
                );
                UEqn.relax();
                // Eigen::MatrixXd thetaonA = problem->DEIMU->onlineCoeffsA(UEqn);
                // Eigen::MatrixXd thetaonB = problem->DEIMU->onlineCoeffsB(UEqn);
                // Eigen::SparseMatrix<double> Mr = problem->DEIMU->MatrixOnlineA[0]*thetaonA(0);
                // Eigen::VectorXd Br = problem->DEIMU->MatrixOnlineB*thetaonB;
                // Eigen::VectorXd Bf;
                // Eigen::SparseMatrix<double> Af;
                // Foam2Eigen::fvMatrix2EigenM(UEqn, Af);
                // Foam2Eigen::fvMatrix2EigenV(UEqn, Bf);
                // for (int i = 1; i < problem->DEIMU->MatrixOnlineA.size(); i++)
                // {
                //     Mr += problem->DEIMU->MatrixOnlineA[i]*thetaonA(i);
                // }
                // std::cout << (Af-Mr).sum() << std::endl;
                // std::cout << (Af).sum() << std::endl;
                // std::cout << (Bf-Br).sum() << std::endl;
                // std::cout << (Bf).sum() << std::endl;
                // exit(0);
                //Eigen::MatrixXd A = EigenFunctions::MVproduct(problem->ReducedMatricesAU,thetaonA);
                //List<Eigen::MatrixXd> RedLinSysU;
                //RedLinSysU.resize(2);
                //RedLinSysU[0] = A;
                //RedLinSysU[1] = problem->ReducedVectorsBU * thetaonB;
                List<Eigen::MatrixXd> RedLinSysU = problem->ULmodes.project(UEqn, UprojN);
                a = reducedProblem::solveLinearSys(RedLinSysU, a, uresidual, vel_now);
                problem->ULmodes.reconstruct(Uaux, a, "U");
                //phi = fvc::flux(Uaux);
                phi = fvc::flux(UEqn.H() / UEqn.A());
                List<Eigen::MatrixXd> RedLinSysP;
                int i = 0;
                while (simple.correctNonOrthogonal())
        		{
	                fvScalarMatrix pEqn
	                (
	                    fvm::laplacian(1 / UEqn.A(), Paux) == fvc::div(phi)
	                );
	                //Eigen::MatrixXd thetaonAP = problem->DEIMP->onlineCoeffsA(pEqn);
	                //Eigen::MatrixXd thetaonBP = problem->DEIMP->onlineCoeffsB(pEqn);
	                //Eigen::MatrixXd AP = EigenFunctions::MVproduct(problem->ReducedMatricesAP,
	                //                    thetaonAP);
	                //List<Eigen::MatrixXd> RedLinSysP;
	                //RedLinSysP.resize(2);
	                //RedLinSysP[0] = AP;
	                //RedLinSysP[1] = problem->ReducedVectorsBP * thetaonBP;
                    //List<Eigen::MatrixXd> RedLinSysP = problem->Pmodes.project(pEqn, PprojN);
	                RedLinSysP = problem->Pmodes.project(pEqn, PprojN);
                    if (i == 0)
                    {
	                   b = reducedProblem::solveLinearSys(RedLinSysP, b, presidual);
                    }
                    else
                    {
                       Eigen::VectorXd dummy;
                       b = reducedProblem::solveLinearSys(RedLinSysP, b, dummy); 
                    }
	                if (simple.finalNonOrthogonalIter())
                    {
                        phi -= pEqn.flux();
                    }
                    problem->Pmodes.reconstruct(Paux, b, "p");
                    i++;
         
                }
                Paux.relax();
                uresidualOld = uresidualOld - uresidual;
                presidualOld = presidualOld - presidual;
                uresidualOld = uresidualOld.cwiseAbs();
                presidualOld = presidualOld.cwiseAbs();
                residual_jump = std::max(uresidualOld.sum(), presidualOld.sum());
                uresidualOld = uresidual;
                presidualOld = presidual;
                uresidual = uresidual.cwiseAbs();
                presidual = presidual.cwiseAbs();
                U_norm_res = uresidual.sum() / (RedLinSysU[1].cwiseAbs()).sum();
                P_norm_res = presidual.sum() / (RedLinSysP[1].cwiseAbs()).sum();

                if (para.debug)
                {
                    std::cout << "Residual jump = " << residual_jump << std::endl;
                    std::cout << "Normalized residual = " << std::max(U_norm_res,
                              P_norm_res) << std::endl;
                }
            }

            std::cout << "Solution " << counter << " converged in " << iter <<
                      " iterations." << std::endl;
            std::cout << "Final normalized residual for velocity: " << U_norm_res <<
                      std::endl;
            std::cout << "Final normalized residual for pressure: " << P_norm_res <<
                      std::endl;
            ULmodes.reconstruct(Uaux, a, "Uaux");
            problem->Pmodes.reconstruct(Paux, b, "Paux");
            Uaux.rename("Uaux");
            Paux.rename("Paux");
            ITHACAstream::exportSolution(Uaux, name(counter), Folder);
            ITHACAstream::exportSolution(Paux, name(counter), Folder);
        }
};


int main(int argc, char* argv[])
{
    NS_geom_par example(argc, argv);
    Eigen::MatrixXd parAlpha;
    std::ifstream exFile("./angles_mat.txt");

    if (exFile)
    {
        parAlpha = ITHACAstream::readMatrix("./angles_mat.txt");
    }
    else
    {
        parAlpha = ITHACAutilities::rand(100, 1, -10, 10);
        ITHACAstream::exportMatrix(parAlpha, "angles", "eigen", "./");
    }

    example.OfflineSolve(parAlpha.leftCols(1), "./ITHACAoutput/Offline/");
    ITHACAstream::read_fields(example.liftfield, example.U, "./lift/");
    example.inletIndex.resize(1, 2);
    example.inletIndex(0, 0) = 3;
    example.inletIndex(0, 1) = 0;
    ITHACAutilities::normalizeFields(example.liftfield);
    example.updateMesh();
    // Homogenize the snapshots
    example.computeLift(example.Ufield, example.liftfield, example.Uomfield);
    //example.updateMesh(parAlpha(0) * 0);
    // Perform POD on velocity and pressure and store the first 10 modes
    ITHACAPOD::getModes(example.Uomfield, example.Umodes, example.Volumes,
                        example.podex, 0, 0,
                        example.NmodesU);
    std::cout << example.NmodesP << std::endl;
    ITHACAPOD::getModes(example.Pfield, example.Pmodes, example.Volumes,
                        example.podex, 0, 0, example.NmodesP);
    //ITHACAPOD::getModes(example.supfield, example.supmodes, example.supex,
    //                    0, 1, example.NmodesSUP);
    //    ITHACAPOD::getModes(example.phiField, example.phiModes, example.Uomfield, 0,
    //                        0, 1, 8);
    example.PODDEIM();
    Eigen::MatrixXd vel(1, 1);
    vel(0, 0) = 1;
    Eigen::MatrixXd onlineAlpha;

    std::ifstream exFileOn("./angles_on_mat.txt");

    if (exFileOn)
    {
        onlineAlpha = ITHACAstream::readMatrix("./angles_on_mat.txt");
    }
    else
    {
        onlineAlpha = ITHACAutilities::rand(50, 1, -9.5, 9.5);
        ITHACAstream::exportMatrix(onlineAlpha, "angles_on", "eigen", "./");
    }

    //Eigen::MatrixXd onlineAlpha = ITHACAutilities::rand(50, 1, -9.5, 9.5);
    example.restart();
    reducedSimpleSteadyNSGeo reduced(example);
    reduced.setOnlineVelocity(vel);
    NS_geom_par checkOff(argc, argv);
    checkOff.offline = false;
    List<scalar> par(1);
    PtrList<volVectorField> Ufull;
    PtrList<volScalarField> Pfull;
    PtrList<volVectorField> Ured;
    PtrList<volScalarField> Pred;
    volVectorField Uaux("Uaux", example._U());
    volScalarField Paux("Paux", example._p());

    for (int i = 0; i < onlineAlpha.rows(); i++)
    {
        //example.updateMesh(onlineAlpha(i, 0));
        checkOff.restart();
        checkOff.updateMesh(onlineAlpha(i, 0));

        //Offline part;
        checkOff.truthSolve2(par, "./ITHACAoutput/checkOff/");
        Ufull.append(checkOff._U());
        Pfull.append(checkOff._p());
        ITHACAstream::exportSolution(checkOff._U(), name(i + 1),
                                    "./ITHACAoutput/checkOff/");
        ITHACAstream::exportSolution(checkOff._p(), name(i + 1),
                                    "./ITHACAoutput/checkOff/");
        ITHACAstream::writePoints(checkOff._mesh().points(), "./ITHACAoutput/checkOff/",
                                 name(i + 1) + "/polyMesh/");
        example.restart();
        example.updateMesh(onlineAlpha(i, 0));
        //Info << reduced->_mesh().points() - checkOff._mesh().points() << endl;
        reduced.solveOnline_Simple(1, example.NmodesU, example.NmodesP,
                                 example.NmodesSUP, "./ITHACAoutput/checkOff/");
    }

    ITHACAstream::read_fields(Ured, Uaux, "./ITHACAoutput/checkOff/");
    ITHACAstream::read_fields(Pred, Paux, "./ITHACAoutput/checkOff/");
    Eigen::MatrixXd errorU; // = ITHACAutilities::error_listfields(Ufull, Ured);
    Eigen::MatrixXd errorP = ITHACAutilities::error_listfields(Pfull, Pred);
    //vector U_fs(1,0,0);
    dimensionedVector U_fs("U_fs", dimVelocity, vector(1, 0, 0));
    errorU.resize(Ufull.size(), 1);

    for (label k = 0; k < Ufull.size(); k++)
    {
        errorU(k, 0) = ITHACAutilities::L2norm(Ufull[k] - Ured[k]) /
                       ITHACAutilities::L2norm(Ufull[k] - U_fs);
    }

    ITHACAstream::exportMatrix(errorU,
                               "errorU_" + name(example.NmodesU) + "_" + name(example.NmodesP), "python", ".");
    ITHACAstream::exportMatrix(errorP,
                               "errorP_" + name(example.NmodesU) + "_" + name(example.NmodesP), "python", ".");
    exit(0);
    //reduced.solveOnline_Simple(1, 10, 10);
    //     ITHACAstream::writePoints(example._mesh().points(), "./ITHACAoutput/Reconstruct", name(i + 1) + "/polyMesh/");
    // }
    //Error check
    // Eigen::MatrixXd ErrorU;
    // Eigen::MatrixXd ErrorP;
    // ErrorU.resize(50, 1);
    // ErrorP.resize(50, 1);
    // PtrList<volScalarField> onlineP;
    // PtrList<volVectorField> onlineU;
    // example.updateMesh(onlineAlpha(0));
    // ITHACAstream::writePoints(example._mesh().points(), "./ITHACAoutput/checkOn",
    //                           name(1) + "/polyMesh/");
    // reduced.solveOnline_Simple(1, example.NmodesU, example.NmodesP,
    //                            example.NmodesSUP, "./ITHACAoutput/checkOn/");
    // exit(0);
    // example.projectSUP("./Matrices", 5, 5, 5);
    // example.C_tensor = example.convective_term_tens_phi(5, 5, 5);
    // example.P_matrix = example.divergence_term_phi(5, 5, 5);
    // reducedSteadyNS ridotto(example);
    // ridotto.solveOnline_sup(vel);
    // Eigen::MatrixXd tmp_sol(ridotto.y.rows() + 1, 1);
    // std::cout << ridotto.y << std::endl;
    // tmp_sol(0) = 1;
    // tmp_sol.col(0).tail(ridotto.y.rows()) = ridotto.y;
    // ridotto.online_solution.append(tmp_sol);
    // ridotto.reconstruct_sup("./ITHACAoutput/Reconstruction/");
    // ITHACAstream::writePoints(example._mesh().points(),
    //                           "./ITHACAoutput/Reconstruction", name(1) + "/polyMesh/");
    // exit(0);
    // for (int i = 0; i < 50; i++)
    // {
    //     ITHACAstream::writePoints(example._mesh().points(), "./ITHACAoutput/checkOn",
    //                               name(1) + "/polyMesh/");
    //     reduced.solveOnline_Simple(1, i + 1, i + 1, i + 1, "./ITHACAoutput/checkOn/");
    // }
    // ITHACAstream::read_fields(onlineU, "Uaux", "./ITHACAoutput/checkOn");
    // ITHACAstream::read_fields(onlineP, "Paux", "./ITHACAoutput/checkOn");
    // for (int i = 0; i < 50; i++)
    // {
    //     ErrorP(i, 0) = ITHACAutilities::error_fields(checkOff.Pfield[0], onlineP[i]);
    //     ErrorU(i, 0) = ITHACAutilities::error_fields(checkOff.Ufield[0], onlineU[i]);
    // }
    // cnpy::save(ErrorP, "ErrorP.npy");
    // cnpy::save(ErrorU, "ErrorU.npy");
    // // /// Compute the offline part of the DEIM procedure
    // //example.PODDEIM();
    // // /// Construct a new set of parameters
    // // Eigen::MatrixXd par_new1 = ITHACAutilities::rand(100, 2, -0.28, 0.28);
    // // /// Solve the online problem with the new parameters
    // // example.OnlineSolve(par_new1, "comparison");
    // // ///
    // // example.OfflineSolveNew(par_new1, "./ITHACAoutput/comparison/");
    // // Eigen::MatrixXd error = ITHACAutilities::error_listfields(example.Tfield_new,
    // //                         example.Tonline);
    // // ITHACAstream::exportMatrix(error,
    // //                            "error_" + name(example.NTmodes) + "_" + name(example.NmodesDEIMA) + "_" + name(
    // //                                example.NmodesDEIMA), "python", ".");
    // // Info << "End\n" << endl;
    return 0;
}
