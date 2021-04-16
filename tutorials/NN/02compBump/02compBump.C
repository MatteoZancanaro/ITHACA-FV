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
Description
    Example of steady NS Reduction Problem
SourceFiles
    03steadyNS.C
\*---------------------------------------------------------------------------*/

#include <torch/script.h>
#include <torch/torch.h>
#include "torch2Eigen.H"
#include "CompressibleSteadyNS.H"
#include "ReducedCompressibleSteadyNS.H"
#include "RBFMotionSolver.H"
#include "ITHACAstream.H"
#include "ITHACAPOD.H"
#include "forces.H"
#include "IOmanip.H"

using namespace ITHACAtorch::torch2Eigen;

class CompressibleSteadyNN : public CompressibleSteadyNS
{
    public:
        /// Constructor
        CompressibleSteadyNN(int argc, char* argv[])
            :
            CompressibleSteadyNS(argc, argv)
        {
            ITHACAparameters* para = ITHACAparameters::getInstance();
            NUmodes = para->ITHACAdict->lookupOrDefault<label>("NmodesUproj", 10);
            NNutModes = para->ITHACAdict->lookupOrDefault<label>("NmodesNutProj", 10);

            Net->push_back(torch::nn::Linear(NUmodes, 128));
            Net->push_back(torch::nn::ReLU());
            Net->push_back(torch::nn::Linear(128, 64));
            Net->push_back(torch::nn::ReLU());
            Net->push_back(torch::nn::Linear(64, NNutModes));
            optimizer = new torch::optim::Adam(Net->parameters(),
                                               torch::optim::AdamOptions(2e-2));
            // std::cerr << "LIFTTTTTTTTTTTTTTTTTTTTTTTTTTT " << inletIndex.rows() << std::endl;
            // exit(0);
        };

    label NUmodes;
    label NNutModes;

    ////////////////////////////////////
    // Eddy viscosity initialization //
    //////////////////////////////////
    
    Eigen::MatrixXd bias_inp;
    Eigen::MatrixXd scale_inp;
    Eigen::MatrixXd bias_out;
    Eigen::MatrixXd scale_out;

    Eigen::MatrixXd coeffL2Nut;
    Eigen::MatrixXd coeffL2U;

    torch::Tensor coeffL2U_tensor;
    torch::Tensor coeffL2Nut_tensor;

    torch::nn::Sequential Net;
    torch::optim::Optimizer* optimizer;
    torch::jit::script::Module netTorchscript;

    void loadNet(word filename)
        {
            std::string Msg = filename +
                              " is not existing, please run the training stage of the net with the correct number of modes for U and Nut";
            M_Assert(ITHACAutilities::check_file(filename), Msg.c_str());
            netTorchscript = torch::jit::load(filename);
            cnpy::load(bias_inp, "ITHACAoutput/NN/minAnglesInp_" + name(
                           NUmodes) + "_" + name(NNutModes) + ".npy");
            cnpy::load(scale_inp, "ITHACAoutput/NN/scaleAnglesInp_" + name(
                           NUmodes) + "_" + name(NNutModes) + ".npy");
            cnpy::load(bias_out, "ITHACAoutput/NN/minOut_" + name(NUmodes) + "_" + name(
                           NNutModes) + ".npy");
            cnpy::load(scale_out, "ITHACAoutput/NN/scaleOut_" + name(NUmodes) + "_" + name(
                           NNutModes) + ".npy");
            netTorchscript.eval();
        }

        // This function computes the coefficients which are later used for training
        void getTurbNN()
        {
            if (!ITHACAutilities::check_folder("ITHACAoutput/NN/coeffs"))
            {
                mkDir("ITHACAoutput/NN/coeffs");
                // Read Fields for Train
                PtrList<volVectorField> UfieldTrain;
                PtrList<volScalarField> PfieldTrain;
                PtrList<volScalarField> nutFieldsTrain;
                ITHACAstream::readMiddleFields(UfieldTrain, _U(),
                                               "./ITHACAoutput/Offline/");
                ITHACAstream::readMiddleFields(PfieldTrain, _p(),
                                               "./ITHACAoutput/Offline/");
                auto nut_train = _mesh().lookupObject<volScalarField>("nut");
                ITHACAstream::readMiddleFields(nutFieldsTrain, nut_train,
                                               "./ITHACAoutput/Offline/");
                /// Compute the coefficients for train
                std::cout << "Computing the coefficients for U train" << std::endl;
                Eigen::MatrixXd coeffL2U_train = ITHACAutilities::getCoeffs(UfieldTrain,
                                                 Umodes,
                                                 0, true);
                std::cout << "Computing the coefficients for p train" << std::endl;
                Eigen::MatrixXd coeffL2P_train = ITHACAutilities::getCoeffs(PfieldTrain,
                                                 Pmodes,
                                                 0, true);
                std::cout << "Computing the coefficients for nuT train" << std::endl;
                Eigen::MatrixXd coeffL2Nut_train = ITHACAutilities::getCoeffs(nutFieldsTrain,
                                                   nutModes,
                                                   0, true);
                coeffL2U_train.transposeInPlace();
                coeffL2P_train.transposeInPlace();
                coeffL2Nut_train.transposeInPlace();
                cnpy::save(coeffL2U_train, "ITHACAoutput/NN/coeffs/coeffL2UTrain.npy");
                cnpy::save(coeffL2P_train, "ITHACAoutput/NN/coeffs/coeffL2PTrain.npy");
                cnpy::save(coeffL2Nut_train, "ITHACAoutput/NN/coeffs/coeffL2NutTrain.npy");
                // Read Fields for Test
                PtrList<volVectorField> UfieldTest;
                PtrList<volScalarField> PfieldTest;
                PtrList<volScalarField> nutFieldsTest;
                /// Compute the coefficients for test
                ITHACAstream::readMiddleFields(UfieldTest, _U(),
                                               "./ITHACAoutput/checkOff/");
                ITHACAstream::readMiddleFields(PfieldTest, _p(),
                                               "./ITHACAoutput/checkOff/");
                auto nut_test = _mesh().lookupObject<volScalarField>("nut");
                ITHACAstream::readMiddleFields(nutFieldsTest, nut_test,
                                               "./ITHACAoutput/checkOff/");
                // Compute the coefficients for test
                Eigen::MatrixXd coeffL2U_test = ITHACAutilities::getCoeffs(UfieldTest,
                                                Umodes,
                                                0, true);
                Eigen::MatrixXd coeffL2P_test = ITHACAutilities::getCoeffs(PfieldTest,
                                                Pmodes,
                                                0, true);
                Eigen::MatrixXd coeffL2Nut_test = ITHACAutilities::getCoeffs(nutFieldsTest,
                                                  nutModes,
                                                  0, true);
                coeffL2U_test.transposeInPlace();
                coeffL2P_test.transposeInPlace();
                coeffL2Nut_test.transposeInPlace();
                cnpy::save(coeffL2U_test, "ITHACAoutput/NN/coeffs/coeffL2UTest.npy");
                cnpy::save(coeffL2P_test, "ITHACAoutput/NN/coeffs/coeffL2PTest.npy");
                cnpy::save(coeffL2Nut_test, "ITHACAoutput/NN/coeffs/coeffL2NutTest.npy");
            }
        }

        Eigen::MatrixXd evalNet(Eigen::MatrixXd a, Eigen::MatrixXd mu_now)
        {
            Eigen::MatrixXd xpred(a.rows() + mu_now.rows(), 1);

            if (xpred.cols() != 1)
            {
                throw std::runtime_error("Predictions for more than one sample not supported yet.");
            }

            xpred.bottomRows(a.rows()) = a;
            xpred.topRows(mu_now.rows()) = mu_now;
            xpred = xpred.array() * scale_inp.array() + bias_inp.array() ;
            xpred.transposeInPlace();
            torch::Tensor xTensor = eigenMatrix2torchTensor(xpred);
            torch::Tensor out;
            std::vector<torch::jit::IValue> XTensorInp;
            XTensorInp.push_back(xTensor);
            out = netTorchscript.forward(XTensorInp).toTensor();
            Eigen::MatrixXd g = torchTensor2eigenMatrix<double>(out);
            g.transposeInPlace();
            g = (g.array() - bias_out.array()) / scale_out.array();
            return g;
        }

        // Function to eval the NN once the input is provided
        Eigen::MatrixXd evalNet(Eigen::MatrixXd a)
        {
            netTorchscript.eval();
            std::cerr << "debug point 1" << std::endl;
            a.transposeInPlace();
            std::cerr << "debug point 2" << std::endl;
            std::cout << a.size() << std::endl;
            std::cout << bias_inp.size() << std::endl;
            std::cout << scale_inp.size() << std::endl;
            a = (a.array() - bias_inp.array()) / scale_inp.array();
            std::cerr << "debug point 3" << std::endl;
            torch::Tensor aTensor = eigenMatrix2torchTensor(a);
            std::cerr << "debug point 4" << std::endl;
            torch::Tensor out;// = Net->forward(aTensor);
            std::cerr << "debug point 5" << std::endl;
            std::vector<torch::jit::IValue> XTensorInp;
            std::cerr << "debug point 6" << std::endl;
            XTensorInp.push_back(aTensor);
            std::cerr << "debug point 7" << std::endl;
            out = netTorchscript.forward(XTensorInp).toTensor();
            std::cerr << "debug point 8" << std::endl;
            Eigen::MatrixXd g = torchTensor2eigenMatrix<double>(out);
            std::cerr << "debug point 9" << std::endl;
            g = g.array() * scale_out.array() + bias_out.array();
            std::cerr << "debug point 10" << std::endl;
            g.transposeInPlace();
            return g;
        }
};

class ReducedCompressibleSteadyNN : public ReducedCompressibleSteadyNS
{
    public:
        /// Constructor
        explicit ReducedCompressibleSteadyNN(CompressibleSteadyNN& FOMproblem)
            :
            problem(&FOMproblem),
            ReducedCompressibleSteadyNS(FOMproblem)
        {}

        /// Full problem.
        CompressibleSteadyNN* problem;

    void setOnlineVelocity(Eigen::MatrixXd vel)
    {
        M_Assert(problem->inletIndex.rows() == vel.size(),
                 "Imposed boundary conditions dimensions do not match given values matrix dimensions into setOnlineVelocity");
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

    void projectReducedOperators(int NmodesUproj, int NmodesPproj, int NmodesEproj)
    {
        PtrList<volVectorField> gradModP;
        for (label i = 0; i < NmodesPproj; i++)
        {
            gradModP.append(fvc::grad(problem->Pmodes[i]));
        }
        label NmodesUprojL = NmodesUproj + problem->inletIndex.rows();
        std::cout << ULmodes.size() << std::endl;
        projGradModP = ULmodes.project(gradModP, NmodesUprojL);
    }

    void solveOnlineCompressible(int NmodesUproj, int NmodesPproj, int NmodesEproj, int NmodesNutProj, Eigen::MatrixXd mu_now,
                                word Folder = "./ITHACAoutput/Online/")
    {
        counter++;

        // Residuals initialization
        scalar residualNorm(1);
        scalar residualJump(1);
        Eigen::MatrixXd uResidualOld = Eigen::MatrixXd::Zero(1, NmodesUproj + problem->inletIndex.rows());
        Eigen::MatrixXd eResidualOld = Eigen::MatrixXd::Zero(1, NmodesEproj);
        Eigen::MatrixXd pResidualOld = Eigen::MatrixXd::Zero(1, NmodesPproj);
        Eigen::VectorXd uResidual(Eigen::Map<Eigen::VectorXd>(uResidualOld.data(),
                                  NmodesUproj + problem->inletIndex.rows()));
        Eigen::VectorXd eResidual(Eigen::Map<Eigen::VectorXd>(eResidualOld.data(),
                                  NmodesEproj));
        Eigen::VectorXd pResidual(Eigen::Map<Eigen::VectorXd>(pResidualOld.data(),
                                  NmodesPproj));
        // Parameters definition
        ITHACAparameters* para = ITHACAparameters::getInstance();
        float residualJumpLim =
            para->ITHACAdict->lookupOrDefault<float>("residualJumpLim", 1e-5);
        float normalizedResidualLim =
            para->ITHACAdict->lookupOrDefault<float>("normalizedResidualLim", 1e-5);
        int maxIter =
            para->ITHACAdict->lookupOrDefault<float>("maxIter", 2000);
        bool closedVolume = false;
        label csolve = 0;

        // Full variables initialization
        fluidThermo& thermo = problem->pThermo();
        volVectorField& U = problem->_U();
        volScalarField& P = problem->pThermo->p();
        volScalarField& E = problem->pThermo->he();
        volScalarField& nut = const_cast<volScalarField&>
                      (problem->_mesh().lookupObject<volScalarField>("nut"));
        volScalarField& rho = problem->_rho();
        volScalarField& psi = problem->_psi();
        surfaceScalarField& phi = problem->_phi();
        Time& runTime = problem->_runTime();
        fvMesh& mesh = problem->_mesh();
        fv::options& fvOptions = problem->_fvOptions();
        scalar cumulativeContErr = problem->cumulativeContErr;
        // Reduced variables initialization
        Eigen::MatrixXd u = Eigen::MatrixXd::Zero(NmodesUproj + problem->inletIndex.rows(), 1);
        Eigen::MatrixXd e = Eigen::MatrixXd::Zero(NmodesEproj, 1);
        Eigen::MatrixXd p = Eigen::MatrixXd::Zero(NmodesPproj, 1);
        Eigen::MatrixXd nutCoeff = ITHACAutilities::getCoeffs(nut, problem->nutModes, NmodesNutProj, true);

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

            // volScalarField nueff = problem->turbulence->nuEff();
            // ITHACAstream::exportSolution(nueff, name(csolve), "./ITHACAoutput/testNuEff/");

            //problem->getUmatrix(U);
            fvVectorMatrix UEqnR
            (
                fvm::div(phi, U) 
                - fvc::div((rho * problem->turbulence->nuEff()) * dev2(T(fvc::grad(U)))) 
                - fvm::laplacian(rho * problem->turbulence->nuEff(), U) 
                ==
                fvOptions(rho, U)
            );
            UEqnR.relax();
            fvOptions.constrain(UEqnR);
            //RedLinSysU = ULmodes.project(problem->Ueqn_global(), NmodesUproj);
            RedLinSysU = ULmodes.project(UEqnR, NmodesUproj + problem->inletIndex.rows());
            Eigen::MatrixXd projGradP = projGradModP * p;
            RedLinSysU[1] = RedLinSysU[1] - projGradP;
            //std::cout << "U " << u << std::endl;
            std::cout << "Ure " << uResidual << std::endl;
            std::cout << "size " << RedLinSysU.size() << std::endl;
            std::cout << "size 1: " << RedLinSysU[0].rows() << std::endl;
            std::cout << "size 2: " << RedLinSysU[0].cols() << std::endl;
            //std::cout << "matrix: " << RedLinSysU[0] << std::endl;
            u = reducedProblem::solveLinearSys(RedLinSysU, u, uResidual, vel_now, "fullPivLu");//"bdcSvd");
            ULmodes.reconstruct(U, u, "U");
            //solve(problem->Ueqn_global() == -problem->getGradP(P)); //For debug purposes only, second part only useful when using uEqn_global==-getGradP
            //solve(UEqnR == -problem->getGradP(P)); //For debug purposes only, second part only useful when using uEqn_global==-getGradP
            fvOptions.correct(U);
            
            //Energy equation phase
            //problem->getEmatrix(U, P);
            fvScalarMatrix EEqnR
            (
                fvm::div(phi, E)
                + fvc::div(phi, volScalarField("Ekp", 0.5 * magSqr(U) + P / rho))
                - fvm::laplacian(problem->turbulence->alphaEff(), E)
                ==
                fvOptions(rho, E)
            );

            EEqnR.relax();
            fvOptions.constrain(EEqnR);
            // List<Eigen::MatrixXd> RedLinSysE = problem->Emodes.project(
            //                                        problem->Eeqn_global(), NmodesEproj);
            List<Eigen::MatrixXd> RedLinSysE = problem->Emodes.project(EEqnR, NmodesEproj);

            e = reducedProblem::solveLinearSys(RedLinSysE, e, eResidual);
            problem->Emodes.reconstruct(E, e, "e");
            //problem->Eeqn_global().solve(); //For debug purposes only
            //EEqnR.solve(); //For debug purposes only
            fvOptions.correct(E);
            thermo.correct(); // Here are calculated both temperature and density based on P,U and he.
            // Pressure equation phase
            // constrainPressure(P, rho, U, problem->getPhiHbyA(problem->Ueqn_global(), U, P),
            //                   problem->getRhorAUf(problem->Ueqn_global()));// Update the pressure BCs to ensure flux consistency
            // surfaceScalarField phiHbyACalculated = problem->getPhiHbyA(problem->Ueqn_global(), U, P);
            constrainPressure(P, rho, U, problem->getPhiHbyA(UEqnR, U, P),
                              problem->getRhorAUf(UEqnR));// Update the pressure BCs to ensure flux consistency
            surfaceScalarField phiHbyACalculated = problem->getPhiHbyA(UEqnR, U, P);
            closedVolume = adjustPhi(phiHbyACalculated, U, P);

            List<Eigen::MatrixXd> RedLinSysP;
            while (problem->_simple().correctNonOrthogonal())
            {
                // problem->getPmatrix(problem->Ueqn_global(), U, P);
                volScalarField rAU(1.0 / UEqnR.A()); // Inverse of the diagonal part of the U equation matrix
                volVectorField HbyA(constrainHbyA(rAU * UEqnR.H(), U, P)); // H is the extra diagonal part summed to the r.h.s. of the U equation
                surfaceScalarField phiHbyA("phiHbyA", fvc::interpolate(rho)*fvc::flux(HbyA));
                surfaceScalarField rhorAUf("rhorAUf", fvc::interpolate(rho * rAU));
                fvScalarMatrix PEqnR
                (
                    fvc::div(phiHbyA)
                    -fvm::laplacian(rhorAUf,P)
                    ==
                    fvOptions(psi, P, rho.name())
                );
                PEqnR.setReference
                (
                    problem->_pressureControl().refCell(),
                    problem->_pressureControl().refValue()
                );

                // RedLinSysP = problem->Pmodes.project(problem->Peqn_global(), NmodesPproj);
                RedLinSysP = problem->Pmodes.project(PEqnR, NmodesPproj);

                p = reducedProblem::solveLinearSys(RedLinSysP, p, pResidual);
                problem->Pmodes.reconstruct(P, p, "p");
                //problem->Peqn_global().solve(); //For debug purposes only
                //PEqnR.solve(); //For debug purposes only

                if (problem->_simple().finalNonOrthogonalIter())
                {
                    // phi = problem->getPhiHbyA(problem->Ueqn_global(), U, P) + problem->Peqn_global().flux();
                    phi = problem->getPhiHbyA(UEqnR, U, P) + PEqnR.flux(); //Are you sure you still can use it?????????????????????????????????
                }
            }

    #include "incompressible/continuityErrs.H"
            P.relax();// Explicitly relax pressure for momentum corrector
            //U = problem->HbyA() - (1.0 / problem->Ueqn_global().A()) * problem->getGradP(P);
            U = problem->HbyA() - (1.0 / UEqnR.A()) * problem->getGradP(P);

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
            std::cout << "Ures = " << (uResidual.cwiseAbs()).sum() / (RedLinSysU[1].cwiseAbs()).sum() << std::endl;
            std::cout << "Eres = " << (eResidual.cwiseAbs()).sum() / (RedLinSysE[1].cwiseAbs()).sum() << std::endl;
            std::cout << "Pres = " << (pResidual.cwiseAbs()).sum() / (RedLinSysP[1].cwiseAbs()).sum() << std::endl;
            // std::cout << "U = " << u << std::endl;
            // std::cout << "E = " << e << std::endl;
            // std::cout << "P = " << p << std::endl;
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
            std::cerr << "debug point 1" << std::endl;
            // Eigen::MatrixXd coeffL2U_prova = ITHACAutilities::getCoeffs(U, ULmodes,0, true);
            // Eigen::MatrixXd nutCoeff = problem->evalNet(coeffL2U_prova.bottomRows(NmodesUproj), mu_now);
            Eigen::MatrixXd nutCoeff = problem->evalNet(u.bottomRows(NmodesUproj), mu_now); ///////////////////////////////////////////////////////////////////// Lifting functions are not needed
            std::cerr << "debug point 2" << std::endl;
            // volScalarField nutProva(nut);
            // problem->nutModes.reconstruct(nutProva, nutCoeff, "nutProva");
            // problem->nutModes.reconstruct(nut, nutCoeff, "nut");
            std::cerr << "debug point 3" << std::endl;
        }
        label k = 1;
        // U.rename("Ur");
        // P.rename("Pr");
        // E.rename("Er");
        // nut.rename("nutR");        
        ITHACAstream::exportSolution(U, name(counter), Folder);
        ITHACAstream::exportSolution(P, name(counter), Folder);
        ITHACAstream::exportSolution(E, name(counter), Folder);
        ITHACAstream::exportSolution(nut, name(counter), Folder);
    }

};

class tutorial02 : public CompressibleSteadyNN
{
public:
    /// Constructor
    explicit tutorial02(int argc, char* argv[])
        :
        CompressibleSteadyNN(argc, argv)
    {
        dyndict = new IOdictionary
            (
                IOobject
                (
                    "dynamicMeshDictRBF",
                    "./constant",
                    _mesh(),
                    IOobject::MUST_READ,
                    IOobject::NO_WRITE
                )
            );
        ITHACAutilities::getPointsFromPatch(_mesh(), 0, top0, top0_ind);
        ITHACAutilities::getPointsFromPatch(_mesh(), 1, bot0, bot0_ind);
        std::cout << _mesh().points().size() << std::endl;
        ms = new RBFMotionSolver(_mesh(), *dyndict);
        vectorField motion(ms->movingPoints().size(), vector::zero);
        movingIDs = ms->movingIDs();
        x0 = ms->movingPoints();
        curX = x0;
        point0 = ms->curPoints();

        /// Export intermediate steps
        middleExport = para->ITHACAdict->lookupOrDefault<bool>("middleExport", true);
    }

    List<vector> top0;
    List<vector> bot0;
    labelList top0_ind;
    labelList bot0_ind;
    IOdictionary* dyndict;
    RBFMotionSolver* ms;
    labelList movingIDs;
    List<vector> x0;
    List<vector> curX;
    vectorField point0;
    vectorField point;
    ITHACAparameters* para = ITHACAparameters::getInstance();


    double f1(double chord, double x)
    {
        double res = chord * (std::pow((x)/chord,0.5)*(1-(x)/chord))/(std::exp(15*(x)/chord));
        return res;
    }

    List<vector> moveBasis(const List<vector>& originalPoints, double par)
    {
            List<vector> movedPoints(originalPoints);
            for(int i = 0; i<originalPoints.size(); i++)
            {
                movedPoints[i][2]+= par*f1(1,movedPoints[i][0]);
            }       

            return movedPoints;
    }

    void updateMesh(double parTop = 0, double parBot = 0)
    {
        _mesh().movePoints(point0);
        if(parTop!=0 || parBot!=0)
        {
            std::cout << parTop << std::endl;
            List<vector> top0_cur = moveBasis(top0, parTop);
            List<vector> bot0_cur = moveBasis(bot0, parBot);
            ITHACAutilities::setIndices2Value(top0_ind, top0_cur, movingIDs, curX);
            ITHACAutilities::setIndices2Value(bot0_ind, bot0_cur, movingIDs, curX);
            ms->setMotion(curX - x0);
            point = ms->curPoints();
            _mesh().movePoints(point);
        }
    }

    /// Perform an Offline solve
    void offlineSolve(word folder = "./ITHACAoutput/Offline/")
    {
        /// Velocity field
        volVectorField& U = _U();
        /// Pressure field
        volScalarField& p = _p();
        /// Energy field
        volScalarField& E = _E();

        // if the offline solution is already performed but POD modes are not present, then read the fields
        if (offline && !ITHACAutilities::check_folder("./ITHACAoutput/POD/1"))
        {
            ITHACAstream::readMiddleFields(Ufield, U, folder);
            ITHACAstream::readMiddleFields(Efield, E, folder);
            ITHACAstream::readMiddleFields(Pfield, p, folder);
            /// Eddy viscosity field
            auto nut = _mesh().lookupObject<volScalarField>("nut");
            ITHACAstream::readMiddleFields(nutFields, nut, folder);
            mu_samples = ITHACAstream::readMatrix("./parsOff_mat.txt");
        }
        // if offline stage ha snot been performed, then perform it
        else if (!offline)
        {
            //Vector<double> Uinl(250, 0, 0);
            //Vector<double> Uinl(170, 0, 0);
            double UIFinit = para->ITHACAdict->lookupOrDefault<double>("UIFinit", 170);
            Vector<double> Uinl(UIFinit, 0, 0);

            for (label i = 0; i < mu.rows(); i++)
            {
                //std::cout << "Current mu = " << mu(i, 0) << std::endl;
                //changeViscosity(mu(i, 0));
                updateMesh(mu(i,0),mu(i,1));
                ITHACAstream::writePoints(_mesh().points(), folder, name(i + 1) + "/polyMesh/");
                //assignIF(_U(), Uinl);
                assignIF(_U(), Uinl);
                truthSolve(folder);

                label j=1;
                word polyMesh2beLinked = folder + name(i+1) + "/" + "polyMesh/";
                while (ITHACAutilities::check_folder(folder + name(i+1) + "/" + name(j)))
                {
                    word folderContLink = folder + name(i+1) + "/" + name(j) + "/";
                    system("ln -s  $(readlink -f " + polyMesh2beLinked + ") " + folderContLink + " >/dev/null 2>&1");
                    j++;

                }
            }
        }
    }
};

int main(int argc, char* argv[])
{
    // Construct the tutorial object  
    tutorial02 example(argc, argv);
    // volScalarField& s = example._p();
    // Info << example._p().boundaryFieldRef()[1] << endl;
    // Info << example._p().boundaryFieldRef()[1][0] << endl;
    // example._p().boundaryFieldRef()[1][0] = 11; 
    // Info << example._p().boundaryFieldRef()[1] << endl;
    // freestreamPressureFvPatchScalarField& Tpatch =
    //     refCast<freestreamPressureFvPatchScalarField>(s.boundaryFieldRef()[1]);
    // scalarField& gradTpatch = Tpatch.freestreamValue();
    // Info << gradTpatch << endl;
    // forAll(gradTpatch, faceI)
    // {
    //     double value = valueList[faceI];
    //     gradTpatch[faceI] = value;
    // }

    //exit(0);

    // try
    // {
    // 	int i=2;
    // 	if(i<3)
    // 	{
    // 		throw("i piccolo");
    // 	}
    // }

    // catch(const char* message)
    // {
    // 	cerr<<"Succede che "<<message<< endl;
    // }

    // exit(0);

    ITHACAparameters* para = ITHACAparameters::getInstance();

    std::ifstream exFileOff("./parsOff_mat.txt");
    if (exFileOff)
    {
        example.mu  = ITHACAstream::readMatrix("./parsOff_mat.txt");
    }

    else
    {
    	int OffNum = para->ITHACAdict->lookupOrDefault<int>("OffNum", 100);
    	double BumpAmp = para->ITHACAdict->lookupOrDefault<double>("BumpAmp", 0.1);
        example.mu.resize(OffNum, 2);
        Eigen::MatrixXd parTop = ITHACAutilities::rand(example.mu.rows(), 1, 0, BumpAmp);
    	Eigen::MatrixXd parBot = ITHACAutilities::rand(example.mu.rows(), 1, -BumpAmp, 0);
        example.mu.leftCols(1) = parTop;
        example.mu.rightCols(1) = parBot;
        ITHACAstream::exportMatrix(example.mu , "parsOff", "eigen", "./");
    }

    // Eigen::MatrixXd parOn;
    // std::ifstream exFileOn("./parsOn_mat.txt");
    // if (exFileOn)
    // {
    //     parOn = ITHACAstream::readMatrix("./parsOn_mat.txt");
    // }

    // else
    // {
    //     parOn = ITHACAutilities::rand(20, 1, 1.00e-05, 1.00e-02);
    //     ITHACAstream::exportMatrix(parOn, "parsOn", "eigen", "./");
    // }
    // 
    Eigen::MatrixXd parsOn;
    std::ifstream exFileOn("./parsOn_mat.txt");
    if (exFileOn)
    {
        parsOn  = ITHACAstream::readMatrix("./parsOn_mat.txt");
    }

    else
    {
    	int OnNum = para->ITHACAdict->lookupOrDefault<int>("OnNum", 20);
    	double BumpAmp = para->ITHACAdict->lookupOrDefault<double>("BumpAmp", 0.1);
        parsOn.resize(OnNum, 2);
        Eigen::MatrixXd parTopOn = ITHACAutilities::rand(OnNum, 1, 0, BumpAmp);
    	Eigen::MatrixXd parBotOn = ITHACAutilities::rand(OnNum, 1, -BumpAmp, 0);
        parsOn.leftCols(1) = parTopOn;
        parsOn.rightCols(1) = parBotOn;
        ITHACAstream::exportMatrix(parsOn , "parsOn", "eigen", "./");
    }
    // Read some parameters from file
    int NmodesUout = para->ITHACAdict->lookupOrDefault<int>("NmodesUout", 15);
    int NmodesPout = para->ITHACAdict->lookupOrDefault<int>("NmodesPout", 15);
    int NmodesEout = para->ITHACAdict->lookupOrDefault<int>("NmodesEout", 15);
    int NmodesNutOut = para->ITHACAdict->lookupOrDefault<int>("NmodesNutOut", 15);
    int NmodesUproj = para->ITHACAdict->lookupOrDefault<int>("NmodesUproj", 10);
    int NmodesPproj = para->ITHACAdict->lookupOrDefault<int>("NmodesPproj", 10);
    int NmodesEproj = para->ITHACAdict->lookupOrDefault<int>("NmodesEproj", 10);
    int NmodesNutProj = para->ITHACAdict->lookupOrDefault<int>("NmodesNutProj", 10);
    //Set the inlet boundaries patch 0 directions x and y
    label idInl = example._mesh().boundaryMesh().findPatchID("inlet");
    example.inletIndex.resize(1, 2);
    example.inletIndex(0, 0) = idInl;
    example.inletIndex(0, 1) = 0;

    example.updateMesh();

    //Perform the offline solve
    example.offlineSolve();
    //exit(0);




    //Read the lift field
    ITHACAstream::read_fields(example.liftfield, example._U(), "./lift/");
    ITHACAutilities::normalizeFields(example.liftfield);

    if(!ITHACAutilities::check_folder("./ITHACAoutput/POD/1")) // If POD has already been performed, you do not have to omogenize anything
    {
	    // Homogenize the snapshots
	    example.computeLift(example.Ufield, example.liftfield, example.Uomfield);
	}
    
    // Perform POD on velocity and pressure and store the first 10 modes
    example.updateMesh(); // Move the mesh to the original geometry to get the modes into a mid mesh

    ITHACAPOD::getModes(example.Uomfield, example.Umodes, example._U().name(), example.podex, 0, 0,
                        NmodesUout);
    ITHACAPOD::getModes(example.Pfield, example.Pmodes, example._p().name(), example.podex, 0, 0,
                        NmodesPout);
    ITHACAPOD::getModes(example.Efield, example.Emodes, example._E().name(), example.podex, 0, 0,
                        NmodesEout);
    ITHACAPOD::getModes(example.nutFields, example.nutModes, "nut", example.podex, 0, 0,
                        NmodesNutOut);

    if(!ITHACAutilities::check_folder("./ITHACAoutput/checkOff"))
    {
        example.updateMesh();
        tutorial02 checkOff(argc, argv);
        checkOff.mu  = ITHACAstream::readMatrix("./parsOn_mat.txt");
        //Set the inlet boundaries patch 0 directions x and y
        label idInl = checkOff._mesh().boundaryMesh().findPatchID("inlet");
        checkOff.inletIndex.resize(1, 2);
        checkOff.inletIndex(0, 0) = idInl;
        checkOff.inletIndex(0, 1) = 0;
        //Perform the offline solve
        checkOff.offline = false;
        //checkOff.middleExport = false;
        checkOff.restart();
        checkOff.offlineSolve("./ITHACAoutput/checkOff/");
        checkOff.offline = true;
    }

    std::cerr << "debug point 100" << std::endl;
    example.getTurbNN();
    std::cerr << "debug point 200" << std::endl;
    example.loadNet("ITHACAoutput/NN/Net_" + name(example.NUmodes) + "_" + name(
                        example.NNutModes) + ".pt");

    // Create the reduced object
    ReducedCompressibleSteadyNN reduced(example);
    
    // // Reads inlet volocities boundary conditions.
    // word vel_file(para->ITHACAdict->lookup("online_velocities"));
    // Eigen::MatrixXd vel = ITHACAstream::readMatrix(vel_file);
    // 
    // Eigen::MatrixXd mu_tr;
    // Eigen::MatrixXd u_tr;
    // Eigen::MatrixXd g_tr(mu_tr.rows(),NmodesNutProj);
    // Eigen::VectorXd snaps_tr;

    // mu_tr = ITHACAstream::readMatrix("./parsOff_mat.txt");
    // cnpy::load(u_tr, "ITHACAoutput/NN/coeffs/coeffL2UTrain.npy");
    // snaps_tr=ITHACAstream::readMatrix("ITHACAoutput/Offline/snaps");
    // auto nut_test = example._mesh().lookupObject<volScalarField>("nut");
    // Eigen::MatrixXd a_tr(u_tr.cols(),1);
    // Eigen::MatrixXd mu_trv(2,1);

    // label k=-1;
    // for(label i=0; i<mu_tr.rows(); i++)
    // {
    //     //k=k+snaps_tr[i]; //final solution
    //     k++;
    //     std::cerr << "debug point 2" << std::endl;
    //     a_tr = u_tr.row(k).transpose();
    //     std::cerr << "debug point 3" << std::endl;
    //     mu_trv = mu_tr.row(i).transpose();
    //     std::cerr << "debug point 4" << std::endl;
    //     //std::cout << a_tr.topRows(NmodesUproj) << std::endl;
    //     //std::cout << mu_trv << std::endl;
    //     Eigen::VectorXd g_intermedia = example.evalNet(a_tr.topRows(NmodesUproj), mu_trv);
    //     std::cerr << "debug point 3" << std::endl;
    //     //g_tr.row(i) = g_intermedia;
    //     std::cerr << "debug point 4" << std::endl;
    //     example.nutModes.reconstruct(nut_test, g_intermedia, "nut");
    //     std::cerr << "debug point 5" << std::endl;
    //     //ITHACAstream::exportSolution(nut_test, name(i+1), "./ITHACAoutput/testNutTrain/"); // final solution
    //     ITHACAstream::exportSolution(nut_test, name(i+1), "./ITHACAoutput/testNutTrainBegin/");
    // }

    // Eigen::MatrixXd mu_te;
    // Eigen::MatrixXd u_te;
    // Eigen::VectorXd snaps_te;

    // mu_te = ITHACAstream::readMatrix("./parsOn_mat.txt");
    // cnpy::load(u_te, "ITHACAoutput/NN/coeffs/coeffL2UTest.npy");
    // snaps_te=ITHACAstream::readMatrix("ITHACAoutput/checkOff/snaps");
    // //auto nut_test = example._mesh().lookupObject<volScalarField>("nut");
    // Eigen::MatrixXd a_te(u_te.cols(),1);
    // Eigen::MatrixXd mu_tev(2,1);

    // k=-1;
    // for(label i=0; i<mu_te.rows(); i++)
    // {
    //     //k=k+snaps_te[i]; //final solution
    //     k++;
    //     std::cerr << "debug point 2" << std::endl;
    //     a_te = u_te.row(k).transpose();
    //     std::cerr << "debug point 3" << std::endl;
    //     mu_tev = mu_te.row(i).transpose();
    //     std::cerr << "debug point 4" << std::endl;
    //     //std::cout << a_tr.topRows(NmodesUproj) << std::endl;
    //     //std::cout << mu_trv << std::endl;
    //     Eigen::VectorXd g_intermedia = example.evalNet(a_te.topRows(NmodesUproj), mu_tev);
    //     std::cerr << "debug point 3" << std::endl;
    //     //g_tr.row(i) = g_intermedia;
    //     std::cerr << "debug point 4" << std::endl;
    //     example.nutModes.reconstruct(nut_test, g_intermedia, "nut");
    //     std::cerr << "debug point 5" << std::endl;
    //     //ITHACAstream::exportSolution(nut_test, name(i+1), "./ITHACAoutput/testNutTest/"); // final solution
    //     ITHACAstream::exportSolution(nut_test, name(i+1), "./ITHACAoutput/testNutTestBegin/");
    // }


    // exit(0);

    
    Eigen::MatrixXd vel(1,1);
    vel(0,0) = example._U().boundaryFieldRef()[idInl][0][0];

    //Perform the online solutions
    for (label k = 0; k < parsOn.rows(); k++)
    {
        //scalar mu_now = parOn(k, 0);
        //example.changeViscosity(mu_now);
        example.updateMesh();
        example.updateMesh(parsOn(k,0), parsOn(k,1));
        ITHACAstream::writePoints(example._mesh().points(), "./ITHACAoutput/Online_"+name(NmodesUproj)+"_"+name(NmodesNutProj)+"/", name(k + 1) + "/polyMesh/");
        std::cout << example.inletIndex.rows() << std::endl;
        reduced.setOnlineVelocity(vel);
        std::cerr << "debug point 10" << std::endl;
        reduced.projectReducedOperators(NmodesUproj, NmodesPproj, NmodesEproj); ///////////////////////////////////////////////////// Dimension for U should be increased by the number of lifting functions, right??? (look into the function)
        std::cerr << "debug point 11" << std::endl;
        example.restart();
        std::cerr << "debug point 12" << std::endl;
        example.turbulence->validate(); ///////////////////////////////////////////////// Is it needed to validate the nut field?
        std::cerr << "eccolooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo" << std::endl;
        Eigen::MatrixXd mu_now = parsOn.row(k);
        mu_now.transposeInPlace();
        reduced.solveOnlineCompressible(NmodesUproj, NmodesPproj, NmodesEproj, NmodesNutProj, mu_now, "./ITHACAoutput/Online_"+name(NmodesUproj)+"_"+name(NmodesNutProj)+"/");
    }

    if(!ITHACAutilities::check_folder("./ITHACAoutput/checkOff"))
    {
        tutorial02 checkOff(argc, argv);
        checkOff.mu  = ITHACAstream::readMatrix("./parsOn_mat.txt");
        //Set the inlet boundaries patch 0 directions x and y
        label idInl = checkOff._mesh().boundaryMesh().findPatchID("inlet");
        checkOff.inletIndex.resize(1, 2);
        checkOff.inletIndex(0, 0) = idInl;
        checkOff.inletIndex(0, 1) = 0;
        //Perform the offline solve
        checkOff.offline = false;
        checkOff.middleExport = false;
        checkOff.restart();
        checkOff.offlineSolve("./ITHACAoutput/checkOff/");

        PtrList<volVectorField> onlineU;
        PtrList<volScalarField> onlineP;
        PtrList<volScalarField> onlineE;
        ITHACAstream::read_fields(onlineU,checkOff._U(),"./ITHACAoutput/Online/");
        ITHACAstream::read_fields(onlineP,checkOff._p(),"./ITHACAoutput/Online/");
        ITHACAstream::read_fields(onlineE,checkOff._E(),"./ITHACAoutput/Online/");
        Eigen::MatrixXd errorU = ITHACAutilities::errorL2Rel(checkOff.Ufield,
                            onlineU);
        Eigen::MatrixXd errorP = ITHACAutilities::errorL2Rel(checkOff.Pfield,
                            onlineP);
        Eigen::MatrixXd errorE = ITHACAutilities::errorL2Rel(checkOff.Efield,
                            onlineE);
        ITHACAstream::exportMatrix(errorU,"errorU", "python", "./ITHACAoutput/checkOff/");
        ITHACAstream::exportMatrix(errorP,"errorP", "python", "./ITHACAoutput/checkOff/");
        ITHACAstream::exportMatrix(errorE,"errorE", "python", "./ITHACAoutput/checkOff/");

        for(label j=0; j<checkOff.mu.rows(); j++)
        {
        	volVectorField Ue = checkOff.Ufield[j] - onlineU[j];
        	volScalarField Pe = checkOff.Pfield[j] - onlineP[j];
        	volScalarField Ee = checkOff.Efield[j] - onlineE[j];
        	Ue.rename("Ue");
        	Pe.rename("Pe");
        	Ee.rename("Ee");
        	ITHACAstream::exportSolution(Ue, name(j+1), "./ITHACAoutput/checkOff/");
	    	ITHACAstream::exportSolution(Pe, name(j+1), "./ITHACAoutput/checkOff/");
	    	ITHACAstream::exportSolution(Ee, name(j+1), "./ITHACAoutput/checkOff/");
        }
    }

    exit(0);
}