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
#include "unsteadyNS.H"
#include "reducedSimpleSteadyNS.H"
#include "reducedSteadyNS.H"
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

// Bump Functions
double f1(double chord, double x)
{
    double res = chord * (std::pow((x + 0.5) / chord,
                                   0.5) * (1 - (x + 0.5) / chord)) / (std::exp(15 * (x + 0.5) / chord));
    return res;
}

double f2(double chord, double x)
{
    double res = chord * std::pow(std::sin(M_PI * std::pow((x + 0.5) / chord,
                                           0.25)), 3);
    return res;
}

double f3(double chord, double x)
{
    double res = chord * std::pow(std::sin(M_PI * std::pow((x + 0.5) / chord,
                                           0.757)), 3);
    return res;
}

double f4(double chord, double x)
{
    double res = chord * std::pow(std::sin(M_PI * std::pow((x + 0.5) / chord,
                                           1.357)), 3);
    return res;
}

double f5(double chord, double x)
{
double res = chord * (std::pow((x + 0.5) / chord,
                                   0.5) * (1 - (x + 0.5) / chord)) / (std::exp(10 * (x + 0.5) / chord));
    return res;
}

// Function to move points according to bump functions and a vector of coefficients
List<vector> moveBasis(const List<vector>& originalPoints, Eigen::MatrixXd par1)
{
    List<vector> movedPoints(originalPoints);

    for (int i = 0; i < originalPoints.size(); i++)
    {
        movedPoints[i][1] += par1(0, 0) * f1(1, movedPoints[i][0]) + par1(0, 1) * f2(1,
                             movedPoints[i][0]) + par1(0, 2) * f3(1, movedPoints[i][0]) + par1(0, 3) * f4(1,
                                     movedPoints[i][0]) + par1(0, 4) * f5(1, movedPoints[i][0]);
    }

    return movedPoints;
}

class NS_geom_par : public unsteadyNS
{
    public:
        explicit NS_geom_par(int argc, char* argv[])
            :
            unsteadyNS(argc, argv),
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
            ITHACAutilities::getPointsFromPatch(mesh, 0, top0, top0_ind);
            ITHACAutilities::getPointsFromPatch(mesh, 1, bot0, bot0_ind);
            ms = new RBFMotionSolver(mesh, *dyndict);
            vectorField motion(ms->movingPoints().size(), vector::zero);
            movingIDs = ms->movingIDs();
            x0 = ms->movingPoints();
            curX = x0;
            point0 = ms->curPoints();
        }

        // Reference to pressure velocity and flux
        volScalarField& p;
        volVectorField& U;
        surfaceScalarField& phi;

        // Axis of rotation in case of changing angle of attack
        vector axis;

        // Pointers to nuT and nuTilda at the initial "time"
        autoPtr<volScalarField> _nut0;
        autoPtr<volScalarField> _nuTilda0;

        // Pointers to nuT and nuTilda
        autoPtr<volScalarField> _nut;
        autoPtr<volScalarField> _nuTilda;

        /// dictionary to store input output infos
        IOdictionary* dyndict;

        // Pointer to the RBFMotionSolver
        RBFMotionSolver* ms;

        // Points on the wind (top and bottom side also) in the reference configuration
        List<vector> wing0;
        List<vector> top0;
        List<vector> bot0;

        // Coordinates of all points of the grid in the reference configuration
        vectorField point0;

        // labelList to identify the moving patches
        labelList movingIDs;
        // Coordinates of the moving points in the reference configuration
        List<vector> x0;

        // Coordinates of the moving points in the current configuration
        List<vector> curX;

        // Indices of the points on the wing
        labelList wing0_ind;
        labelList top0_ind;
        labelList bot0_ind;

        // Function to solve the offline problem
        void OfflineSolve(Eigen::MatrixXd parTop, Eigen::MatrixXd parBot, word Folder)
        {
            fvMesh& mesh = _mesh();
            Time& runTime = _runTime();

            if (offline)
            {
                ITHACAstream::read_fields(Ufield, U, "./ITHACAoutput/Offline/");
                ITHACAstream::read_fields(Pfield, p, "./ITHACAoutput/Offline/");
                volVectorField Usup("Usup", U);
                ITHACAstream::read_fields(supfield, Usup, "./ITHACAoutput/supfield/");
            }

            else
            {
                for (int k = 0; k < parTop.rows(); k++)
                {
                    List<scalar> par(1);
                    mkDir("./defGeom/"+name(k+1));
                    system("cp -r constant defGeom/"+name(k+1)+"/constant");
                    system("cp -r system defGeom/"+name(k+1)+"/system");
                    system("cp -r 0 defGeom/"+name(k+1)+"/0");
                    updateMesh2(parTop.row(k), parBot.row(k));
                    // truthSolve(par);
                    ITHACAstream::writePoints(mesh.points(), Folder, name(k + 1) + "/polyMesh/");
                    ITHACAstream::writePoints(mesh.points(), "defGeom", name(k + 1) + "/constant/polyMesh/");
                    ITHACAstream::exportSolution(U, name(k + 1), Folder);
                    ITHACAstream::exportSolution(p, name(k + 1), Folder);
                    //restart();
                }
            }
        };

        void restart()
        {
            fvMesh& mesh = _mesh();
            volScalarField& p = _p();
            volVectorField& U = _U();
            surfaceScalarField& phi = _phi();
            p = _p0();
            U = _U0();
            phi = _phi0();
        }

        void updateMesh(double par)
        {
            fvMesh& mesh = _mesh();
            mesh.movePoints(point0);
            List<vector> wing0_cur = ITHACAutilities::rotatePoints(wing0, axis, par);
            ITHACAutilities::setIndices2Value(wing0_ind, wing0_cur, movingIDs, curX);
            ms->setMotion(curX - x0);
            mesh.movePoints(ms->curPoints());
            mesh.moving(false);
        }

        void updateMesh2(Eigen::MatrixXd parTop, Eigen::MatrixXd parBot)
        {
            fvMesh& mesh = _mesh();
            mesh.movePoints(point0);
            List<vector> top0_cur = moveBasis(top0, parTop);
            List<vector> bot0_cur = moveBasis(bot0, parBot);
            ITHACAutilities::setIndices2Value(top0_ind, top0_cur, movingIDs, curX);
            ITHACAutilities::setIndices2Value(bot0_ind, bot0_cur, movingIDs, curX);
            ms->setMotion(curX - x0);
            mesh.movePoints(ms->curPoints());
            mesh.moving(false);
        }
};


int main(int argc, char* argv[])
{
    NS_geom_par example(argc, argv);
    example.startTime = 0;
    example.finalTime = 10;
    example.timeStep = 0.001;
    example.writeEvery = 0.1;
    Eigen::MatrixXd parTop = ITHACAutilities::rand(1000, 5, 0, 0.03);
    Eigen::MatrixXd parBot = ITHACAutilities::rand(1000, 5, -0.03, 0);
    cnpy::save(parTop,"parTop.npy");
    cnpy::save(parBot,"parBot.npy");
    example.OfflineSolve(parTop, parBot, "./ITHACAoutput/Offline/");
    return 0;
}
