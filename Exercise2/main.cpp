#include <iostream>
#include "Eigen/Eigen"
#include <iomanip>

using namespace std;
using namespace Eigen;






void SolvePALU(const MatrixXd& A,
               const VectorXd& b,
               const VectorXd& solution,
               Vector2d& solutionPALU,
               double& errRelPALU);

void SolveQR(const MatrixXd& A,
             const VectorXd& b,
             const VectorXd& solution,
             Vector2d& solutionQR,
             double& errRelQR);

bool CheckSV(const MatrixXd& A);






int main()
{
    Vector2d x(-1.0e+0, -1.0e+00);


    Matrix2d A1{{5.547001962252291e-01, -3.770900990025203e-02}, { 8.320502943378437e-01,
                                                                  -9.992887623566787e-01}};
    Vector2d b1(-5.169911863249772e-01, 1.672384680188350e-01);


    Matrix2d A2{{5.547001962252291e-01, -5.540607316466765e-01}, { 8.320502943378437e-01,
                                                                  -8.324762492991313e-01}};
    Vector2d b2(-6.394645785530173e-04, 4.259549612877223e-04);


    Matrix2d A3{{5.547001962252291e-01, -5.547001955851905e-01}, { 8.320502943378437e-01,
                                                                  -8.320502947645361e-01}};
    Vector2d b3(-6.400391328043042e-10, 4.266924591433963e-10);





    if (!CheckSV(A1))
    {
        cerr << "System 1 is unsolvable." << endl;
    }
    else
    {
        Vector2d solPALU1;
        Vector2d solQR1;
        double erPALU1 = 0;
        double erQR1 = 0;

        SolvePALU(A1, b1, x, solPALU1, erPALU1);
        SolveQR(A1, b1, x, solQR1, erQR1);

        cout << scientific << setprecision(16) <<
            "System 1: \n" <<
            "Solution with PALU decomposition:   " << solPALU1.transpose() << "\n" <<
            "Solution with QR decomposition:   " << solQR1.transpose() << "\n" <<
            "Relative error with PALU decomposition:   " << erPALU1 << "\n" <<
            "Relative error with QR decomposition:   " << erQR1 << endl;

    }



    if (!CheckSV(A2))
    {
        cerr << "System 2 is unsolvable." << endl;

    }
    else
    {   Vector2d solPALU2;
        Vector2d solQR2;
        double erPALU2 = 0;
        double erQR2 = 0;

        SolvePALU(A2, b2, x, solPALU2, erPALU2);
        SolveQR(A2, b2, x, solQR2, erQR2);

        cout << scientific << setprecision(16) <<
            "System 2: \n" <<
            "Solution with PALU decomposition:   " << solPALU2.transpose() << "\n" <<
            "Solution with QR decomposition:   " << solQR2.transpose() << "\n" <<
            "Relative error with PALU decomposition:   " << erPALU2 << "\n" <<
            "Relative error with QR decomposition:   " << erQR2 << endl;
    }



    if (!CheckSV(A3))
    {
        cerr << "System 3 is unsolvable." << endl;

    }
    else
    {   Vector2d solPALU3;
        Vector2d solQR3;
        double erPALU3 = 0;
        double erQR3 = 0;

        SolvePALU(A3, b3, x, solPALU3, erPALU3);
        SolveQR(A3, b3, x, solQR3, erQR3);

        cout << scientific << setprecision(16) <<
            "System 3: \n" <<
            "Solution with PALU decomposition:   " << solPALU3.transpose() << "\n" <<
            "Solution with QR decomposition:   " << solQR3.transpose() << "\n" <<
            "Relative error with PALU decomposition:   " << erPALU3 << "\n" <<
            "Relative error with QR decomposition:   " << erQR3 << endl;
    }



    return 0;
}




void SolvePALU(const MatrixXd& A,
                   const VectorXd& b,
                   const VectorXd& solution,
                   Vector2d& solutionPALU,
                   double& errRelPALU
                   )
{
    solutionPALU = A.fullPivLu().solve(b);
    errRelPALU = (solutionPALU-solution).norm()/solution.norm();

}



void SolveQR(const MatrixXd& A,
               const VectorXd& b,
               const VectorXd& solution,
               Vector2d& solutionQR,
               double& errRelQR
               )
{
    solutionQR = A.colPivHouseholderQr().solve(b);
    errRelQR = (solutionQR-solution).norm()/solution.norm();

}



bool CheckSV(const MatrixXd& A)
{
    JacobiSVD<MatrixXd> svd(A);
    VectorXd singularValuesA = svd.singularValues();

    if( singularValuesA.minCoeff() < 1e-16)
    {

        return false;
    }

    return true;
}


