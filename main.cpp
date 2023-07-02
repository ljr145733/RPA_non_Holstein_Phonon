#include <cstdio>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <complex>
#include "parameters.h"
#include "omp.h"
#include <fstream>
#include <vector>
#include "mpi.h"

typedef std::complex<double> cplx;

struct ind
{
    double k1, k2;
    cplx iwn;
    ind(double k1_in, double k2_in, cplx iwn_in) : k1(k1_in), k2(k2_in), iwn(iwn_in) {}
    ind operator-() const
    {
        return ind(-k1, -k2, -iwn);
    }
    ind operator+(const ind &other) const
    {
        return ind(k1 + other.k1, k2 + other.k2, iwn + other.iwn);
    }
};

cplx g(const ind &k, const ind &q)
{
    //g0:1
    //g1:0.4
    //g2:-0.1
    //g3:0.1/sqrt(2)
    //g4:0.1/2

    /*
    double g0=1.2;
    double g1=g0*0.4;
    double g2=-0.1*g0;
    double g3=g2/sqrt(2);
    double g4=g2/2.0;
    return g0+g1*2*(cos(q.k1)+cos(q.k2))+g2*4*cos(q.k1)*cos(q.k2) +g3*2*(cos(2*q.k1)+cos(2*q.k2))+4*g4*cos(2*q.k1)*cos(2*q.k2);
    */

    //return sqrt(pow(sin(q.k1/2.0),2) + pow(sin(q.k2/2.0),2)); //breathing phonon
    //return 7.0 *(sin(k.k1 / 2) * sin(k.k1 / 2 - q.k1 / 2) * cos(q.k2 / 2.0) - sin(k.k2 / 2.0) * sin(k.k2 / 2.0 - q.k2 / 2.0) * cos(q.k1 / 2.0));
    //return 3.0 *(sin(k.k1 / 2) * sin(k.k1 / 2 - q.k1 / 2) * cos(q.k2 / 2.0) + sin(k.k2 / 2.0) * sin(k.k2 / 2.0 - q.k2 / 2.0) * cos(q.k1 / 2.0));
     
     return 3.0*(cos(k.k1)-cos(k.k2))*(cos(k.k1/2-q.k1/2)-cos(k.k2/2-q.k2/2));
}

cplx lambda(const ind &p, const ind &k, const ind &q)
{
    return 0.5 * omega_ph / (omega_ph * omega_ph - q.iwn * q.iwn) * g(p, -q) * g(k, q);
}

double eps(const ind &p)
{
    return -2 * t * (cos(p.k1) + cos(p.k2)) - 4 * tp * cos(p.k1) * cos(p.k2) - 2 * tpp * (cos(p.k1 * 2) + cos(p.k2 * 2)) - mu;
}

double fd(double enrg)
{
    return 1.0 / (exp(beta * enrg) + 1);
}

cplx chi0(const ind &p, const ind &q)
{
    // return -1.0/(p.iwn-eps(p))*1.0/(p.iwn+q.iwn-eps(p+q));
    return -(fd(eps(p)) - fd(eps(p + q))) / (eps(p) - eps(p + q) + q.iwn);
}

cplx P(const ind &p, const ind &q)
{
    cplx res = 0;
#pragma omp declare reduction(plus : std::complex<double> : omp_out += omp_in) initializer(omp_priv = std::complex<double>(0.0, 0.0))

#pragma omp parallel for reduction(plus : res)
    for (int i = 0; i < L * L; ++i)
    {
        ind k = ind(int(i / L) * 2.0 * M_PI / L, int(i % L) * 2.0 * M_PI / L, 0);
        res += chi0(k, q) * lambda(p, k, q);
    }
    return res / double(L * L);
}

cplx chi(const ind &p, const ind &q)
{
    return chi0(p, q) / (1.0 - P(p, q));
}

cplx chi_mat(double q1, double q2)
{
    cplx res = 0;
    ind q = ind(q1, q2, cplx(0, delta));
    for (int i = 0; i < L; ++i)
    {
        for (int j = 0; j < L; ++j)
        {
            ind p = ind(i * 2.0 * M_PI / L, j * 2.0 * M_PI / L, 0);
            res += chi(p, q);
        }
    }
    return res / double(L * L);
}

cplx chi0_mat(double q1, double q2)
{
    cplx res = 0;
    ind q = ind(q1, q2, cplx(0, delta));
    for (int i = 0; i < L; ++i)
    {
        for (int j = 0; j < L; ++j)
        {
            ind p = ind(i * 2.0 * M_PI / L, j * 2.0 * M_PI / L, 0);
            res += chi0(p, q);
        }
    }
    return res / double(L * L);
}

double n()
{
    double ntot = 0;
    for (int i = 0; i < L; ++i)
    {
        for (int j = 0; j < L; ++j)
        {
            ind tmp = ind(i * 2 * M_PI / L, j * 2 * M_PI / L, 0);
            double ee = eps(tmp);
            ntot += 2.0 / (exp(beta * ee) + 1);
        }
    }
    return ntot / (L * L);
}

double find_mu(double n_target = 1)
{
    double left = -20;
    double right = 20;

    while (true)
    {
        mu = (left + right) / 2.0;

        if (n() < n_target)
        {
            left = mu;
        }
        else
        {
            right = mu;
        }
        if (fabs(left - right) < 1e-6)
        {
            break;
        }
    }
    return left;
}

int main(int argc, char **argv)
{

    int procID;
    int procNum;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &procID);
    MPI_Comm_size(MPI_COMM_WORLD, &procNum);

    omp_set_num_threads(4);
    // L^6 m^2
    cplx mat[L / 2 + 1][L / 2 + 1];
    cplx mat0[L / 2 + 1][L / 2 + 1];

    cplx mat_ALL[L / 2 + 1][L / 2 + 1];
    cplx mat0_ALL[L / 2 + 1][L / 2 + 1];

    find_mu(0.5);
    if (procID == 0)
        std::cout << mu << ' ' << n() << std::endl;

    std::vector<std::pair<int,int> > plist;
    for (int i = 0; i <= L / 2; i++)
    {
        for (int j = 0; j <= i; j++)
        {
            plist.push_back(std::make_pair(i,j));
        }
    }

    int fr = plist.size() * procID / procNum;
    int to = plist.size() * (procID + 1) / procNum;

    for (int i = 0; i <= L / 2; i++)
    {
        for (int j = 0; j <= L / 2; j++)
        {
            mat[i][j] = 0;
            mat0[i][j] = 0;
            mat_ALL[i][j] = 0;
            mat0_ALL[i][j] = 0;
        }
    }

    for (int ind = fr; ind < to; ind++)
    {
        int i=plist[ind].first;
        int j=plist[ind].second;

        cplx tmp = chi_mat(i * 2 * M_PI / L, j * 2 * M_PI / L);
        cplx tmptmp = chi0_mat(i * 2 * M_PI / L, j * 2 * M_PI / L);

        mat[i][j] = tmp;
        mat[j][i] = tmp;

        mat0[i][j] = tmptmp;
        mat0[j][i] = tmptmp;
    }

    MPI_Reduce(&mat[0][0], &mat_ALL[0][0], (L / 2 + 1) * (L / 2 + 1), MPI_DOUBLE_COMPLEX, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&mat0[0][0], &mat0_ALL[0][0], (L / 2 + 1) * (L / 2 + 1), MPI_DOUBLE_COMPLEX, MPI_SUM, 0, MPI_COMM_WORLD);

    if (procID == 0)
    {

        std::ofstream fout("res.dat", std::ios::out);
        std::ofstream fout2("res0.dat", std::ios::out);
        for (int i = -L / 2; i <= L / 2; i++)
        {
            for (int j = -L / 2; j <= L / 2; j++)
            {
                int ii = i < 0 ? -i : i;
                int jj = j < 0 ? -j : j;
                fout << mat_ALL[ii][jj].real() << ' ';
                fout2 << mat0_ALL[ii][jj].real() << ' ';
            }
            fout << std::endl;
            fout2 << std::endl;
        }

        fout.close();
        fout2.close();
    }
    MPI_Finalize();
}