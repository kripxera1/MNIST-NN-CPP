#include "algebra.h"

//Pre: number of columns of firs equals number of rows of second
vector<vector<double>> dot(const vector<vector<double>> & M1,
                           const vector<vector<double>> & M2){

    vector<vector<double>>M3(M1.size(),vector<double>(M2[0].size(),0));
    for(int i = 0; i < M1.size(); i++)
        for(int j = 0; j < M2[0].size(); j++)
            for(int k = 0; k < M2.size(); k++)
                M3[i][j] += M1[i][k]*M2[k][j];

    return M3;
}

//sums vector b with each column of the matrix M
//Pre: M.size() == b.size();
vector<vector<double>> sum(const vector<vector<double>> & M,
                           const vector<double> & b){

    vector<vector<double>> M2(M.size(),vector<double>(M[0].size(),0));
    for(int i = 0; i < M.size(); i++)
        for(int j = 0; j < M[0].size(); j++)
           M2[i][j]=b[i]+M[i][j];

    return M2;
}


vector<vector<double>> T(const vector<vector<double>>&m){

    vector<vector<double>> mt(m[0].size(),vector<double>(m.size(),0));
    for(int i = 0; i < m.size(); i++)
        for(int j = 0; j < m[0].size(); j++)
            mt[j][i] = m[i][j];

    return mt;
}


vector<vector<double>> minusM(const vector<vector<double>> & m1,
                             const vector<vector<double>> & m2){

    vector<vector<double>> m(m1.size(),vector<double>(m1[0].size(),0));
    for(int i = 0; i < m1.size(); i++)
        for(int j = 0; j < m1[0].size(); j++)
            m[i][j]=m1[i][j]-m2[i][j];

    return m;
}


vector<vector<double>> product(vector<vector<double>> m1, double a){

    for(int i = 0; i < m1.size(); i++)
        for(int j = 0; j < m1[0].size(); j++)
            m1[i][j]*=a;

    return m1;
}


vector<double> product(vector<double> v, double a){

    for(int i = 0; i<v.size(); i++)
        v[i]*=a;

    return v;
}

//devuelve un vector con la suma de cada una de las filas
vector<double> rowsSum(const vector<vector<double>> & m){

    vector<double> v (m.size(), 0);
    for(int i = 0; i < m.size(); i++)
        for(int j = 0; j< m[i].size(); j++)
            v[i]+=m[i][j];
    
    return v;
}


vector<vector<double>> hadamard(const vector<vector<double>> & m1,
                                const vector<vector<double>> & m2){

    vector<vector<double>> m(m1.size(),vector<double>(m1[0].size(),0));
    for(int i = 0; i < m1.size(); i++)
        for(int j = 0; j < m1[0].size(); j++)
            m[i][j]=m1[i][j]*m2[i][j];

    return m;
}