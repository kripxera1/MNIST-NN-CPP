#ifndef ALGEBRA_H
#define ALGEBRA_H

#include <vector>
using namespace std;


vector<double> rowsSum(const vector<vector<double>> & M);




vector<double> product(vector<double> v, double a);




vector<vector<double>> product(vector<vector<double>> M, double a);




vector<vector<double>> T(const vector<vector<double>>&M);




vector<vector<double>> dot(const vector<vector<double>> & M1,
                           const vector<vector<double>> & M2);



vector<vector<double>> sum(const vector<vector<double>> & M,
                           const vector<double> & b);



vector<vector<double>> minusM(const vector<vector<double>> & M1,
                              const vector<vector<double>> & M2);



vector<vector<double>> hadamard(const vector<vector<double>> & M1,
                                const vector<vector<double>> & M2);



#endif