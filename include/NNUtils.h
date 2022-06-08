#ifndef NNUTILS_H
#define NNUTILS_H

#include <fstream>
#include <sstream>
#include <iterator>
#include <vector>
#include <cmath>

#include "algebra.h"

using namespace std;

//Activation functions
//////////////////////////////////////////////////////////////////////////////

void softMax(vector<double> & input);




vector<vector<double>> softMax(vector<vector<double>> & input);




vector<vector<double>> leakyRelu(const vector<vector<double>> & z);




vector<vector<double>> leakyReluDeriv(const vector<vector<double>> & z);




void initWeightsBias(vector<vector<double>> & W, vector<double> & b);



void updateWeightsBias(vector<vector<double>> & W,
                       const vector<vector<double>> & dW, vector<double> &b,
                       const vector<double> &db, double learnRate);


vector<vector<int>> loadData(const char* fileName);




vector<vector<double>> loadBatch(const vector<vector<int>> & data,
                                 int batchSize, int it);



vector<vector<double>> oneHotEncoding(const vector<vector<int>> & data,
                                      int batchSize, int it);



vector<double> getPrediction(const vector<vector<double>> & A);



double accuracy(vector<vector<double>>A, vector<vector<double>>Y);

#endif