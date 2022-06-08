#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <iterator>
#include <cmath>
#include <iomanip>
#include "bitmap.h"

using namespace std;

void initWeightsBias(vector<vector<double>> & W, vector<double> & b){
    
    for(int i = 0; i < W.size();i++){
        b[i]=((double)rand()/RAND_MAX)-0.5;
        for(int j = 0; j < W[0].size(); j++)
            W[i][j]=((double)rand()/RAND_MAX)-0.5;
        }
}

vector<vector<int>> loadData(const char* fileName){

    ifstream file(fileName,ios::in);
    string line;
    vector<vector<int>> data;
    while ( getline( file, line ) ) {
        istringstream is( line );
        data.push_back(vector<int>(istream_iterator<int>(is)
        ,istream_iterator<int>()));
    }
    return data;
}


void softMax(vector<double> & input){

	double m, Z, constant;

	m = -INFINITY;
	for (int i = 0; i < input.size(); i++)
		if (m < input[i]) 
			m = input[i];
		
	Z = 0.0;
	for (int i = 0; i < input.size(); i++) 
		Z += exp(input[i] - m);

	constant = m + log(Z);
	for (int i = 0; i < input.size(); i++) 
		input[i] = exp(input[i] - constant);
}


vector<vector<double>> softMax(vector<vector<double>> & input){

    vector<vector<double>> A(input.size(),vector<double>(input[0].size()));
    vector<double> aux(input.size(),0);
    auto aux2 = aux;
    for(int j = 0; j < input[0].size(); j++){
        for(int i = 0; i < input.size(); i++)
            aux[i]=input[i][j];
        softMax(aux);
        for(int i = 0; i < aux.size(); i++)
            A[i][j] = aux[i];
    }

    return A;
}


vector<vector<double>> leakyRelu(const vector<vector<double>> & z){

    vector<vector<double>> A(z.size(),vector<double>(z[0].size()));
    for(int i = 0; i < z.size(); i++)
        for(int j = 0; j < z[0].size(); j++)
            A[i][j] = z[i][j] >= 0 ? z[i][j]:(0.001*z[i][j]);

    return A;
}


vector<vector<double>> leakyReluDeriv(const vector<vector<double>> & z){

    vector<vector<double>> A(z.size(),vector<double>(z[0].size()));
    for(int i = 0; i < z.size(); i++)
        for(int j = 0; j < z[0].size(); j++)
            A[i][j] = z[i][j] >= 0 ? 1 : 0.001;

    return A;
}

//Pre: number of columns of first matrix is equal to the number of rows of the second matrix
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

//devuelve un vector con la suma de cada una de las filas de la matriz proporcionada
vector<double> sumaFilas(const vector<vector<double>> & m){

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


double precision(vector<vector<double>>A, vector<vector<double>>Y){

    double correctos = 0;
    for(int j = 0; j < A[0].size(); j++){
        int maxInt;
        double max = -INFINITY;
        for(int i = 0; i < A.size(); i++)
            if(A[i][j]>max){
                max = A[i][j];
                maxInt = i;
            }
        if(Y[maxInt][j]==1)
            correctos+=1;
    }

    return correctos/Y[0].size();
}


vector<double> getPrediction(const vector<vector<double>> & A){

    vector<double> v;
    for(int j = 0; j < A[0].size(); j++){
        int maxInt;
        double max = -INFINITY;
        for(int i = 0; i < A.size(); i++)
            if(A[i][j]>max){
                max = A[i][j];
                maxInt = i;                
            }
        v.push_back(maxInt);
    }

    return v;
}


void updateWeightsBias(vector<vector<double>> & W,
                   const vector<vector<double>> & dW,
                   vector<double> &b,
                   const vector<double> &db,
                   double learnRate){

    for(int i = 0; i < b.size(); i++)
        b[i]-= db[i]*learnRate;
    for(int i = 0; i < W.size(); i++)
        for(int j = 0; j < W[0].size(); j++)
            W[i][j]-=dW[i][j]*learnRate;
}


vector<vector<double>> oneHotEncoding(const vector<vector<int>> & data, int batchSize, int it){

    vector<vector<double>> oneHot(10,vector<double>(batchSize,0));
    vector<double> labels(batchSize,0);

    for(int i = batchSize*it, k = 0; i < (it+1)*batchSize;k++, i++)
        labels[k] = data[i][0];
    for(int i = 0; i < labels.size();i++)
        oneHot[labels[i]][i] = 1;
    
    return oneHot;
}

vector<vector<double>> loadBatch(const vector<vector<int>> & data,
                                 int batchSize,
                                 int it){

    vector<vector<double>> A(batchSize,vector<double>(data[0].size()-1,0));
    for(int i = batchSize*it,k=0; i < batchSize*(it+1); i++,k++)
        for(int j = 1; j < data[0].size(); j++)
            A[k][j]=(double)data[i][j]/255;

    A=T(A);

    return A;
}


int main(){
  
    srand(time(NULL));

    //Data loading
    auto trainingData=loadData("mnist_train.txt");

    //Hyperparameters
    double learnRate = 0.005;
    int batchSize = 100;
    int nEpochs = 15;

    int trainSize = trainingData.size();
    int nBatch = trainSize/batchSize;

    //Layer sizes
    int size0 = 784;
    int size1 = 20;
    int size2 = 15;
    int size3 = 10;

    vector<vector<double>> W1(size1,vector<double>(size0,0));
    vector<double> b1(size1,0);
    vector<vector<double>> W2(size2,vector<double>(size1,0));
    vector<double> b2(size2,0);
    vector<vector<double>> W3(size3,vector<double>(size2,0));
    vector<double> b3(size3,0);

    //Weights and Biases initialization
    initWeightsBias(W1,b1);
    initWeightsBias(W2,b2);
    initWeightsBias(W3,b3);

    
    for(int epoch = 0; epoch < nEpochs; epoch++){
        double precission = 0;
        for(int it = 0; it < nBatch; it++){

            //Labels and images are loaded and processed in batch
            auto A0=loadBatch(trainingData,batchSize,it);
            auto Y = oneHotEncoding(trainingData,batchSize,it);
            
            //pass forward
            auto Z1=sum(dot(W1,A0),b1);
            auto A1=leakyRelu(Z1);

            auto Z2=sum(dot(W2,A1),b2);
            auto A2=leakyRelu(Z2);

            auto Z3=sum(dot(W3,A2),b3);
            auto A3=softMax(Z3);

            //precission update
            precission += precision(A3,Y)/nBatch;

            //Backpropagation
            auto dZ3 = minusM(A3,Y);
            auto dW3 = product(dot(dZ3,T(A2)),1.0/batchSize);
            auto db3 = product(sumaFilas(dZ3),1.0/batchSize);

            auto dZ2 = hadamard(dot(T(W3),dZ3),(leakyReluDeriv(Z2)));
            auto dW2 = product(dot(dZ2,T(A1)),1.0/batchSize);
            auto db2 = product(sumaFilas(dZ2),1.0/batchSize);

            auto dZ1 = hadamard(dot(T(W2),dZ2),(leakyReluDeriv(Z1)));
            auto dW1 = product(dot(dZ1,T(A0)),1.0/batchSize);
            auto db1 = product(sumaFilas(dZ1),1.0/batchSize);

            //Weights and bias update
            updateWeightsBias(W1,dW1,b1,db1,learnRate);
            updateWeightsBias(W2,dW2,b2,db2,learnRate);
            updateWeightsBias(W3,dW3,b3,db3,learnRate);
        }

        cout << "Epoch " << to_string(epoch) << ": " << precission << endl;
    }


    //clasification test

    //Data loading    
    auto testData = loadData("mnist_test.txt");
    double testSize = testData.size();
    double nBatchTest = testSize/batchSize;

    double precission = 0;
    for(int it = 0; it < nBatchTest; it++){

        //Labels and images are loaded and processed in batch
        auto A0=loadBatch(testData,batchSize,it);
        auto Y = oneHotEncoding(testData,batchSize,it);
        
        //Pass forward
        auto Z1=sum(dot(W1,A0),b1);
        auto A1=leakyRelu(Z1);

        auto Z2=sum(dot(W2,A1),b2);
        auto A2=leakyRelu(Z2);

        auto Z3=sum(dot(W3,A2),b3);
        auto A3=softMax(Z3);

        //Precission update
        precission += precision(A3,Y)/nBatchTest;
    }

    cout << "Test: " << precission << endl;

    return 0;
}
