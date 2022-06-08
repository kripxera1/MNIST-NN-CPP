#include "NNUtils.h"

//Activation functions
//////////////////////////////////////////////////////////////////////////////

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


void initWeightsBias(vector<vector<double>> & W, vector<double> & b){
    
    for(int i = 0; i < W.size();i++){
        b[i]=((double)rand()/RAND_MAX)-0.5;
        for(int j = 0; j < W[0].size(); j++)
            W[i][j]=((double)rand()/RAND_MAX)-0.5;
        }
}


void updateWeightsBias(vector<vector<double>> & W,
                   const vector<vector<double>> & dW, vector<double> &b,
                   const vector<double> &db, double learnRate){

    for(int i = 0; i < b.size(); i++)
        b[i]-= db[i]*learnRate;
    for(int i = 0; i < W.size(); i++)
        for(int j = 0; j < W[0].size(); j++)
            W[i][j]-=dW[i][j]*learnRate;
}


vector<vector<int>> loadData(const char* fileName){

    ifstream file(fileName,ios::in);
    string line;
    vector<vector<int>> data;
    while (getline(file, line)) {
        istringstream is(line);
        data.push_back(vector<int>(istream_iterator<int>(is)
        ,istream_iterator<int>()));
    }
    return data;
}


vector<vector<double>> loadBatch(const vector<vector<int>> & data,
                                 int batchSize, int it){

    vector<vector<double>> A(batchSize,vector<double>(data[0].size()-1,0));
    for(int i = batchSize*it,k=0; i < batchSize*(it+1); i++,k++)
        for(int j = 1; j < data[0].size(); j++)
            A[k][j]=(double)data[i][j]/255;

    A=T(A);

    return A;
}


vector<vector<double>> oneHotEncoding(const vector<vector<int>> & data,
                                      int batchSize, int it){

    vector<vector<double>> oneHot(10,vector<double>(batchSize,0));
    vector<double> labels(batchSize,0);

    for(int i = batchSize*it, k = 0; i < (it+1)*batchSize;k++, i++)
        labels[k] = data[i][0];
    for(int i = 0; i < labels.size();i++)
        oneHot[labels[i]][i] = 1;
    
    return oneHot;
}


vector<int> getPrediction(const vector<vector<double>> & A){

    vector<int> v;
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


double accuracy(vector<vector<double>>A, vector<vector<double>>Y){

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