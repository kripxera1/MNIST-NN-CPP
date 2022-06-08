#include <iostream>
#include "bitmap.h"
#include "algebra.h"
#include "NNUtils.h"

using namespace std;


int main(){
  
    srand(time(NULL));

    //Hyperparameters
    double learnRate = 0.005;
    int batchSize = 100;
    int nEpochs = 20;

    cout << "Hyperparameters:\n"
         << "\n\tLearning rate:\t\t" << learnRate
         << "\n\tBatch size:\t\t" << batchSize
         << "\n\tNumber of epochs:\t" << nEpochs
         << "\n" << endl;

    //Data loading
    cout << "\nLoading training set..." << endl;
    auto trainingData=loadData("mnist_train.txt");
    cout << "\nDone" << endl;

    int trainSize = trainingData.size();
    int nBatch = trainSize/batchSize;


    //Layer sizes
    int size0 = 784;
    int size1 = 25;
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

    cout << "\n\nTraining accuracy:\n" << endl;
    for(int epoch = 0; epoch < nEpochs; epoch++){
        double accuracy_ = 0;
        for(int it = 0; it < nBatch; it++){

            //Labels and images are loaded and processed in batch
            auto A0 = loadBatch(trainingData,batchSize,it);
            auto Y = oneHotEncoding(trainingData,batchSize,it);
            
            //pass forward
            auto Z1 = sum(dot(W1,A0),b1);
            auto A1 = leakyRelu(Z1);
            auto Z2 = sum(dot(W2,A1),b2);
            auto A2 = leakyRelu(Z2);
            auto Z3 = sum(dot(W3,A2),b3);
            auto A3 = softMax(Z3);

            //Accuracy update
            accuracy_ += accuracy(A3,Y)/nBatch;

            //Backpropagation
            auto dZ3 = minusM(A3,Y);
            auto dW3 = product(dot(dZ3,T(A2)),1.0/batchSize);
            auto db3 = product(rowsSum(dZ3),1.0/batchSize);

            auto dZ2 = hadamard(dot(T(W3),dZ3),(leakyReluDeriv(Z2)));
            auto dW2 = product(dot(dZ2,T(A1)),1.0/batchSize);
            auto db2 = product(rowsSum(dZ2),1.0/batchSize);

            auto dZ1 = hadamard(dot(T(W2),dZ2),(leakyReluDeriv(Z1)));
            auto dW1 = product(dot(dZ1,T(A0)),1.0/batchSize);
            auto db1 = product(rowsSum(dZ1),1.0/batchSize);

            //Weights and bias update
            updateWeightsBias(W1,dW1,b1,db1,learnRate);
            updateWeightsBias(W2,dW2,b2,db2,learnRate);
            updateWeightsBias(W3,dW3,b3,db3,learnRate);
        }

        cout << "\tEpoch " << to_string(epoch) << ":\t" << accuracy_ << endl;
    }


    //Test

    //Data loading
    cout << "\n\nLoading test set..." << endl;
    auto testData = loadData("mnist_test.txt");
    cout << "\nDone" << endl;

    double testSize = testData.size();
    double nBatchTest = testSize/batchSize;


    cout << "\n\nTest accuracy:\n" << endl;
    double accuracy_ = 0;
    for(int it = 0; it < nBatchTest; it++){

        //Labels and images are loaded and processed in batch
        auto A0 = loadBatch(testData,batchSize,it);
        auto Y = oneHotEncoding(testData,batchSize,it);
        
        //Pass forward
        auto Z1 = sum(dot(W1,A0),b1);
        auto A1 = leakyRelu(Z1);
        auto Z2 = sum(dot(W2,A1),b2);
        auto A2 = leakyRelu(Z2);
        auto Z3 = sum(dot(W3,A2),b3);
        auto A3 = softMax(Z3);

        //Accuracy update
        accuracy_ += accuracy(A3,Y)/nBatchTest;
    }

    cout << "\tTest:\t" << accuracy_ << endl;

    //Image classification example

    //number of images to classify
    int nImages = 100;
    int index = 10;

    int height = 28,
        width  = 28;

    //Image memory reserve
    auto image = reserveSpaceImage(height,width);

    auto A0 = loadBatch(testData,nImages,index);
    auto Y = oneHotEncoding(testData,nImages,index);
    
    //Pass forward
    auto Z1 = sum(dot(W1,A0),b1);
    auto A1 = leakyRelu(Z1);
    auto Z2 = sum(dot(W2,A1),b2);
    auto A2 = leakyRelu(Z2);
    auto Z3 = sum(dot(W3,A2),b3);
    auto A3 = softMax(Z3);

    auto prediction = getPrediction(A3);
    
    for(int images = 0; images < nImages; images++){
        auto it = testData[(index*nImages)+images].begin();
        for(int i = height-1; i >= 0; i--){
            for(int j = 0; j < width; j++){
                it++;
                image[i][j][0] = (unsigned char)*it;
                image[i][j][1] = (unsigned char)*it;
                image[i][j][2] = (unsigned char)*it;
            }
        }
        generateBitmapImage(image,height,width,(char*)
        ((string(to_string(images))+"th image is "+
        string(to_string(prediction[images]))+
        ".bmp").c_str()));
    }

    
    return 0;
}
