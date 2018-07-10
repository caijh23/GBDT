#include "Data.h"
#include "GBDT.h"
#include <iostream>
#include <ctime>

using namespace std;

int main()
{
    clock_t startTime,endTime;
    startTime = clock();
    cout << "load train data ..." << endl;
    Data::getInstance()->loadTrainData();
    cout << "load train data completed" << endl;
    GBDT gbdt;
    gbdt.setParameters(10,10,0.01,0.01,0.08,0.05);
    gbdt.initModel();
    gbdt.train();
    cout << "load predict data ..." << endl;
    Data::getInstance()->loadPredictData();
    cout << "load predict data completed" << endl;
    float* prediction = gbdt.predict();
    Data::getInstance()->savePrediction(prediction,"../data/predictionV1.txt");
    endTime = clock();
    cout << "Totle Time : " <<(double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
    return 0;
}