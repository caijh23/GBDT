#include "Data.h"
#include "GBDT.h"
#include <iostream>
#include <ctime>

using namespace std;

int main()
{
    struct timespec start, finish;
    clock_gettime(CLOCK_REALTIME, &start);
    cout << "load train data ..." << endl;
    Data::getInstance()->loadTrainDataByColumn();
    cout << "load train data completed" << endl;
    GBDT gbdt;
    gbdt.setParameters(5,5,0.01,0.01,0.08,0.05);
    gbdt.initModel();
    gbdt.train();
    cout << "load predict data ..." << endl;
    Data::getInstance()->loadPredictData();
    cout << "load predict data completed" << endl;
    float* prediction = gbdt.predict();
    Data::getInstance()->savePrediction(prediction,"../data/predictionV2.txt");
    clock_gettime(CLOCK_REALTIME, &finish);
    cout << "Totle Time : " << (finish.tv_sec + 1.e-9 * finish.tv_nsec) - (start.tv_sec + 1.e-9 * start.tv_nsec) << "s" << endl;
    return 0;
}