#ifndef __DATA_
#define __DATA_

#include <string>

using namespace std;

typedef struct _sampleItem_
{
    int label;
    float* feature_value;
} sampleItem;

class Data
{
public:
    static Data* getInstance();
    bool loadTrainData();
    void getLabelColumn(int* output);
    float* getFeatureColumn(int feature);
    float* getPredictFeatureByIndex(int index);
    int getLabelBySampleIndex(int index);
    void getFeatureByFeatureIndex(float* output, int* input, int sampleNum, int featureIndex);
    int getFeatureNum();
    int getTrainNum();
    int getPredictNum();
    bool loadPredictData();
    bool loadTrainDataByColumn();
    bool savePrediction(float* prediction, string path);
private:
    Data();
    ~Data();
    sampleItem* trainSample;
    sampleItem* predictSample;
    float** trainMat; //[featureNum][trainNum]
    int* label;

    static string trainDataPath;
    static string predictDataPath;
    static int featureNum;
    static Data* instance;
    static int trainNum;
    static int predictNum;
};

#endif