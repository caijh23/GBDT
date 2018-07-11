#include "Data.h"
#include <fstream>
#include <iostream>
#include <sstream>

Data* Data::instance = NULL;

string Data::trainDataPath = "../data/train.txt";
string Data::predictDataPath = "../data/test.txt";
int Data::featureNum = 201;
int Data::trainNum = 1719692;
int Data::predictNum = 429923;

Data::Data()
{
    predictSample = NULL;
    label = NULL;
    trainMat = NULL;
}

Data::~Data()
{
    if (predictSample)
    {
        for (int i = 0;i < predictNum;i++)
        {
            delete[] (predictSample[i].feature_value);
        }
        delete[] predictSample;
    }
    if (instance)
    {
        delete instance;
    }
    if (trainMat)
    {
        for (int i = 0;i < featureNum;i++)
        {
            delete[] trainMat[i];
        }
        delete[] trainMat;
    }
    if (label)
    {
        delete label;
    }
}

Data* Data::getInstance()
{
    if (instance == NULL)
    {
        instance = new Data();
    }
    return instance;
}

bool Data::loadPredictData()
{
    predictSample = new sampleItem[predictNum];

    ifstream inFile;
    inFile.open(predictDataPath.c_str());
    if (!inFile)
    {
        cout << "open file failed!" << endl;
        cout << "path is " << predictDataPath << endl;
        return false;
    }
    int index = 0;
    while (!inFile.eof())
    {
        predictSample[index].feature_value = new float[featureNum];
        string line;
        getline(inFile,line);
        stringstream ss;
        int sampleIndex;
        ss << line;
        ss >> sampleIndex;
        cout << index << " : " << sampleIndex << endl;
        if (sampleIndex != index)
        {
            // cout << "error index "  << index  << " " << sampleIndex << endl;
            break;
        }
        int currentIndex = 0;
        while (!ss.eof())
        {
            string strIndex;
            string strFeature;
            string temp;
            ss >> temp;
            if (ss.eof())
                break;
            stringstream ss_temp(temp);
            getline(ss_temp, strIndex, ':');
            getline(ss_temp, strFeature);
            int tempIndex = std::stoi(strIndex) - 1;
            float tempFeature = std::stof(strFeature);
            while (currentIndex < tempIndex)
            {
                predictSample[index].feature_value[currentIndex] = 0.0;
                currentIndex++;
            }
            predictSample[index].feature_value[currentIndex] = tempFeature;
            currentIndex++;
        }
        while (currentIndex < featureNum)
        {
            predictSample[index].feature_value[currentIndex] = 0.0;
            currentIndex++;
        }
        index++;
    }
    inFile.close();
    return true;
}

bool Data::loadTrainDataByColumn()
{
    trainMat = new float*[featureNum];
    for (int i = 0;i < featureNum;i++)
    {
        trainMat[i] = new float[trainNum];
        for (int j = 0;j < trainNum;j++)
            trainMat[i][j] = 0.0;
    }
    label = new int[trainNum];
    
    ifstream inFile;
    inFile.open(trainDataPath.c_str());
    if (!inFile)
    {
        cout << "open file failed!" << endl;
        cout << "path is " << trainDataPath << endl;
        return false;
    }
    int index = 0;
    while (!inFile.eof())
    {
        string line;
        getline(inFile,line);
        stringstream ss;
        ss << line;
        ss >> label[index];
        int currentIndex = 0;
        while (!ss.eof())
        {
            string strIndex;
            string strFeature;
            string temp;
            ss >> temp;
            if (ss.eof())
                break;
            stringstream ss_temp(temp);
            getline(ss_temp, strIndex, ':');
            getline(ss_temp, strFeature);
            int tempIndex = std::stoi(strIndex) - 1;
            float tempFeature = std::stof(strFeature);
            while (currentIndex < tempIndex)
                currentIndex++;
            trainMat[currentIndex][index] = tempFeature;
            currentIndex++;
        }
        index++;
    }
    inFile.close();
    return true;
}

/*bool Data::loadTrainData()
{
    trainSample = new sampleItem[trainNum];

    ifstream inFile;
    inFile.open(trainDataPath.c_str());
    if (!inFile)
    {
        cout << "open file failed!" << endl;
        cout << "path is " << trainDataPath << endl;
        return false;
    }
    int index = 0;
    while(!inFile.eof()) {
        trainSample[index].feature_value = new float[featureNum];
        string line;
        getline(inFile,line);
        stringstream ss;
        ss << line;
        ss >> trainSample[index].label;
        int currentIndex = 0;
        while (!ss.eof())
        {
            string strIndex;
            string strFeature;
            string temp;
            ss >> temp;
            if (ss.eof())
                break;
            stringstream ss_temp(temp);
            getline(ss_temp, strIndex, ':');
            getline(ss_temp, strFeature);
            int tempIndex = std::stoi(strIndex) - 1;
            float tempFeature = std::stof(strFeature);
            while (currentIndex < tempIndex)
            {
                trainSample[index].feature_value[currentIndex] = 0.0;
                currentIndex++;
            }
            trainSample[index].feature_value[currentIndex] = tempFeature;
            currentIndex++;
        }
        while (currentIndex < featureNum)
        {
            trainSample[index].feature_value[currentIndex] = 0.0;
            currentIndex++;
        }
        index++;
    }
    inFile.close();
    return true;
}*/

int Data::getFeatureNum()
{
    return featureNum;
}

int Data::getTrainNum()
{
    return trainNum;
}

int Data::getPredictNum()
{
    return predictNum;
}

bool Data::savePrediction(float* prediction, string path)
{
    ofstream outFile;
    outFile.open(path.c_str());
    if (!outFile)
    {
        cout << "open out file failed" << endl;
        return false;
    }
    outFile << "id,label\n";
    for (int i = 0;i < predictNum;i++)
    {
        outFile << i << "," << prediction[i] << "\n";
    }
    outFile.close();
    return true;
}

float* Data::getPredictFeatureByIndex(int index)
{
    return predictSample[index].feature_value;
}

float* Data::getFeatureColumn(int feature)
{
    return trainMat[feature];
}

int* Data::getLabelColumn()
{
    return label;
}
