#include "GBDT.h"
#include "Data.h"
#include <iostream>
#include <algorithm>
#include <cstring>

using namespace std;

GBDT::GBDT()
{
    gbdt_round = 1;
    gbdt_feature_num = Data::getInstance()->getFeatureNum();
    gbdt_train_num = Data::getInstance()->getTrainNum();
    gbdt_predict_num = Data::getInstance()->getPredictNum();
}

GBDT::~GBDT()
{
    for (int i = 0;i < gbdt_round;i++)
    {
        for (int j = 0;j < gbdt_maxBoostingNum;j++)
        {
            pruneTree(&gbdt_trees[i][j]);
        }
        delete[] gbdt_trees[i];
        delete[] gbdt_train[i];
        delete[] gbdt_prediction[i];
    }
    delete[] gbdt_trees;
    delete[] gbdt_train;
    delete[] gbdt_prediction;
    cout << "GBDT destory" << endl;
}

void GBDT::initModel()
{
    gbdt_trees = new treeNode*[gbdt_round];
    for (int i = 0;i < gbdt_round;i++)
    {
        gbdt_trees[i] = new treeNode[gbdt_maxBoostingNum];
        for (int j = 0;j < gbdt_maxBoostingNum;j++)
        {
            gbdt_trees[i][j].tn_feature = -1;
            gbdt_trees[i][j].tn_large = NULL;
            gbdt_trees[i][j].tn_smallAndEqual = NULL;
            gbdt_trees[i][j].tn_weight = 0.0;
            gbdt_trees[i][j].tn_samples = NULL;
            gbdt_trees[i][j].tn_sampleNum = 0;
        }
    }
    gbdt_train = new float*[gbdt_round];
    for (int i = 0;i < gbdt_round;i++)
    {
        gbdt_train[i] = new float[gbdt_train_num];
        for (int j =  0;j < gbdt_train_num;j++)
        {
            gbdt_train[i][j] = 0.0;
        }
    }
    gbdt_prediction = new float*[gbdt_round];
    for (int i = 0;i < gbdt_round;i++)
    {
        gbdt_prediction[i] = new float[gbdt_predict_num];
        for (int j = 0;j < gbdt_predict_num;j++)
        {
            gbdt_prediction[i][j] = 0.0;
        }
    }
}

float GBDT::predictFromOneTree (treeNode* node, float* inputItem)
{
    if (node->tn_smallAndEqual == NULL && node->tn_large == NULL)
        return node->tn_weight;
    int feature_index = node->tn_feature;
    float splitPoint = node->tn_weight;
    float feature = inputItem[feature_index];
    if (feature <= splitPoint)
        return predictFromOneTree (node->tn_smallAndEqual, inputItem);
    return predictFromOneTree (node->tn_large, inputItem);
}

void GBDT::pruneTree (treeNode* node)
{
    if (node->tn_samples)
    {
        delete[] node->tn_samples;
    }
    node->tn_sampleNum = 0;
    
    if (node->tn_smallAndEqual)
        pruneTree(node->tn_smallAndEqual);
    if (node->tn_large)
        pruneTree(node->tn_large);
    delete node;
}

// the input node should have not null tn_sample
// and not 0 tn_sampleNum
void GBDT::SplitOneNodeByFeature (treeNode* node, int feature, int round, float* maxGain, float* bestSplitPoint)
{
    cout << "split node by one feature started" << endl;
    int sampleNum = node->tn_sampleNum;
    float* feature_value = new float[sampleNum];
    Data::getInstance()->getFeatureByFeatureIndex(feature_value,node->tn_samples,sampleNum,feature);
    float* feature_temp = new float[sampleNum];
    memcpy(feature_temp,feature_value,sampleNum * sizeof(float));
    sort(feature_value,feature_value + sampleNum);
    int length = unique(feature_value, feature_value + sampleNum) - feature_value;
    *maxGain = 0.0;
    *bestSplitPoint = 0.0;
    for (int i = 0;i < length;i++)
    {
        float tempSplitPoint = feature_value[i];
        treeNode* leftNode = new treeNode;
        treeNode* rightNode = new treeNode;
        leftNode->tn_smallAndEqual = NULL;
        rightNode->tn_smallAndEqual = NULL;
        leftNode->tn_large = NULL;
        rightNode->tn_large = NULL;
        leftNode->tn_sampleNum = 0;
        rightNode->tn_sampleNum = 0;
        leftNode->tn_samples = new int[sampleNum];
        rightNode->tn_samples = new int[sampleNum];
        for (int j = 0;j < sampleNum;j++)
        {
            if (feature_temp[j] <= tempSplitPoint)
            {
                leftNode->tn_samples[leftNode->tn_sampleNum] = node->tn_samples[j];
                leftNode->tn_sampleNum++;
            }
            else {
                rightNode->tn_samples[rightNode->tn_sampleNum] = node->tn_samples[j];
                rightNode->tn_sampleNum++;
            }
        }
        int* tempLeftSample = new int[leftNode->tn_sampleNum];
        memcpy(tempLeftSample,leftNode->tn_samples,leftNode->tn_sampleNum * sizeof(int));
        delete[] leftNode->tn_samples;
        leftNode->tn_samples = tempLeftSample;
        int* tempRightSample = new int[rightNode->tn_sampleNum];
        memcpy(tempRightSample,rightNode->tn_samples,rightNode->tn_sampleNum * sizeof(int));
        delete[] rightNode->tn_samples;
        rightNode->tn_samples = tempRightSample;
        float GL = getGj(leftNode,round);
        float GR = getGj(rightNode,round);
        float HL = getHj(leftNode,round);
        float HR = getHj(rightNode,round);
        float tempGain = (GL * GL) / (HL + gbdt_lambda) + (GR * GR) / (HR + gbdt_lambda) - ((GL + GR) * (GL + GR)) / (HL + HR + gbdt_lambda) - gbdt_gamma;
        if (tempGain > *maxGain)
        {
            *bestSplitPoint = tempSplitPoint;
            *maxGain = tempGain;
        }
        delete[] leftNode->tn_samples;
        delete[] rightNode->tn_samples;
        delete leftNode;
        delete rightNode;
    }
    cout << "split node by one feature completed" << endl;
    delete[] feature_value;
    delete[] feature_temp;
}

void GBDT::initalOneTree(treeNode* node)
{
    node->tn_sampleNum = gbdt_train_num;
    node->tn_samples = new int[node->tn_sampleNum];
    for (int i = 0;i < node->tn_sampleNum;i++)
    {
        node->tn_samples[i] = i;
    }
}

// the input node should have not null tn_samples
// and not 0 tn_sampleNum
float GBDT::getGj(treeNode* node, int round)
{
    float Gj = 0.0;
    for (int i = 0;i < node->tn_sampleNum;i++)
    {
        float yi_pre = gbdt_train[round][node->tn_samples[i]];
        int yi = Data::getInstance()->getLabelBySampleIndex(node->tn_samples[i]);
        float gi = 2 * (yi_pre - (float)yi);
        Gj += gi;
    }
    return Gj;
}

float GBDT::getHj(treeNode* node, int round)
{
    float Hj = 0.0;
    for (int i = 0;i < node->tn_sampleNum;i++)
    {
        float hi = 2.0;
        Hj += hi;
    }
    return Hj;
}

bool GBDT::SplitOneNodeByAllFeature (treeNode* node, int round)
{
    float bestSplitPoint = 0.0;
    float maxGain = 0.0;
    int bestSplitFeature = -1;
    for (int i = 0;i < gbdt_feature_num;i++)
    {
        float tempGain = 0.0;
        float tempSplitPoint = 0.0;
        SplitOneNodeByFeature(node,i,round,&tempGain,&tempSplitPoint);
        if (tempGain > maxGain)
        {
            maxGain = tempGain;
            bestSplitFeature = i;
            bestSplitPoint = tempSplitPoint;
        }
    }
    if (maxGain < gbdt_min_split_gain)
    {
        return false;
    }
    int sampleNum = node->tn_sampleNum;
    float* feature_value = new float[sampleNum];
    Data::getInstance()->getFeatureByFeatureIndex(feature_value,node->tn_samples,sampleNum,bestSplitFeature);
    node->tn_feature = bestSplitFeature;
    node->tn_weight = bestSplitPoint;
    treeNode* leftNode = new treeNode;
    treeNode* rightNode = new treeNode;
    leftNode->tn_feature = -1;
    rightNode->tn_feature = -1;
    leftNode->tn_weight = 0.0;
    rightNode->tn_weight = 0.0;
    leftNode->tn_smallAndEqual = NULL;
    rightNode->tn_smallAndEqual = NULL;
    leftNode->tn_large = NULL;
    rightNode->tn_large = NULL;
    leftNode->tn_samples = new int[node->tn_sampleNum];
    rightNode->tn_samples = new int[node->tn_sampleNum];
    leftNode->tn_sampleNum = 0;
    rightNode->tn_sampleNum = 0;
    for (int i = 0;i < sampleNum;i++)
    {
        if (feature_value[i] <= bestSplitPoint)
        {
            leftNode->tn_samples[leftNode->tn_sampleNum] = node->tn_samples[i];
            leftNode->tn_sampleNum++;
        }
        else {
            rightNode->tn_samples[rightNode->tn_sampleNum] = node->tn_samples[i];
            rightNode->tn_sampleNum++;
        }
    }
    int* tempLeftSample = new int[leftNode->tn_sampleNum];
    memcpy(tempLeftSample,leftNode->tn_samples,leftNode->tn_sampleNum * sizeof(int));
    delete[] leftNode->tn_samples;
    leftNode->tn_samples = tempLeftSample;
    int* tempRightSample = new int[rightNode->tn_sampleNum];
    memcpy(tempRightSample,rightNode->tn_samples,rightNode->tn_sampleNum * sizeof(int));
    delete[] rightNode->tn_samples;
    rightNode->tn_samples = tempRightSample;
    node->tn_smallAndEqual = leftNode;
    node->tn_large = rightNode;
    delete[] feature_value;
    cout << "Spilt one node" << endl;
    return true;
}

// input a tree root with depth 0
// the input node should has not null tn_sample
// and not 0 tn_sampleNum,the tn_feature will be set
// in the train process,use dfs to train until the
// split gain is smaller than gbdt_min_split_gain
void GBDT::trainSingleTree(treeNode* node, int round, int depth)
{
    if (SplitOneNodeByAllFeature(node,round) && depth < gbdt_maxDepth)
    {
        trainSingleTree(node->tn_smallAndEqual,round,depth + 1);
        trainSingleTree(node->tn_large,round,depth + 1);
    }
    else {
        node->tn_weight = - getGj(node, round) / (getHj(node,round) + gbdt_lambda);
        cout << "train one tree" << endl;
        for (int i = 0;i < node->tn_sampleNum;i++)
        {
            // boosting
            gbdt_train[round][node->tn_samples[i]] += gbdt_learningRate * node->tn_weight;
        }
    }
}

void GBDT::setParameters(int boostNum, int maxDepth, float lambda, float gamma, float rate, float min_gain)
{
    gbdt_maxBoostingNum = boostNum;
    gbdt_maxDepth = maxDepth;
    gbdt_lambda = lambda;
    gbdt_gamma = gamma;
    gbdt_learningRate = rate;
    gbdt_min_split_gain = min_gain;
}

void GBDT::train()
{
    for (int i = 0;i < gbdt_round;i++)
    {
        for (int j = 0;j < gbdt_maxBoostingNum;j++)
        {
            initalOneTree(&gbdt_trees[i][j]);
            trainSingleTree(&gbdt_trees[i][j],i,0);
        }
    }
}

float* GBDT::predict()
{
    for (int i = 0;i < gbdt_round;i++)
    {
        for (int j = 0;j < gbdt_predict_num;j++)
        {
            for (int k = 0;k < gbdt_maxBoostingNum;k++)
            {
                gbdt_prediction[i][j] += predictFromOneTree(&gbdt_trees[i][k],Data::getInstance()->getPredictFeatureByIndex(j));
            }
        }
    }
    return gbdt_prediction[0];
}