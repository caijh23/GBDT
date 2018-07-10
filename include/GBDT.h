#ifndef __GBDT_
#define __GBDT_

typedef struct _treeNode_
{
    int tn_feature; // the decision feature
    float tn_weight; // the weight of leaf or the split point of feature
    struct _treeNode_* tn_smallAndEqual; // tree node <= tn_weight
    struct _treeNode_* tn_large; // tree node > tn_weight
    int* tn_samples;
    int tn_sampleNum;
} treeNode;


class GBDT
{
public:
    GBDT();
    ~GBDT();
    void initModel();
    void setParameters(int boostNum, int maxDepth, float lambda, float gamma, float rate, float min_gain);
    void train();
    float* predict();
private:
    float predictFromOneTree (treeNode* node, float* inputItem);
    void pruneTree (treeNode* node);
    bool SplitOneNodeByAllFeature (treeNode* node, int round);
    void SplitOneNodeByFeature (treeNode* node, int feature, int round, float* maxGain, float* bestSplitPoint);
    void initalOneTree(treeNode* node);
    float getGj(treeNode* node, int round);
    float getHj(treeNode* node, int round);
    float getGi(int index, int round);
    float getHi(int index, int round);
    void trainSingleTree(treeNode* node, int round, int depth);
    

    treeNode** gbdt_trees; //[round][boostingStep]
    float** gbdt_prediction; //[round][predictSampleNum]
    float** gbdt_train; //[round][trainSampleNum]
    int* gbdt_label;

    int gbdt_round;
    int gbdt_maxBoostingNum;
    int gbdt_maxDepth;
    int gbdt_feature_num;
    int gbdt_train_num;
    int gbdt_predict_num;
    float gbdt_gamma;
    float gbdt_lambda;
    float gbdt_learningRate;
    float gbdt_min_split_gain;
};

#endif
