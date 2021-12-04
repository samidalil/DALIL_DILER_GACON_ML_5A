
struct MyMLP {

};


extern "C" {
__declspec(dllexport) int return42() {
    return 51;
}


__declspec(dllexport) MyMLP *createMlpModel(int npl[], int nplSize) {
    auto model = new MyMLP();
    return model;
}


__declspec(dllexport) void trainMlpModel(MyMLP *model,
                                         double samplesInputs[],
                                         double samplesExpectedOutputs[],
                                         int sampleCount,
                                         int inputDim,
                                         int outputDim,
                                         double alpha,
                                         int nbIter
) {
    // Fit model
}

__declspec(dllexport) double *predictMlpModelClassification(MyMLP *model,
                                                            double sampleInputs[],
                                                            int inputDim
) {
    return new double[]{1.0, 2.0, 3.0};
}

__declspec(dllexport) void destroyMlpModel(MyMLP *model) {
    delete model;
}

__declspec(dllexport) void destroyMlpResult(const double *result) {
    delete[]result;
}

}
