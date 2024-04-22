#pragma once
#include "nn_layer.hh"

// for unit testing purposes only
namespace {
	class LinearLayerTest_ShouldReturnOutputAfterForwardProp_Test;
	class NeuralNetworkTest_ShouldPerformForwardProp_Test;
	class LinearLayerTest_ShouldReturnDerivativeAfterBackprop_Test;
	class LinearLayerTest_ShouldUptadeItsBiasDuringBackprop_Test;
	class LinearLayerTest_ShouldUptadeItsWeightsDuringBackprop_Test;
}

class LinearLayer : public NNLayer {
private:
	const float weights_init_threshold = 0.01;

	Matrix W;
	Matrix b;

	Matrix Z;
	Matrix A;
	Matrix dA;

	void initializeBiasWithZeros();
	void initializeWeightsRandomly();
	void initializeWeightsFrom(float* weights);
	void initializeBiasFrom(float* weights);


	void computeAndStoreBackpropError(Matrix& dZ);
	void computeAndStoreLayerOutput(Matrix& A, cudaStream_t stream);
	void updateWeights(Matrix& dZ, float learning_rate);
	void updateBias(Matrix& dZ, float learning_rate);

public:
	LinearLayer(std::string name, Shape W_shape, float* weights, float* bias);
	~LinearLayer();
	LinearLayer(std::string name, Shape W_shape);


	Matrix& forward(Matrix& A, cudaStream_t stream=0);
	Matrix& backprop(Matrix& dZ, float learning_rate = 0.01);

	int getXDim() const;
	int getYDim() const;

	Matrix getWeightsMatrix() const;
	Matrix getBiasVector() const;

	void setWeightsMatrix(const Matrix& newWeights);
    void setBiasVector(const Matrix& newBiases);

	// for unit testing purposes only
	friend class LinearLayerTest_ShouldReturnOutputAfterForwardProp_Test;
	friend class NeuralNetworkTest_ShouldPerformForwardProp_Test;
	friend class LinearLayerTest_ShouldReturnDerivativeAfterBackprop_Test;
	friend class LinearLayerTest_ShouldUptadeItsBiasDuringBackprop_Test;
	friend class LinearLayerTest_ShouldUptadeItsWeightsDuringBackprop_Test;
};
