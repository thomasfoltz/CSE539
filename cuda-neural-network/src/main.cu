#include <fstream>
#include <vector>
#include <iostream>
#include <time.h>

#include "neural_network.hh"
#include "layers/linear_layer.hh"
#include "layers/relu_activation.hh"
#include "layers/sigmoid_activation.hh"
#include "nn_utils/nn_exception.hh"
#include "nn_utils/bce_cost.hh"
#include "nn_utils/matrix.hh"

#include "coordinates_dataset.hh"

float computeAccuracy(const Matrix& predictions, const Matrix& targets);

void saveModel(NeuralNetwork nn, const char* filePath) {
    std::ofstream file(filePath, std::ios::binary);
    std::vector<NNLayer*> layers = nn.getLayers();
    for (NNLayer* layer : layers) {
        LinearLayer* linearLayer = dynamic_cast<LinearLayer*>(layer);
        if (linearLayer) {
            Matrix weights = linearLayer->getWeightsMatrix();
            Matrix biases = linearLayer->getBiasVector();

            // Save weights
            int weightsSize = weights.shape.x * weights.shape.y;
            file.write(reinterpret_cast<char*>(&weightsSize), sizeof(weightsSize));
			file.write((char*)(weights.data_host.get()), weightsSize * sizeof(float));

            // Save biases
            int biasesSize = biases.shape.x * biases.shape.y;
            file.write(reinterpret_cast<char*>(&biasesSize), sizeof(biasesSize));
            file.write((char*)(biases.data_host.get()), biasesSize * sizeof(float));
        }
    }

    file.close();
}

void loadModel(NeuralNetwork& nn, const char* filePath) {
    std::ifstream file(filePath, std::ios::binary);
    std::vector<NNLayer*> layers = nn.getLayers();
    for (NNLayer* layer : layers) {
        LinearLayer* linearLayer = dynamic_cast<LinearLayer*>(layer);
        if (linearLayer) {
			int weightsRows = linearLayer->getXDim();
			int weightsCols = linearLayer->getXDim();

            Matrix weights = Matrix(weightsRows, weightsCols);
			weights.allocateCudaMemory();
			std::unique_ptr<float[]> weightsData(new float[weightsRows * weightsCols]);
			file.read(reinterpret_cast<char*>(weightsData.get()), weightsRows * weightsCols * sizeof(float));
			weights.copyHostToDevice();

            int biasSize = linearLayer->getXDim();
			Matrix bias = Matrix(biasSize);
            bias.allocateCudaMemory();
			std::unique_ptr<float[]> biasData(new float[biasSize]);
			file.read(reinterpret_cast<char*>(biasData.get()), biasSize * sizeof(float));
			bias.copyHostToDevice();

            linearLayer->setWeightsMatrix(weights);
            linearLayer->setBiasVector(bias);
        }
    }
}

int main() {
	srand( time(NULL) );

	CoordinatesDataset dataset(100, 21);
	BCECost bce_cost;

	NeuralNetwork nn;
	nn.addLayer(new LinearLayer("linear_1", Shape(2, 30)));
	nn.addLayer(new ReLUActivation("relu_1"));
	nn.addLayer(new LinearLayer("linear_2", Shape(30, 1)));
	nn.addLayer(new SigmoidActivation("sigmoid_output"));

	// network training
	Matrix Y;
	for (int epoch = 0; epoch < 1001; epoch++) {
		float cost = 0.0;

		for (int batch = 0; batch < dataset.getNumOfBatches() - 1; batch++) {
			Y = nn.forward(dataset.getBatches().at(batch));
			nn.backprop(Y, dataset.getTargets().at(batch));
			cost += bce_cost.cost(Y, dataset.getTargets().at(batch));
		}

		if (epoch % 100 == 0) {
			std::cout 	<< "Epoch: " << epoch
						<< ", Cost: " << cost / dataset.getNumOfBatches()
						<< std::endl;
		}
	}

	saveModel(nn, "model.bin");

	// compute accuracy
	Y = nn.forward(dataset.getBatches().at(dataset.getNumOfBatches() - 1));
	Y.copyDeviceToHost();

	float accuracy = computeAccuracy(
			Y, dataset.getTargets().at(dataset.getNumOfBatches() - 1));
	std::cout 	<< "Accuracy: " << accuracy << std::endl;

	return 0;
}

float computeAccuracy(const Matrix& predictions, const Matrix& targets) {
	int m = predictions.shape.x;
	int correct_predictions = 0;

	for (int i = 0; i < m; i++) {
		float prediction = predictions[i] > 0.5 ? 1 : 0;
		if (prediction == targets[i]) {
			correct_predictions++;
		}
	}

	return static_cast<float>(correct_predictions) / m;
}
