#include <fstream>
#include <vector>
#include <iostream>
#include <time.h>

#include "neural_network.hh"
#include "layers/linear_layer.hh"
#include "layers/relu_activation.hh"
#include "layers/sigmoid_activation.hh"
#include "nn_utils/nn_exception.hh"
#include "nn_utils/matrix.hh"

#include "coordinates_dataset.hh"

void saveModel(NeuralNetwork nn, const char* filePath) {
    std::ofstream file(filePath);
    std::vector<NNLayer*> layers = nn.getLayers();
    for (NNLayer* layer : layers) {
        if (!layer){
            std::cerr << "Encountered null layer pointer" << std::endl;
            continue;
        }
        LinearLayer* linearLayer = dynamic_cast<LinearLayer*>(layer);
        if (linearLayer) {
            Matrix weights = linearLayer->getWeightsMatrix();
            Matrix biases = linearLayer->getBiasVector();

            // Save layer type
            file << "Linear\n";

            // Save weights
            file << weights.shape.x << "\n" << weights.shape.y << "\n";
            for (int i = 0; i < weights.shape.x; i++) {
                for (int j = 0; j < weights.shape.y; j++) {
                    file << weights.data_host.get()[i * weights.shape.y + j] << "\n";
                }
            }

            // Save biases
            file << biases.shape.x << "\n" << biases.shape.y << "\n";
            for (int i = 0; i < biases.shape.x; i++) {
                for (int j = 0; j < biases.shape.y; j++) {
                    file << biases.data_host.get()[i * biases.shape.y + j] << "\n";
                }
            }
        }
        else if(layer->getName().find("sigmoid") != std::string::npos){
            file << "Sigmoid\n";
        }
        else{
            file << "Relu\n";

        }
    }

    file.close();
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

	saveModel(nn, "model.txt");
	
	return 0;
}
