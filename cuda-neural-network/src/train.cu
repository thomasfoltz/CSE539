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
            for (int i = 0; i < weights.shape.x*weights.shape.y; i++) {
                    file << weights.data_host.get()[i] << "\n";
            }

            // Save biases
            file << biases.shape.x << "\n" << biases.shape.y << "\n";
			for (int i = 0; i < biases.shape.x*biases.shape.y; i++) {
                file << biases.data_host.get()[i] << "\n";
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
	srand(1000);

    CoordinatesDataset dataset(100, 21);

    BCECost bce_cost;

	// NeuralNetwork nn1;
	// nn1.addLayer(new LinearLayer("linear_1", Shape(2, 30)));
	// nn1.addLayer(new ReLUActivation("relu_1"));
	// nn1.addLayer(new LinearLayer("linear_2", Shape(30, 1)));
	// nn1.addLayer(new SigmoidActivation("sigmoid_output"));

    NeuralNetwork nn2;
    nn2.addLayer(new LinearLayer("linear_1", Shape(2, 60)));
    nn2.addLayer(new ReLUActivation("relu_1"));
    // nn2.addLayer(new LinearLayer("linear_2", Shape(30, 60)));
    // nn2.addLayer(new ReLUActivation("relu_2"));
    nn2.addLayer(new LinearLayer("linear_3", Shape(60, 1)));
    nn2.addLayer(new ReLUActivation("relu_3"));
    nn2.addLayer(new SigmoidActivation("sigmoid_output"));

	// network training
	Matrix Y1, Y2;
	for (int epoch = 0; epoch < 1001; epoch++) {
		float cost1 = 0.0;
        float cost2 = 0.0;

		for (int batch = 0; batch < dataset.getNumOfBatches() - 1; batch++) {
			// Y1 = nn1.forward(dataset.getBatches().at(batch));
            Y2 = nn2.forward(dataset.getBatches().at(batch));
			// nn1.backprop(Y1, dataset.getTargets().at(batch));
            nn2.backprop(Y2, dataset.getTargets().at(batch));
            // cost1 += bce_cost.cost(Y1, dataset.getTargets().at(batch));
            cost2 += bce_cost.cost(Y2, dataset.getTargets().at(batch));
		}

		if (epoch % 100 == 0) {
			std::cout 	<< "Epoch: " << epoch
						<< ", Cost: " << cost1 / dataset.getNumOfBatches()
                        << ", Cost: " << cost2 / dataset.getNumOfBatches()
						<< std::endl;
		}
	}

	// saveModel(nn1, "nn1.txt");
    saveModel(nn2, "nn2.txt");
	
	return 0;
}
