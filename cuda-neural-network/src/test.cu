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

void loadModel(NeuralNetwork& nn, const std::string& modelFilePath) {
    std::string item_name;
    std::ifstream nameFileout;
    nameFileout.open(modelFilePath);
    std::string line;
    while(std::getline(nameFileout, line))
    {
        if(line.compare("Linear") == 0){
            // printf("LINEAR\n");
            std::string shapeX, shapeY;
            std::getline(nameFileout, shapeX);
            std::getline(nameFileout, shapeY);

            int xW = std::stoi(shapeX);
            int yW = std::stoi(shapeY);

            float* weights = new float[xW*yW];
            
            std::string number;

            for(int i = 0; i < xW*yW; i++){
                std::getline(nameFileout, number);
                // std::cout << number << "\n";
                weights[i] = stof(number);
            }

            std::getline(nameFileout, shapeX);
            std::getline(nameFileout, shapeY);

            int x = std::stoi(shapeX);
            int y = std::stoi(shapeY);

            float* bias = new float[x*y];
            
            for(int i = 0; i < x*y; i++){
                std::getline(nameFileout, number);
                bias[i] = stof(number);
            }


            nn.addLayer(new LinearLayer("linear", Shape(xW, yW), weights, bias));

        }   // the item name, replace line with item_name in the code above)
        else if(line.compare("Sigmoid") == 0){
	        nn.addLayer(new SigmoidActivation("sigmoid"));
        }
        else if(line.compare("Relu") == 0){
	        nn.addLayer(new ReLUActivation("relu"));
        }
        // std::cout << "line:" << line << std::endl;
    }
}

int main() {
	
    NeuralNetwork nn1, nn2;
    loadModel(nn1, "nn1.txt");
    loadModel(nn2, "nn2.txt");

    srand( time(NULL) );

	CoordinatesDataset dataset(100, 21);
	Matrix Y1, Y2;

	// compute accuracy
    for (int i = 0; i < 10; i++) {
        Y1 = nn1.forward(dataset.getBatches().at(dataset.getNumOfBatches() - 1));
        printf("\ny1");
        Y2 = nn2.forward(dataset.getBatches().at(dataset.getNumOfBatches() - 1));
        printf("\ny2");
    }

    Y1.copyDeviceToHost();
    Y2.copyDeviceToHost();

	float accuracy1 = computeAccuracy(Y1, dataset.getTargets().at(dataset.getNumOfBatches() - 1));
	float accuracy2 = computeAccuracy(Y2, dataset.getTargets().at(dataset.getNumOfBatches() - 1));

	std::cout 	<< "Accuracy 1: " << accuracy1 << std::endl;
    std::cout 	<< "Accuracy 2: " << accuracy2 << std::endl;


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
