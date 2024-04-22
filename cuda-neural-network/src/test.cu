#include <chrono>
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

#define total_coordinates 32768

float computeAccuracy(const Matrix& predictions, const Matrix& targets);

void loadModel(NeuralNetwork& nn, const char* filePath) {
    std::string item_name;
    std::ifstream nameFileout;
    nameFileout.open(filePath);
    std::string line;
    while(std::getline(nameFileout, line))
    {
        if(line.compare("Linear") == 0){
            std::string shapeX, shapeY;
            std::getline(nameFileout, shapeX);
            std::getline(nameFileout, shapeY);

            int xW = std::stoi(shapeX);
            int yW = std::stoi(shapeY);

            float* weights = new float[xW*yW];
            
            std::string number;

            for(int i = 0; i < xW*yW; i++){
                std::getline(nameFileout, number);
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
        } 
        else if(line.compare("Sigmoid") == 0){
	        nn.addLayer(new SigmoidActivation("sigmoid"));
        }
        else if(line.compare("Relu") == 0){
	        nn.addLayer(new ReLUActivation("relu"));
        }
    }
}
 
int main(int argc, char** argv) {
    NeuralNetwork nn1, nn2, nn3, nn4;
    loadModel(nn1, "nn1.txt");
    loadModel(nn2, "nn2.txt");
    loadModel(nn3, "nn3.txt");
    loadModel(nn4, "nn4.txt");

    int number_of_streams = std::stoi(argv[1]);
    int batch_size = std::stoi(argv[2]);
    int number_of_batches = total_coordinates/batch_size;

    srand(1000);
    CoordinatesDataset dataset(batch_size, number_of_batches);
    Matrix Y1, Y2, Y3, Y4;

    if (number_of_streams == 1) {
        cudaStream_t stream1; 
        cudaStreamCreate (&stream1);
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < number_of_batches; i++) {
            Y1 = nn1.forward(dataset.getBatches().at(i), stream1);
            Y2 = nn2.forward(dataset.getBatches().at(i), stream1);
            Y3 = nn3.forward(dataset.getBatches().at(i), stream1);
            Y4 = nn4.forward(dataset.getBatches().at(i), stream1);
            Y1.copyDeviceToHost();
            Y2.copyDeviceToHost();
            Y3.copyDeviceToHost();
            Y4.copyDeviceToHost();
        }
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Total time: " << duration.count() << " milliseconds" << std::endl;
    }
    else if (number_of_streams == 2) {
        cudaStream_t stream1, stream2;
        cudaStreamCreate (&stream1);
        cudaStreamCreate (&stream2);
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < number_of_batches; i++) {
            Y1 = nn1.forward(dataset.getBatches().at(i), stream1);
            Y2 = nn2.forward(dataset.getBatches().at(i), stream2);
            Y3 = nn3.forward(dataset.getBatches().at(i), stream1);
            Y4 = nn4.forward(dataset.getBatches().at(i), stream2);
            Y1.copyDeviceToHost();
            Y2.copyDeviceToHost();
            Y3.copyDeviceToHost();
            Y4.copyDeviceToHost();
        }
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Total time: " << duration.count() << " milliseconds" << std::endl;
    }
    else if (number_of_streams == 4) {
        cudaStream_t stream1, stream2, stream3, stream4;
        cudaStreamCreate (&stream1);
        cudaStreamCreate (&stream2);
        cudaStreamCreate (&stream3);
        cudaStreamCreate (&stream4);
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < number_of_batches; i++) {
            Y1 = nn1.forward(dataset.getBatches().at(i), stream1);
            Y2 = nn2.forward(dataset.getBatches().at(i), stream2);
            Y3 = nn3.forward(dataset.getBatches().at(i), stream3);
            Y4 = nn4.forward(dataset.getBatches().at(i), stream4);
            Y1.copyDeviceToHost();
            Y2.copyDeviceToHost();
            Y3.copyDeviceToHost();
            Y4.copyDeviceToHost();
        }
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Total time: " << duration.count() << " milliseconds" << std::endl;
    }
    else {
        std::cout << "Invalid number of streams" << std::endl;
        return 1;
    }
    
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
