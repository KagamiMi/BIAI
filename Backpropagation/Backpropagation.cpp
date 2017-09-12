// Backpropagation.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "Data.h"
#include "NeuralNetwork.h"
using namespace std;

int main(int argc, char* argv[])
{
	// argv[0] - name of program
	// argv[1] - .txt with data
	// argv[2] - split data ratio
	// argv[3] - input neurons 
	// argv[4] - output neurons
	// argv[5] - hidden neurons
	// argv[6] - learning rate
	// argv[7] - momentum
	// argv[8] - number of epochs
	// argv[9] - checking test data interval
	if (argc != 10) {
		std::cout << "Wrong number of input parameters";
		return 1;
	}
	string dataPath = argv[1];
	double splitRatio = atof(argv[2]);
	int inputNeurons = atoi(argv[3]);
	int outputNeurons = atoi(argv[4]);
	int hiddenNeurons = atoi(argv[5]);
	double learningRate = atof(argv[6]);
	double momentum = atof(argv[7]);
	int epochs = atoi(argv[8]);
	int interval = atoi(argv[9]);

	Data data(dataPath,splitRatio);
	NeuralNetwork neurons(inputNeurons,outputNeurons,hiddenNeurons,learningRate,momentum);
	std::vector<std::vector<double>> train;
	std::vector<std::vector<double>> test;
	srand(time(NULL));
	/*for (int i = 0; i < 800; ++i)
	{
		std::vector<double> temp;
		temp.push_back((rand()%4 + 1));
		for (int j = 0; j < 27; ++j)
		{
			temp.push_back((((double)(rand() * 20) / RAND_MAX) - 10));
		}
		train.push_back(temp);
	}
	for (int i = 0; i < 200; ++i)
	{
		std::vector<double> temp;
		temp.push_back((rand() % 4 + 1));
		for (int j = 0; j < 27; ++j)
		{
			temp.push_back((((double)(rand() * 20) / RAND_MAX) - 10));
		}
		test.push_back(temp);
	}*/
	neurons.setTrainData(data.getTrainData());
	neurons.setTestData(data.getTestData());
	//neurons.setTrainData(train);
	//neurons.setTestData(test);
	neurons.trainNetwork(epochs,interval);
	neurons.writeNeuralNetworkToFile("weights.txt");
	return 0;
}

