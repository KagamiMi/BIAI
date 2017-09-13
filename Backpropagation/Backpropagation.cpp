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
	if (argc != 9) {
		std::cout << "Wrong number of input parameters";
		return 1;
	}
	string dataPath = argv[1];
	double splitRatio = atof(argv[2]);
	if ((splitRatio <= 0) || (splitRatio >= 1)) {
		std::cout << "Wrong split ratio";
		return 1;
	}
	int hiddenNeurons = atoi(argv[3]);
	if (hiddenNeurons<1) {
		std::cout << "Wrong hidden neurons number";
		return 1;
	}
	double learningRate = atof(argv[4]);
	if (learningRate<=0) {
		std::cout << "Wrong learningRate";
		return 1;
	}
	double momentum = atof(argv[5]);
	if (momentum<0) {
		std::cout << "Wrong momentum";
		return 1;
	}
	int epochs = atoi(argv[6]);
	if (epochs<=0) {
		std::cout << "Wrong epochs number";
		return 1;
	}
	int interval = atoi(argv[7]);
	if ((interval <= 0) || (interval>epochs)) {
		std::cout << "Wrong interval number";
		return 1;
	}
	string savePath = argv[8];
	Data data(dataPath,splitRatio);
	NeuralNetwork neurons(27,4,hiddenNeurons,learningRate,momentum);
	neurons.setTrainData(data.getTrainData());
	neurons.setTestData(data.getTestData());
	neurons.trainNetwork(epochs,interval);
	neurons.writeNeuralNetworkToFile(savePath);
	getchar();
	return 0;
}

