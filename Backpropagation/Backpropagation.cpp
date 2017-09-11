// Backpropagation.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "Data.h"
#include "NeuralNetwork.h"
using namespace std;

int main()
{
	Data data("C:\\Users\\Kagami\\Documents\\Visual Studio 2015\\Projects\\Backpropagation\\Backpropagation\\all.txt",0.8);
	NeuralNetwork neurons(27,4,10,0.05,0.01);
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
	neurons.trainNetwork(100,1);
	return 0;
}

