#pragma once
#include <vector>
#include <cmath>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <iomanip>

class NeuralNetwork
{
private:
	int inputNeurons;
	int outputNeurons;
	int hiddenNeurons;
	double learningRate;
	double momentum;

	std::vector<double> outputBias;
	std::vector<double> hiddenBias;
	std::vector<double> outputPreviousBiasDelta;
	std::vector<double> hiddenPreviousBiasDelta;
	std::vector<std::vector<double>> outputPreviousWeightDelta;
	std::vector<std::vector<double>> hiddenPreviousWeightDelta;
	std::vector<std::vector<double>> outputWeights;
	std::vector<std::vector<double>> hiddenWeights;
	std::vector<double> outputSignals; //lokalny gradient b³êdu sygna³ów wyjœciowych 
	std::vector<double> hiddenSignals; //lokalny gradient b³êdu sygna³ów ukrytych
	std::vector<std::vector<double>> outputGradient;
	std::vector<std::vector<double>> hiddenGradient;
	std::vector<double> outputBiasGradient;
	std::vector<double> hiddenBiasGradient;
	std::vector<std::vector<double>> trainData;
	std::vector<std::vector<double>> testData;
	std::vector<double> correctOutput;
	std::vector<double> computedOutput;
	std::vector<double> hiddenOutput;
	void computeOutputs(std::vector<double> data);
	void calculateGradients(int index);
	void setRandomWeightsAndBiases();
	void updateWeights();
	void DurstenfeldShuffle(std::vector<int>& table);
public:
	NeuralNetwork(int inputNeurons, int outputNeurons, int hiddenNeurons, double learningRate, double momentum);
	~NeuralNetwork();
	void setTrainData(std::vector<std::vector<double>> &trainData);
	void setTestData(std::vector<std::vector<double>> &testData);
	void trainNetwork(int epochs, int step);
	void writeNeuralNetworkToFile(std::string filename);
};

