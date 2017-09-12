#include "stdafx.h"
#include "NeuralNetwork.h"


void NeuralNetwork::setTrainData(std::vector<std::vector<double>>& _trainData)
{
	trainData = _trainData;
}

void NeuralNetwork::setTestData(std::vector<std::vector<double>>& _testData)
{
	testData = _testData;
}

void NeuralNetwork::setRandomWeightsAndBiases()
{
	//hiddenWeights
	for (int i = 0; i < inputNeurons; i++)
	{
		for (int j = 0; j < hiddenNeurons; j++)
		{
			//hiddenWeights[i][j] = (((double)(rand() * 20) / RAND_MAX) - 10);
			hiddenWeights[i][j] = ((0.001 - 0.0001)*((double)(rand()) / RAND_MAX) + 0.00001);
		}
	}

	//hiddenBias
	for (int i = 0; i < hiddenNeurons; i++)
	{
		//hiddenBias[i] = (((double)(rand() * 20) / RAND_MAX) - 10);
		hiddenBias[i] = ((0.001 - 0.0001)*((double)(rand()) / RAND_MAX) + 0.00001);
	}

	//outputWeights
	for (int i = 0; i < hiddenNeurons; i++)
	{
		for (int j = 0; j < outputNeurons; j++)
		{
			//outputWeights[i][j] = (((double)(rand() * 20) / RAND_MAX) - 10);
			outputWeights[i][j] = ((0.001 - 0.0001)*((double)(rand()) / RAND_MAX) + 0.00001);
		}
	}
	//outputBias
	for (int i = 0; i < outputNeurons; i++)
	{
		//outputBias[i] = (((double)(rand() * 20) / RAND_MAX) - 10);
		outputBias[i] = ((0.001 - 0.0001)*((double)(rand()) / RAND_MAX) + 0.00001);
	}
}

NeuralNetwork::NeuralNetwork(int inputNeurons, int outputNeurons, int hiddenNeurons, double learningRate, double momentum):
	inputNeurons(inputNeurons), outputNeurons(outputNeurons), hiddenNeurons(hiddenNeurons), 
	learningRate(learningRate), momentum(momentum)
{
	//srand(1);
	srand(time(NULL));
	outputWeights = std::vector<std::vector<double>>(hiddenNeurons);
	for (int i = 0; i < hiddenNeurons; i++) { outputWeights[i] = std::vector<double>(outputNeurons); }
	hiddenWeights = std::vector<std::vector<double>>(inputNeurons);
	for (int i = 0; i < inputNeurons; i++) { hiddenWeights[i] = std::vector<double>(hiddenNeurons); }
	outputSignals = std::vector<double>(outputNeurons);
	hiddenSignals = std::vector<double>(hiddenNeurons);
	outputGradient = std::vector<std::vector<double>>(hiddenNeurons);
	for (int i = 0; i < hiddenNeurons; i++) { outputGradient[i] = std::vector<double>(outputNeurons); }
	hiddenGradient = std::vector<std::vector<double>>(inputNeurons);
	for (int i = 0; i < inputNeurons; i++) { hiddenGradient[i] = std::vector<double>(hiddenNeurons); }
	outputBiasGradient = std::vector<double>(outputNeurons);
	hiddenBiasGradient = std::vector<double>(hiddenNeurons);
	correctOutput = std::vector<double>(outputNeurons);
	computedOutput = std::vector<double>(outputNeurons);
	hiddenOutput = std::vector<double>(hiddenNeurons);
	outputPreviousWeightDelta = std::vector<std::vector<double>>(hiddenNeurons);
	for (int i = 0; i < hiddenNeurons; i++) { outputPreviousWeightDelta[i] = std::vector<double>(outputNeurons); }
	hiddenPreviousWeightDelta = std::vector<std::vector<double>>(inputNeurons);
	for (int i = 0; i < inputNeurons; i++) { hiddenPreviousWeightDelta[i] = std::vector<double>(hiddenNeurons); }
	outputBias = std::vector<double>(outputNeurons);
	hiddenBias = std::vector<double>(hiddenNeurons);
	outputPreviousBiasDelta = std::vector<double>(outputNeurons);
	hiddenPreviousBiasDelta = std::vector<double>(hiddenNeurons);
	setRandomWeightsAndBiases();

}

NeuralNetwork::~NeuralNetwork()
{
}

void NeuralNetwork::computeOutputs(std::vector<double> data)
{
	std::vector<double> tempSumHidden(hiddenNeurons);
	std::vector<double> tempSumOutput(outputNeurons);
	//ustala computedOutputs i hiddenOutputs

	//dla neuronów ukrytych, mnożymy przez wagi
	for (int i = 0; i < hiddenNeurons; i++)
	{
		for (int j = 0; j < inputNeurons; j++)
		{
			tempSumHidden[i] += data[j + 1] * hiddenWeights[j][i];
		}
	}

	//dodajemy biasy
	for (int i = 0; i < hiddenNeurons; i++)
	{
		tempSumHidden[i] += hiddenBias[i];
	}

	//funkcja aktywacji - tangens hiperboliczny
	for (int i = 0; i < hiddenNeurons; i++)
	{
		hiddenOutput[i] = tanh(tempSumHidden[i]);
	}

	//teraz można liczyc neurony wyjściowe, najpierw z neronów ukrytych * wagi
	for (int i = 0; i < outputNeurons; i++)
	{
		for (int j = 0; j < hiddenNeurons; j++)
		{
			tempSumOutput[i] += hiddenOutput[j] * outputWeights[j][i];
		}
	}

	//dodanie biasów
	for (int i = 0; i < outputNeurons; i++)
	{
		tempSumOutput[i] += outputBias[i];
	}

	//funkcja aktywacji softmax
	double sumSoftmax = 0;
	for (int i = 0; i < outputNeurons; i++)
	{
		sumSoftmax += exp(tempSumOutput[i]);
	}

	for (int i = 0; i < outputNeurons; i++)
	{
		computedOutput[i] = (exp(tempSumOutput[i]) / sumSoftmax);
	}

}

void NeuralNetwork::calculateGradients(int index)
{
	double errorSignal;
	double softmaxDerivative, tanhDerivative;
	//obliczanie lokalnych gradientow bledu dla neuronów wyjściowych
	for (int i = 0; i < outputNeurons; i++)
	{
		errorSignal = correctOutput[i] - computedOutput[i]; //obliczanie sygnalu bledu dla neuronu (ej = oj - tj)
		softmaxDerivative = computedOutput[i] * (1 - computedOutput[i]); // (oj' = oj * (1 - oj))
		outputSignals[i] = errorSignal * softmaxDerivative; // (qj = ej * oj')
	}

	//obliczenie gradientów dla wag i biasów wyjściowych (dE/dwij = qj * xi)
	for (int i = 0; i < hiddenNeurons; i++)
	{
		for (int j = 0; j < outputNeurons; j++)
		{
			outputGradient[i][j] = outputSignals[j] * hiddenOutput[i];
		}
	}

	for (int i = 0; i < outputNeurons; i++)
	{
		outputBiasGradient[i] = outputSignals[i];
	}

	//obliczanie lokalnych gradientów błędów dla neuronów ukrytych
	for (int i = 0; i < hiddenNeurons; i++)
	{
		tanhDerivative = (1 - hiddenOutput[i])*(1 + hiddenOutput[i]); //tu wychodza cuda, bo tu zawsze beda same zera...
		double sum = 0;
		for (int j = 0; j < outputNeurons; j++)
		{
			sum += (outputSignals[j] * outputWeights[i][j]);
		}
		hiddenSignals[i] = sum * tanhDerivative;
	}

	//obliczanie gradientów dla wag i biasów ukrytych
	for (int i = 0; i < inputNeurons; i++)
	{
		for (int j = 0; j < hiddenNeurons; j++)
		{
			hiddenGradient[i][j] = hiddenSignals[j] * trainData[index][i + 1];
		}
	}

	for (int i = 0; i < hiddenNeurons; i++)
	{
		hiddenBiasGradient[i] = hiddenSignals[i];
	}
}


void NeuralNetwork::updateWeights()
{
	double delta;

	//aktualizacja wag wartwy ukrytej
	for (int i = 0; i < inputNeurons; i++)
	{
		for (int j = 0; j < hiddenNeurons; j++)
		{
			delta = hiddenGradient[i][j] * learningRate; // delta wij = alfa * dE/dwij
			hiddenWeights[i][j] += (delta + (hiddenPreviousWeightDelta[i][j]*momentum));
			hiddenPreviousWeightDelta[i][j] = delta;
		}
	}
	//aktualizacja biasów warstwy ukrytej
	for (int i = 0; i < hiddenNeurons; i++)
	{
		delta = hiddenBiasGradient[i] * learningRate;
		hiddenBias[i] += (delta + (hiddenPreviousBiasDelta[i] * momentum));
		hiddenPreviousBiasDelta[i] = delta;
	}

	//aktualizacja wag wartwy wyjściowej
	for (int i = 0; i < hiddenNeurons; i++)
	{
		for (int j = 0; j < outputNeurons; j++)
		{
			delta = outputGradient[i][j] * learningRate;
			outputWeights[i][j] += (delta + (outputPreviousWeightDelta[i][j]*momentum));
			outputPreviousWeightDelta[i][j] = delta;
		}
	}

	//aktualizacja biasów wartwy wyjściowej
	for (int i = 0; i < outputNeurons; i++)
	{
		delta = outputBiasGradient[i] * learningRate;
		outputBias[i] += (delta + (outputPreviousBiasDelta[i] * momentum));
		outputPreviousBiasDelta[i] = delta;
	}

}

void NeuralNetwork::DurstenfeldShuffle(std::vector<int> &table)
{
	
	//https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle#The_modern_algorithm
	//--To shuffle an array a of n elements(indices 0..n - 1) :
	//	for i from 0 to n−2 do
	//		j ← random integer such that i ≤ j < n
	//		exchange a[i] and a[j]
	int size = table.size();
	for (int i = 0; i < (size - 1); i++)
	{
		int j = rand() % ((size - 1) - i) + i;
		int temp = table[i];
		table[i] = table[j];
		table[j] = temp;
	}
}



void NeuralNetwork::trainNetwork(int epochs, int step)
{
	
	std::vector<int> indexes(trainData.size()); //wektor z indexami, ktore będą mieszane

	for (int i = 0; i < indexes.size(); i++)
	{
		indexes[i]=i;
	}

	for (int i = 1; i <= epochs; i++) //dla każdej epoki
	{
		//trenuj
		//przejscie przez cala tablice trainData
		// TODO jaki size() ? jeden wymiar czy dwa?
		for (int i = 0; i < trainData.size(); i++)
		{
			int index = indexes[i]; //jeden z pomieszanych indeksów
			
			//ustalenie poprawnej tablicy wyjsciowej
			switch (((int)trainData[index][0]))
			{
			case 1:
				correctOutput[0] = 1;
				correctOutput[1] = 0;
				correctOutput[2] = 0;
				correctOutput[3] = 0;
				break;
			case 2:
				correctOutput[0] = 0;
				correctOutput[1] = 1;
				correctOutput[2] = 0;
				correctOutput[3] = 0;
				break;
			case 3:
				correctOutput[0] = 0;
				correctOutput[1] = 0;
				correctOutput[2] = 1;
				correctOutput[3] = 0;
				break;
			case 4:
				correctOutput[0] = 0;
				correctOutput[1] = 0;
				correctOutput[2] = 0;
				correctOutput[3] = 1;
				break;
			}

			computeOutputs(trainData[index]); //obliczenie przez siec wyjsc
			calculateGradients(index); //obliczenie gradientów
			updateWeights();
		}

		//if ((i!=0) && ((i % step == 0) || (i == epochs))) //liczenie co okreslony krok
		if (i%step ==0)
		{
			long double errorSum = 0;
			int right = 0;
			for (int j = 0; j < testData.size(); j++)
			{
				switch (((int)testData[j][0]))
				{
				case 1:
					correctOutput[0] = 1;
					correctOutput[1] = 0;
					correctOutput[2] = 0;
					correctOutput[3] = 0;
					break;
				case 2:
					correctOutput[0] = 0;
					correctOutput[1] = 1;
					correctOutput[2] = 0;
					correctOutput[3] = 0;
					break;
				case 3:
					correctOutput[0] = 0;
					correctOutput[1] = 0;
					correctOutput[2] = 1;
					correctOutput[3] = 0;
					break;
				case 4:
					correctOutput[0] = 0;
					correctOutput[1] = 0;
					correctOutput[2] = 0;
					correctOutput[3] = 1;
					break;
				}
				computeOutputs(testData[j]);
				int maxCorrect = 0, maxComputed = 0;
				for (int k = 0; k < outputNeurons; k++)
				{
					double error = correctOutput[k] - computedOutput[k];
					errorSum += (error*error); //mean square error 

					if (correctOutput[k] > correctOutput[maxCorrect]) { maxCorrect = k; } //wyszukanie 1 w poprawnych wyjsciach
					if (computedOutput[k] > computedOutput[maxComputed]) { maxComputed = k; } //wyszukanie najwiekszej w obliczonych wyjsciach
				
				}
				if (maxCorrect == maxComputed) { right++; }
			}
			long double mse = errorSum / testData.size();
			double percentage = ((double)right) / testData.size();
			std::cout <<"Epoch: " << std::setw(10)<< i << " MSE: " << std::setw(10) << mse << " Correct %: "<< std::setw(10)<< percentage << std::endl;
			/*errorSum = 0;
			for (int j = 0; j < trainData.size(); j++)
			{
				switch (((int)trainData[j][0]))
				{
				case 1:
					correctOutput[0] = 1;
					correctOutput[1] = 0;
					correctOutput[2] = 0;
					correctOutput[3] = 0;
					break;
				case 2:
					correctOutput[0] = 0;
					correctOutput[1] = 1;
					correctOutput[2] = 0;
					correctOutput[3] = 0;
					break;
				case 3:
					correctOutput[0] = 0;
					correctOutput[1] = 0;
					correctOutput[2] = 1;
					correctOutput[3] = 0;
					break;
				case 4:
					correctOutput[0] = 0;
					correctOutput[1] = 0;
					correctOutput[2] = 0;
					correctOutput[3] = 1;
					break;
				}
				computeOutputs(trainData[j]);
				for (int k = 0; k < outputNeurons; k++)
				{
					double error = correctOutput[k] - computedOutput[k];
					errorSum += (error*error);
				}
			}
			mse = errorSum / trainData.size();
			std::cout << "   train: " << mse << std::endl;*/

			//testuj
			//na konsole wyniki testowania
		}
		DurstenfeldShuffle(indexes);
		//mieszanie indeksow
	}
}

void NeuralNetwork::writeNeuralNetworkToFile(std::string filename)
{

}

