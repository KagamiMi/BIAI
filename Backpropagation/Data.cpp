#include "stdafx.h"
#include "Data.h"

void Data::readData()
{
	file.open(filename);
	if (file) //jesli otworzylismy
	{
		char temp;
		file.get(temp);
		while (temp != '\n') 
		{
			file.get(temp);
		}; //przechodzimy przez pierwsz¹ liniê
		
		//help
		std::vector<std::vector<double>> sugi;
		std::vector<std::vector<double>> hinoki;
		std::vector<std::vector<double>> deciduous;
		std::vector<std::vector<double>> other;

		int i = 0;
		std::vector<double> oneLine;
		while (!file.eof())
		{
			std::vector<double> oneLine;
			file >> temp;
			switch (temp) {
			case 's':
				oneLine.push_back(1);
				break;
			case 'h':
				oneLine.push_back(2);
				break;
			case 'd':
				oneLine.push_back(3);
				break;
			case 'o':
				oneLine.push_back(4);
				break;
			}
			file >> temp;
			do {
				double tmp;
				file >> tmp;
				oneLine.push_back(tmp);
				file.get(temp);
			} while((temp !='\n') && (!file.eof()));
			if (i < trainData)
			{
				train.push_back(oneLine);
			}
			else
			{
				test.push_back(oneLine);
			}
			i++;
		}
	}
}

std::vector<std::vector<double>> &Data::getTrainData()
{
	return train;
}

std::vector<std::vector<double>> &Data::getTestData()
{
	return test;
}

Data::Data(std::string filename, double splitData):
	filename(filename)
{
	file.open(filename);
	size = std::count(std::istreambuf_iterator<char>(file),
		std::istreambuf_iterator<char>(), '\n');
	file.close();
	trainData = size * splitData;
	testData = size - trainData;
	readData();
	normalize();
}

void Data::normalize() {
	//musze i dla train i dla test razem, pierwszego wyniku nie brac
	double sum = 0;
	int count = 0;
	//train data
	for (int i = 0; i < train.size(); i++) {
		for (int j = 1; j < train[i].size(); j++) {
			sum += train[i][j];
			++count;
		}
	}
	//test data
	for (int i = 0; i < test.size(); i++) {
		for (int j = 1; j < test[i].size(); j++) {
			sum += test[i][j];
			++count;
		}
	}
	double mean = sum / count;
	
	//standard deviation sum
	double sd_sum = 0;
	int difference = 0;
	//train data
	for (int i = 0; i < train.size(); i++) {
		for (int j = 1; j < train[i].size(); j++) {
			difference = train[i][j] - mean;
			sd_sum += (difference*difference);
		}
	}
	//test data
	for (int i = 0; i < test.size(); i++) {
		for (int j = 1; j < test[i].size(); j++) {
			difference = test[i][j] - mean;
			sd_sum += (difference*difference);
		}
	}
	double standardDeviation = std::sqrt((sd_sum/count));

	//normalization
	//train data
	for (int i = 0; i < train.size(); i++) {
		for (int j = 1; j < train[i].size(); j++) {
			train[i][j] = (train[i][j] - mean) / standardDeviation;
		}
	}
	//test data
	for (int i = 0; i < test.size(); i++) {
		for (int j = 1; j < test[i].size(); j++) {
			test[i][j] = (test[i][j] - mean) / standardDeviation;
		}
	}
}

Data::~Data()
{
}
