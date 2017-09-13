#pragma once
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>

class Data
{
private:
	std::ifstream file;
	std::string filename;
	double splitData;
	std::vector<std::vector<double>> train;
	std::vector<std::vector<double>> test;
	void readData();
	void normalize();
public:
	std::vector<std::vector<double>> &getTrainData();
	std::vector<std::vector<double>> &getTestData();
	Data(std::string filename, double splitData);
	~Data();
};

