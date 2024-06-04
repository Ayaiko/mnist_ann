// NeuralNetwork.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "mnist_reader_less.hpp"
#include "model.hpp"
#include <vector>
#include "mnist_utils.hpp"
#include <chrono>
#include "time_measurment.hpp"

using namespace std;

int main()
{
	auto dataset = mnist::read_dataset();
	mnist::normalize_dataset(dataset);

	vector<vector<uint8_t>> data = dataset.training_images;
	vector<uint8_t> label = dataset.training_labels;

	vector<vector<uint8_t>> data_tpack;
	vector<uint8_t> label_tpack;

	data_tpack.push_back(data[1]);
	label_tpack.push_back(label[1]);

	//initiate instance
	Model model;
	InputLayer inputLayer({ 28, 28 });
	Dense hiddenLayer1(5, "relu");
	Dense outputLayer(10, "softmax");

	model.add_layer(&inputLayer);
	model.add_layer(&hiddenLayer1);
	model.add_layer(&outputLayer);

	auto start = chrono::high_resolution_clock::now();
	model.fit(data_tpack, label_tpack);
	//model.fit(data, label);
	auto end = chrono::high_resolution_clock::now();

	tmeasure::duration_measure(start, end);

	// Calculate the duration
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

	
	//testing site
	
	vector<vector<float>> test = model.debug();

	cout << "output size: " << test[0].size() << endl;
	for (auto i : test[0])
	{
		cout << i << endl;
	}


	return 0;
}

