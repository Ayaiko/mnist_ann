#pragma once
#include <cmath>
#include <iostream>
#include <random>
#include <vector>
#include <string>
#include "vector_utils.hpp"
#include "activation_function.hpp"


using namespace std;

class Layer 
{
public:

	virtual vector<float> forward_propagation(vector<float> input, string output = "none") = 0;

	virtual vector<float> backward_propagation() = 0;

	//now using he initialization
	void weight_initiliazation(vector<vector<float>>& weight, int connected_neuron)
	{
		random_device rd;
		mt19937 gen(rd());

		float variance = 2.0f / connected_neuron;

		for (int i = 0; i < weight.size(); i++)
		{
			//resize inner vector
			weight[i].resize(connected_neuron);
			for (int j = 0; j < connected_neuron; j++)
			{
				// Sample weight from Gaussian distribution with mean 0 and variance 2 / n
				normal_distribution<float> distribution(0.0f, std::sqrt(variance));
				weight[i][j] = distribution(gen);
			}
		}
	}
};

class InputLayer : public Layer 
{
private:
	vector<uint8_t> neuron_unit;
	size_t flatten_size;

public:
	InputLayer(vector<int> shape) 
	{
		this->flatten_size = vector_opt::element_product(shape);
		this->neuron_unit.reserve(flatten_size);
	}

	vector<float> forward_propagation(vector<float> input, string output = "none") override
	{
		return input;
	}

	vector<float> backward_propagation() override
	{
		return {};
	}
};

class Dense : public Layer {
private:
	vector<vector<float>> weight;
	string activation_function;
	int neuron;


public:

	Dense(int neuron = 1, string afunction = "relu") 
	{
		this->neuron = neuron;
		this->weight.resize(neuron);
		this->activation_function = afunction;
	}

	vector<float> forward_propagation(vector<float> input, string output = "none") override
	{
		weight_initiliazation(this->weight, input.size());

		if (output == "raw") return activation_function::weighted_sum(weight, input);
		return activation_function::apply_activation(activation_function, weight, input);
	}

	vector<float> backward_propagation() override
	{
		return {};
	}

};