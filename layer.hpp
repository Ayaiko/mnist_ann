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
protected:
	vector<vector<float>> weights; // weight connected to previous layer 
	vector<float> input;
	
public:

	virtual vector<float> forward_propagation(vector<float> input, string output = "none") = 0;

	virtual vector<vector<float>> backward_propagation(vector<vector<float>>& input, float gradient = 1) = 0;

	void weight_update(vector<vector<float>>& gradients, float learning_rate = 0.1)
	{
		for (int i = 0; i < weights.size(); ++i) {
			for (int j = 0; j < weights[0].size(); ++j) {
				// Subtract the gradient from the weights
				weights[i][j] -= learning_rate * gradients[i][j];
			}
		}
	}

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

	vector<vector<float>> backward_propagation(vector<vector<float>>& input, float gradient = 1) override
	{
		return {};
	}

};

class Dense : public Layer {
private:
	size_t input_size;
	vector<float> logits;
	string activation_function;
	int neuron;
	bool weights_initialized = false;

public:

	Dense(int neuron = 1, string afunction = "relu") 
	{
		this->neuron = neuron;
		this->weights.resize(neuron);
		this->activation_function = afunction;
	}

	vector<float> forward_propagation(vector<float> input, string output = "none") override
	{
		this->input = input;
		input_size = input.size();

		if (!weights_initialized) {
			weight_initiliazation(weights, input.size());
			weights_initialized = true;
		}

		logits = activation_function::weighted_sum(weights, input);
		if (output == "raw") return logits;
		return activation_function::apply_activation(activation_function, logits);
	}

	vector<vector<float>> backward_propagation(vector<vector<float>>& input, float loss_gradient = 1) override
	{
		//loss gradient from respect to output of output layer if not the output layer it's 1
		vector<float> dadz;
		vector<vector<float>> gradients(neuron, vector<float>(input_size));

		//find differentiation of a(activation function) respect to z(logit)
		dadz = activation_function::apply_activation_grad(activation_function, logits, loss_gradient) ;
		
		//find differentiation of z(logit) respect to each weight
		for (int i = 0; i < neuron; i++)
		{
			for (int j = 0; j < input_size; j++)
			{
				//fix here
				gradients[i][j] = dadz[i] * this->input[j];
			}
		}

		return gradients;
	}
};