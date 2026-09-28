#pragma once
#include "layer.hpp"
#include "loss.hpp"
#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <initializer_list>

class NeuralNetwork {
public:
    // layer_sizes[0] — входной размер, далее размеры слоёв
    NeuralNetwork(std::initializer_list<std::size_t> layer_sizes);

    Eigen::VectorXd predict(const Eigen::VectorXd& input);

    // Один шаг обучения: forward -> loss -> backward -> обновление весов
    double train_step(const Eigen::VectorXd& input, const Eigen::VectorXd& target,
                      double learning_rate);

    void print_structure() const;

private:
    std::vector<std::unique_ptr<Layer>> layers_;
    std::size_t input_size_;
};