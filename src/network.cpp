#include "network.hpp"
#include <iostream>

NeuralNetwork::NeuralNetwork(std::initializer_list<std::size_t> layer_sizes) {
    auto it = layer_sizes.begin();
    input_size_ = *it;

    std::size_t prev = input_size_;
    std::size_t total = layer_sizes.size();
    std::size_t idx = 1; // номер текущего слоя (1..n)
    for (++it; it != layer_sizes.end(); ++it, ++idx) {
        // Скрытые слои — ReLU, последний слой — Sigmoid (для задач классификации)
        ActivationType act = (idx == total - 1) ? ActivationType::Sigmoid
                                                : ActivationType::ReLU;
        layers_.push_back(std::make_unique<Layer>(*it, prev, act));
        prev = *it;
    }
}

Eigen::VectorXd NeuralNetwork::predict(const Eigen::VectorXd& input) {
    if (static_cast<std::size_t>(input.size()) != input_size_)
        throw std::invalid_argument("Razmery vkhoda ne sovpadayut s topologiyey seti");
    Eigen::VectorXd out = input;
    for (const auto& layer : layers_)
        out = layer->forward(out);
    return out;
}

double NeuralNetwork::train_step(const Eigen::VectorXd& input,
                                 const Eigen::VectorXd& target,
                                 double learning_rate) {
    Eigen::VectorXd pred = predict(input);                   // 1. forward
    double loss = MSELoss::compute(pred, target);            // 2. ошибка
    Eigen::VectorXd grad = MSELoss::gradient(pred, target);  // 3. dL/dpred
    for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) // 4. backprop
        grad = (*it)->backward(grad);
    for (auto& layer : layers_)                              // 5. обновление весов
        layer->update_weights(learning_rate);
    return loss;
}

void NeuralNetwork::print_structure() const {
    std::cout << "Сетка: " << input_size_;
    for (const auto& l : layers_) std::cout << " -> " << l->size();
    std::cout << '\n';
}