#include "layer.hpp"
#include <random>

Layer::Layer(std::size_t num_neurons, std::size_t num_inputs, ActivationType act)
    : act_(act) {
    std::mt19937 gen(std::random_device{}());
    std::normal_distribution<double> dist(0.0, std::sqrt(2.0 / num_inputs));
    W_.resize(num_neurons, num_inputs);
    for (std::size_t i = 0; i < num_neurons; ++i)
        for (std::size_t j = 0; j < num_inputs; ++j)
            W_(i, j) = dist(gen);
    b_ = Eigen::VectorXd::Zero(num_neurons);
}

Eigen::VectorXd Layer::forward(const Eigen::VectorXd& input) {
    input_ = input;           // кэшируем — без этого backprop невозможен
    z_ = W_ * input + b_;     // линейная часть, одна матричная операция
    a_ = activate(z_, act_);
    return a_;
}

Eigen::VectorXd Layer::backward(const Eigen::VectorXd& dL_da) {
    // Правило цепочки: dL/dz = dL/da ⊙ f'(z)
    Eigen::VectorXd delta = dL_da.cwiseProduct(activate_deriv(z_, act_));

    // Градиенты параметров: dW = delta · inputᵀ,  db = delta
    dW_ = delta * input_.transpose();
    db_ = delta;

    // Распространяем ошибку назад: dL/dinput = Wᵀ · delta
    return W_.transpose() * delta;
}

void Layer::update_weights(double lr) {
    W_ -= lr * dW_;  // шаг градиентного спуска
    b_ -= lr * db_;
}