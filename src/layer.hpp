#pragma once
#include "activation.hpp"
 #include <Eigen/Dense>

// Полносвязный слой: y = f(W·x + b)
// W — матрица (num_neurons × num_inputs), строка i = веса i-го нейрона
class Layer {
public:
    Layer(std::size_t num_neurons, std::size_t num_inputs, ActivationType act);

    // Прямой проход: сохраняем промежуточные значения для backprop
    Eigen::VectorXd forward(const Eigen::VectorXd& input);

    // Обратный проход: получаем dL/da (градиент по выходам слоя),
    // сохраняем dW и db, возвращаем dL/dinput для предыдущего слоя
    Eigen::VectorXd backward(const Eigen::VectorXd& dL_da);

    void update_weights(double learning_rate); // шаг градиентного спуска

    std::size_t size() const { return W_.rows(); }

private:
    Eigen::MatrixXd W_;      // веса всех нейронов слоя
    Eigen::VectorXd b_;      // смещения
    ActivationType act_;

    // Кэши прямого прохода (обязательны для backprop):
    Eigen::VectorXd input_;  // вход слоя (= выход предыдущего)
    Eigen::VectorXd z_;      // до активации (W·x + b)
    Eigen::VectorXd a_;      // после активации

    // Градиенты, накопленные в backward:
    Eigen::MatrixXd dW_;
    Eigen::VectorXd db_;
};