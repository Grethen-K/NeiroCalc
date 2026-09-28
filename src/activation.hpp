#pragma once
#include <cmath>
#include <Eigen/Dense>
#include <stdexcept>

enum class ActivationType { ReLU, Sigmoid, Linear };

// Поэлементная активация для вектора (Eigen вычислит это векторизованно)
inline Eigen::VectorXd activate(const Eigen::VectorXd& z, ActivationType t) {
    switch (t) {
        case ActivationType::ReLU:    return z.cwiseMax(0.0);
        case ActivationType::Sigmoid: return (1.0 / (1.0 + (-z).array().exp())).matrix();
        case ActivationType::Linear:  return z;
    }
    throw std::invalid_argument("Unknown activation type");
}

// Производная активации (нужна в backprop): f'(z)
inline Eigen::VectorXd activate_deriv(const Eigen::VectorXd& z, ActivationType t) {
    switch (t) {
        case ActivationType::ReLU: {
            Eigen::VectorXd d = z;
            d = (d.array() > 0.0).cast<double>().matrix();
            return d;
        }
        case ActivationType::Sigmoid: {
            Eigen::VectorXd s = activate(z, ActivationType::Sigmoid);
            return (s.array() * (1.0 - s.array())).matrix();
        }
        case ActivationType::Linear:
            return Eigen::VectorXd::Ones(z.size());
    }
    throw std::invalid_argument("Unknown activation type");
}