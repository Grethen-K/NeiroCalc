#pragma once
#include <Eigen/Dense>
#include <stdexcept>

// Среднеквадратичная ошибка: L = ½·‖pred − target‖²
class MSELoss {
public:
    static double compute(const Eigen::VectorXd& pred, const Eigen::VectorXd& target) {
        if (pred.size() != target.size())
            throw std::invalid_argument("Razmery pred i target dolzhny sovpadat'");
        return 0.5 * (pred - target).squaredNorm();
    }
    // Градиент по предсказанию: dL/dpred = pred − target
    static Eigen::VectorXd gradient(const Eigen::VectorXd& pred, const Eigen::VectorXd& target) {
        return pred - target;
    }
};