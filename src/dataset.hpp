#pragma once
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <utility>

// Один обучающий пример: вход (a, b) -> желаемый выход (произведение)
// first = входной вектор, second = целевой вектор
using Sample  = std::pair<Eigen::VectorXd, Eigen::VectorXd>;
using Dataset = std::vector<Sample>;

// Читает CSV формата "a;b;product" и возвращает набор примеров
Dataset load_dataset(const std::string& path);