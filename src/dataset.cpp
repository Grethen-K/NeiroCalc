#include "dataset.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>

// Разбивает строку по разделителю: "0.10;0.20;0.0200" -> ["0.10", "0.20", "0.0200"]
static std::vector<std::string> split(const std::string& s, char delim) {
    std::vector<std::string> parts;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) parts.push_back(item);
    return parts;
}

Dataset load_dataset(const std::string& path) {
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("Не удалось открыть файл датасета: " + path);

    Dataset data;
    std::string line;
    std::getline(in, line); // пропускаем строку-заголовок "a;b;product"

    while (std::getline(in, line)) {
        if (line.empty()) continue; // пропускаем пустые строки
        auto parts = split(line, ';');
        if (parts.size() != 3)
            throw std::runtime_error("Битая строка в датасете: " + line);
        double a = std::stod(parts[0]);
        double b = std::stod(parts[1]);
        double p = std::stod(parts[2]);
        // Вход — вектор (a, b), цель — вектор (произведение)
        data.emplace_back((Eigen::VectorXd(2) << a, b).finished(),
                          (Eigen::VectorXd(1) << p).finished());
    }
    return data;
}