#include "network.hpp"
#include "dataset.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <string>
#include <sstream>

#ifdef _WIN32
#include <windows.h>
#endif

int main() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8); // консоль в UTF-8
#endif

    // --- 1. Загружаем датасет из файла ---
    // train_data — примеры, на которых сеть УЧИТСЯ
    // test_data  — примеры, на которых мы только ПРОВЕРЯЕМ (сеть их не видела)
    Dataset data = load_dataset("data/multiplication.csv");
    std::cout << "Zagruzheno primerov: " << data.size() << '\n';

    // --- 2. Перемешиваем и разбиваем: 90% на обучение, 10% на проверку ---
    std::mt19937 gen(42); // фиксированный seed — воспроизводимое разбиение
    std::shuffle(data.begin(), data.end(), gen);
    std::size_t test_size = data.size() / 10;
    Dataset test_data(data.end() - test_size, data.end());   // последние 10%
    Dataset train_data(data.begin(), data.end() - test_size); // остальные 90%

    // --- 3. Создаём сеть: 2 входа -> 32 -> 32 -> 1 выход ---
    // Умножение сложнее XOR, поэтому слоёв больше
    NeuralNetwork net{2, 32, 32, 1};
    net.print_structure();

    // --- 4. Цикл обучения ---
    for (int epoch = 0; epoch < 10000; ++epoch) {
        double total_loss = 0.0;
        for (auto& sample : train_data) {
            Eigen::VectorXd& x = sample.first;   // вход (a, b)
            Eigen::VectorXd& y = sample.second;  // правильный ответ (a*b)
            total_loss += net.train_step(x, y, 0.05); // lr = 0.05
        }
        if (epoch % 500 == 0)
            std::cout << "Epocha " << epoch << ", loss = "
                      << total_loss / train_data.size() << '\n';
    }

    // --- 5. Проверка на НЕВИДЕННЫХ примерах ---
    std::cout << "\nProverka (test examples, kotorych ne bylo v treninge):\n";
    double test_loss = 0.0;
    for (auto& sample : test_data) {
        Eigen::VectorXd out = net.predict(sample.first);
        test_loss += 0.5 * (out - sample.second).squaredNorm();
        std::cout << sample.first[0] << " * " << sample.first[1]
                  << " = " << out[0] << "  (верно: " << sample.second[0] << ")\n";
    }
    std::cout << "Srednyaya loss na teste: " << test_loss / test_data.size() << '\n';

    // --- 6. Интерактивный калькулятор ---
    std::string line;
    std::cout << "\n--- Interaktivny kalkulyator ---\n";
    std::cout << "Vvedite dva chisla cherez probel (ili 'q' dlya vyhoda):\n";

    while (true) {                    // бесконечный цикл, выходим через break
        std::cout << "> ";
        std::cout.flush();            // гарантируем, что приглашение видно ДО ввода

        // Читаем ВСЮ строку, введённую пользователем
        if (!std::getline(std::cin, line)) break;  // EOF (Ctrl+Z) или ошибка -> выход

        // Выход по 'q'
        if (line == "q" || line == "Q") break;

        // Разбираем строку на два числа
        std::stringstream ss(line);
        double a, b;
        if (!(ss >> a >> b)) {        // ввод не распарсился -> подсказка и заново
            std::cout << "Oshibka vvoda. Primer: 0.3 0.4   ili   q\n";
            continue;                 // пропускаем итерацию, идём к началу цикла
        }

        // Нормализация: делим на 10 — ТО ЖЕ масштабирование, что в dataset.hpp!
        Eigen::VectorXd x(2);
        x << a / 1, b / 1;

        Eigen::VectorXd out = net.predict(x);

        // Денормализация: (a/10)*(b/10) = a*b/100 -> умножаем обратно на 100
        double result = out[0] * 1;

        std::cout << a << " * " << b << " = " << result << '\n';
    }

    std::cout << "Poka!\n";
}