# tools/gen_dataset.py — генератор обучающего набора для умножения
# Формат строки CSV: "a;b;product"  (разделитель — точка с запятой)
with open("data/multiplication.csv", "w", encoding="utf-8") as f:
    f.write("a;b;product\n")          # заголовок — его пропускаем при чтении
    for i in range(21):               # a = 0.00, 0.05, ..., 1.00
        for j in range(21):           # b = 0.00, 0.05, ..., 1.00
            a, b = i * 0.05, j * 0.05
            f.write(f"{a:.2f};{b:.2f};{a*b:.4f}\n")
    print("Готово: data/multiplication.csv, 441 пример")