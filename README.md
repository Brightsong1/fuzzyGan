# Запуск фаззинга OpenSSL

## 1. Требования
- Linux, Bash, Python 3.9+ (Pytorch)
- Docker (для сборки через OSS-Fuzz)

## 2. Клонирование и инициализация форка oss-fuzz
```bash
git clone https://example.com/your/fuzzygan.git
cd fuzzygan
git submodule update --init --recursive
```

## 3. Восстановление корпусов и summary
Архив `oss-fuzz/projects/openssl/openssl-support.tar.gz` содержит corpus/analysis_summary и исходники скриптов. Распакуйте его **из корня репозитория**, чтобы восстановить правильные пути:
```bash
tar -xzf oss-fuzz/projects/openssl/openssl-support.tar.gz -C .
```
После распаковки убедитесь, что появился файл `fuzz_out` 

## 4. Подготовка OSS-Fuzz окружения
```bash
python3 oss-fuzz/infra/helper.py build_image --pull openssl
```

## 5. Сборка фаззеров
```bash
python3 oss-fuzz/infra/helper.py build_fuzzers --sanitizer address openssl
```

## 6. Циклический фаззинг (10 минут на итерацию)
Скрипт `run_openssl_cycle.sh`. Он вызывает `vae_fuzzing.py` с уже подготовленными путями и задаёт по умолчанию `FUZZ_SECONDS=600` (10 минут на каждую эпоху/итерацию).

Базовый запуск:
```bash
./run_openssl_cycle.sh
```

Что происходит:
1. Скрипт читает `fuzz_out/openssl/analysis_summary.json`.
2. Проходит по всем функциям, отмеченным как worth_fuzzing, в цикле (`CYCLES=1000` по умолчанию).
3. Каждая эпоха фаззера выполняется 600 секунд. Значения можно переопределить переменными окружения:
   ```bash
   CYCLES=50 FUZZ_SECONDS=900 ./run_openssl_cycle.sh --functions ocsp,ess
   ```
4. Логи и статистика пишутся в `fuzz_out/fuzzygan.db`. Остановить цикл можно `Ctrl+C`.


