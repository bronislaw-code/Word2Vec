import pickle
# Importujemy TensorFlow tylko na Twoim komputerze, żeby odczytać stary plik
import tensorflow as tf

print("Ładowanie starego tokenizera...")
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Wyciągamy czysty słownik Pythona (dict)
word_index = tokenizer.word_index
print(f"Wyciągnięto słownik zawierający {len(word_index)} słów.")

# Zapisujemy jako czysty, lekki plik pickle
with open('word_index.pkl', 'wb') as f:
    pickle.dump(word_index, f)

print("🎉 Sukces! Słownik został zapisany jako 'word_index.pkl'")