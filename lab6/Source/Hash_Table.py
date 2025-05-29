from constants import RUSSIAN_ALPHABET

class HashTable:
    def __init__(self, size, base_address):
        self.size = size
        self.base_address = base_address
        self.table = [{
            'ID': None,
            'C': False,
            'U': False,
            'T': False,
            'L': False,
            'D': False,
            'Po': None,
            'Pi': None
        } for _ in range(size)]

        self.alphabet = {letter: idx for idx, letter in enumerate(RUSSIAN_ALPHABET)}

    def _calculate_key_value(self, key):
        if len(key) < 2:
            key = key.ljust(2, 'А')

        first_char = key[0].upper()
        second_char = key[1].upper()

        first_num = self.alphabet.get(first_char, 0)
        second_num = self.alphabet.get(second_char, 0)

        return first_num * 33 + second_num

    def _hash(self, value):
        return (value % self.size) + self.base_address - self.base_address % self.size

    def _double_hash(self, value, attempt):
        return (self._hash(value) + attempt * (1 + value % (self.size - 1))) % self.size

    def insert_in_table(self, key, data):
        value = self._calculate_key_value(key)
        attempt = 0
        index = self._double_hash(value, attempt)

        while attempt < self.size:
            cell = self.table[index]

            if not cell['U'] or cell['ID'] == key:
                cell['ID'] = key
                cell['U'] = True
                cell['Pi'] = data
                cell['L'] = False

                if attempt > 0:
                    prev_index = self._double_hash(value, attempt - 1)
                    self.table[prev_index]['C'] = True
                    self.table[prev_index]['Po'] = index

                return True

            attempt += 1
            index = self._double_hash(value, attempt)

        return False

    def search_in_table(self, key):
        value = self._calculate_key_value(key)
        attempt = 0
        index = self._double_hash(value, attempt)

        while attempt < self.size:
            cell = self.table[index]

            if cell['ID'] == key and cell['U']:
                return cell['Pi']

            if not cell['C']:
                return None

            attempt += 1
            index = self._double_hash(value, attempt)

        return None

    def update_by_key(self, key, new_data):
        value = self._calculate_key_value(key)
        attempt = 0
        index = self._double_hash(value, attempt)

        while attempt < self.size:
            cell = self.table[index]

            if cell['ID'] == key and cell['U']:
                cell['Pi'] = new_data
                return True

            if not cell['C']:
                return False

            attempt += 1
            index = self._double_hash(value, attempt)

        return False

    def delete_from_table(self, key):
        index, prev_index = self._find_key_index_and_previous(key)
        if index is None:
            return False

        if self._is_last_in_chain(index):
            self._delete_last_in_chain(index, prev_index)
        else:
            self._delete_and_promote_next(index)

        return True

    def _find_key_index_and_previous(self, key):
        value = self._calculate_key_value(key)
        attempt = 0
        index = self._double_hash(value, attempt)
        prev_index = None

        while attempt < self.size:
            cell = self.table[index]

            if cell['ID'] == key and cell['U']:
                return index, prev_index

            if not cell['C']:
                return None, None

            prev_index = index
            attempt += 1
            index = self._double_hash(value, attempt)

        return None, None

    def _is_last_in_chain(self, index):
        return self.table[index]['Po'] is None

    def _delete_last_in_chain(self, index, prev_index):
        cell = self.table[index]
        cell['ID'] = None
        cell['U'] = False
        cell['C'] = False
        cell['Pi'] = None

        if prev_index is not None:
            self.table[prev_index]['C'] = False
            self.table[prev_index]['Po'] = None

    def _delete_and_promote_next(self, index):
        cell = self.table[index]
        next_in_chain = cell['Po']
        next_cell = self.table[next_in_chain]

        cell['ID'] = next_cell['ID']
        cell['Pi'] = next_cell['Pi']
        cell['Po'] = next_cell['Po']

        next_cell['ID'] = None
        next_cell['U'] = False
        next_cell['C'] = False
        next_cell['Pi'] = None
        next_cell['Po'] = None

    def display_table(self):
        print("Хеш-таблица:")
        print("Индекс | ID       | U | C | Po | Данные")
        for i, cell in enumerate(self.table):
            print(
                f"{i:6} | {cell['ID'] if cell['ID'] is not None else 'None':8} | "
                f"{int(cell['U'])} | {int(cell['C'])} |"
                f" {cell['Po'] if cell['Po'] is not None else 'None':3} |"
                f" {cell['Pi']}")
