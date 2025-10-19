class JsonProcessor:
    def __init__(self, input_json_path, output_json_path):
        self.input_json_path = input_json_path
        self.output_json_path = output_json_path

    def read_json(self):
        with open(self.input_json_path, 'r', encoding='utf-8') as file:
            return json.load(file)

    def save_json(self, data):
        with open(self.output_json_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

    def process_json(self):
        data = self.read_json()
        # Modify the data if necessary
        self.save_json(data)