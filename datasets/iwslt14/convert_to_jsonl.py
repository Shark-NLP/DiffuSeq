import json
import re

def clean_text(text):
    # Remove special characters such as @@ and \u<numbers>
    text = re.sub(r'@@ ', '', text)
    # text = re.sub(r'\\u\d{4}', '', text)
    return text.strip()

def read_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return file.readlines()

def write_jsonl(filename, data):
    with open(filename, 'w', encoding='utf-8') as file:
        for item in data:
            json.dump(item, file, ensure_ascii=False)
            file.write('\n')

def process_and_write(input_de, input_en, output_jsonl):
    de_lines = read_file(input_de)
    en_lines = read_file(input_en)

    combined_data = []
    # max_length_de = 0
    # max_length_en = 0
    for de_line, en_line in zip(de_lines, en_lines):
        # Clean up German and English text
        de_line = clean_text(de_line)
        en_line = clean_text(en_line)
        # max_length_de = max(max_length_de, len(de_line.split()))
        # max_length_en = max(max_length_en, len(en_line.split()))
        combined_data.append({'src': de_line, 'trg': en_line})
    # print(f"Max length of German sequence: {max_length_de}")
    # print(f"Max length of English sequence: {max_length_en}")
    write_jsonl(output_jsonl, combined_data)

    
process_and_write('iwslt14.tokenized.de-en/train.de', 'iwslt14.tokenized.de-en/train.en', 'train.jsonl')
process_and_write('iwslt14.tokenized.de-en/test.de', 'iwslt14.tokenized.de-en/test.en', 'test.jsonl')
process_and_write('iwslt14.tokenized.de-en/valid.de', 'iwslt14.tokenized.de-en/valid.en', 'valid.jsonl')
