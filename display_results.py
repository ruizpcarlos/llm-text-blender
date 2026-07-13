import json
import sys

text_id = int(sys.argv[1])

with open("output/experiment_results.json", 'r', encoding='utf-8') as f:
    results = json.load(f)

text = results['texts'][text_id]

# print(json.dumps(results['texts'][1], indent=4))

print(text['original'])
print('----------')

for k, txt in enumerate(text['iterations']):
    print(f"----- k={k} ------")
    print(txt)
    print(f"{len(txt.split())} words")