import json

if __name__=="__main__":
    with open("output/experiment_results_1.json", 'r', encoding='utf-8') as f:
            results = json.load(f)

    # print(json.dumps(results['texts'][1], indent=4))

    for t in results['texts'][0]['iterations']:
          print(t)
          print(f"{len(t.split())} words")