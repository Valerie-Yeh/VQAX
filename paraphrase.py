from openai import OpenAI
client = OpenAI()
import json
import argparse
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--src_data_path", type=str,
                    help="load nle_data/VQA-X/vqaX_test.json or nle_data/eSNLI-VE/esnlive_test.json dataset")
parser.add_argument("--dst_data_path", type=str,
                    help="store paraphrased results")
args = parser.parse_args()

def get_api_output(input):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
            "role": "system",
            "content": [
                {
                "type": "text",
                "text": "You are a helpful assistant designed to paraphrase sentences."
                }
            ]
            },
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": input
                }
            ]
            },
        ],
        temperature=0,
        max_tokens=60,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
        )
    result = response.choices[0].message.content.strip(" \n")
    result = result.replace('\n', ' ')

    return result

if __name__ == '__main__':

    text_dict = json.load(open(args.src_data_path, "r"))
    text_idx_list = list(text_dict.keys())
    
    for key in tqdm(text_idx_list):
        with open(args.dst_data_path) as f:
            dictObj = json.load(f)

        result = get_api_output(text_dict[key]['question'])
        dictObj.update({key: {"question": result}})

        with open(args.dst_data_path, "w") as file:
            json.dump(dictObj, file, indent=4, separators=(',',': '))