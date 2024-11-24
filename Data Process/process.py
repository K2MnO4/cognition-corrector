from datasets import load_dataset
import jsonlines


def convert_pub_data(ds):
    with jsonlines.open("Data Process/Datasets/pubmedqa/pub_data.jsonl", mode='w') as writer:
        for item in ds:
            writer.write({"id": item["pubid"],
                          "url": '',
                          "subject": '',
                          "topic": '',
                          "context": " ".join(item["context"]["contexts"]),
                          "question": item["question"],
                          "answer": item['final_decision'] + ". "+ item['long_answer'],})
if __name__ == "__main__":
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled")["train"]
    convert_pub_data(ds)