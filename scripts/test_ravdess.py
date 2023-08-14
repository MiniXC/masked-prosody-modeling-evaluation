from datasets import load_dataset

dataset = load_dataset("narad/ravdess", split="train")


def map_emotion(example):
    emotions = [
        "neutral",
        "calm",
        "happy",
        "sad",
        "angry",
        "fearful",
        "disgust",
        "surprised",
    ]
    example["emotion"] = emotions[example["labels"]]
    return example


dataset = dataset.map(map_emotion)
