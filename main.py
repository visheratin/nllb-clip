from open_clip import create_model_and_transforms
from transformers import AutoTokenizer

from experiment import Experiment

if __name__ == "__main__":
    # test base model
    model, _, transform = create_model_and_transforms(
        "nllb-clip-base", "v1", device="cuda"
    )
    tokenizer = AutoTokenizer.from_pretrained("visheratin/nllb-clip-base-oc")
    experiment = Experiment(512, "nllb-clip-base", "./results")
    experiment.run(model, transform, tokenizer)
    experiment.save()

    # test large model
    model, _, transform = create_model_and_transforms(
        "nllb-clip-large", "v1", device="cuda"
    )
    tokenizer = AutoTokenizer.from_pretrained("visheratin/nllb-clip-large-oc")
    experiment = Experiment(1024, "nllb-clip-large", "./results")
    experiment.run(model, transform, tokenizer)
    experiment.save()
