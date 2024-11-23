# from transformers import WhisperProcessor, WhisperForConditionalGeneration
# from datasets import load_dataset, Audio
#
# model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny", low_cpu_mem_usage=True)
# processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
#
# model.to("cuda")
#
# common_voice = load_dataset("mozilla-foundation/common_voice_16_1", "fa", split="validation", streaming=True)
# common_voice = common_voice.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))
#
# inputs = processor(next(iter(common_voice))["audio"]["array"], sampling_rate=16000, return_tensors="pt")
# input_features = inputs.input_features
#
# generated_ids = model.generate(input_features.to("cuda"), max_new_tokens=128)
# pred_text = processor.decode(generated_ids[0], skip_special_tokens=True)
#
# print("Pred text:", pred_text)
# print("Environment set up successful?", generated_ids.shape[-1] == 20)


from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import Audio, load_dataset

# load model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-large")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large")
forced_decoder_ids = processor.get_decoder_prompt_ids(language="persian", task="transcribe")

# load streaming dataset and read first audio sample
# ds = load_dataset("common_voice", "fa", split="test", streaming=True)
ds = load_dataset("mozilla-foundation/common_voice_16_1", "fa", split="validation", streaming=True)
ds = ds.cast_column("audio", Audio(sampling_rate=16_000))
for index, input_speech in enumerate(iter(ds)):
    if index == 10:
        break
print(input_speech)
input_speech = input_speech["audio"]
input_features = processor(input_speech["array"],
                           sampling_rate=input_speech["sampling_rate"],
                           return_tensors="pt").input_features

# generate token ids
predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
# decode token ids to text
transcription = processor.batch_decode(predicted_ids)

transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
print(transcription)
