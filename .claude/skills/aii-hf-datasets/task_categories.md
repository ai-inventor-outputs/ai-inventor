# HuggingFace Dataset Task Categories

This file contains all available task categories, modalities, formats, and languages for filtering datasets.

## Task Categories

Use these with `--tags "task_categories:<category>"`:

### Text Tasks
- `text-classification` - Classify text into categories
- `text-generation` - Generate text (LLM, completion)
- `translation` - Translate between languages
- `summarization` - Text summarization
- `question-answering` - Question answering datasets
- `text-retrieval` - Information retrieval
- `sentence-similarity` - Semantic similarity
- `token-classification` - NER, POS tagging
- `fill-mask` - Masked language modeling
- `zero-shot-classification` - Zero-shot text classification

### Code Tasks
- `text-generation` - Code generation (use with language:en or code-related tags)

### Vision Tasks
- `image-classification` - Image categorization
- `image-segmentation` - Segment images
- `image-to-text` - Image captioning
- `object-detection` - Detect objects in images
- `depth-estimation` - Depth prediction
- `image-text-to-text` - Visual instruction following
- `text-to-image` - Text to image generation
- `visual-question-answering` - VQA datasets

### Audio Tasks
- `audio-classification` - Audio categorization
- `automatic-speech-recognition` - Speech to text (ASR)
- `text-to-speech` - TTS datasets
- `audio-to-audio` - Audio processing

### Video Tasks
- `video-classification` - Video categorization
- `video-text-to-text` - Video instruction following
- `image-to-video` - Image to video generation
- `text-to-video` - Text to video generation

### Multimodal Tasks
- `visual-question-answering` - Answer questions about images
- `document-question-answering` - Document QA
- `table-question-answering` - Table-based QA
- `visual-document-retrieval` - Document retrieval with images

### Other Tasks
- `multiple-choice` - Multiple choice QA
- `reinforcement-learning` - RL datasets
- `robotics` - Robotics datasets
- `feature-extraction` - Embeddings/features

## Modalities

Use with `--tags "modality:<type>"`:
- `text`
- `image`
- `audio`
- `video`
- `tabular`
- `3d`
- `document`
- `timeseries`

## Common Formats

Use with `--tags "format:<type>"`:
- `json`
- `parquet`
- `csv`
- `imagefolder`
- `webdataset`
- `arrow`

## Languages

Use with `--tags "language:<code>"`:
- `en` - English
- `zh` - Chinese
- `es` - Spanish
- `fr` - French
- `de` - German
- `ja` - Japanese
- `ko` - Korean
- `ar` - Arabic
- `ru` - Russian
- `pt` - Portuguese
- `it` - Italian
- `nl` - Dutch
- `hi` - Hindi
- And many more...

## Size Categories

Use with `--tags "size_categories:<range>"`:
- `n<1K` - Less than 1,000 samples
- `1K<n<10K` - 1,000 to 10,000 samples
- `10K<n<100K` - 10,000 to 100,000 samples
- `100K<n<1M` - 100,000 to 1 million samples
- `1M<n<10M` - 1 million to 10 million samples
- `10M<n<100M` - 10 million to 100 million samples
- `100M<n<1B` - 100 million to 1 billion samples
- `n>1B` - More than 1 billion samples

## Combining Tags

You can combine multiple tags with commas:

```bash
# English translation datasets in parquet format
--tags "task_categories:translation,language:en,format:parquet"

# Large image classification datasets
--tags "task_categories:image-classification,size_categories:1M<n<10M"

# Text generation datasets sorted by popularity
--tags "task_categories:text-generation,modality:text" --sort downloads
```
