# Chinese-News-Summarization

## 📚 Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Inference](#inference)
- [Acknowledgements](#acknowledgements)
- [License](#license)
- [Contact](#contact)

## 🌟 Overview
The Chinese-News-Summarization project focuses on generating concise and coherent summaries from Chinese news articles. With the massive amount of content published daily, summarization is essential for improving the readability and accessibility of news. This project utilizes pre-trained models from Hugging Face's Transformers library to perform abstractive summarization specifically for Chinese text.

## 💻 Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/Shengwei-Peng/Chinese-News-Summarization.git
    ```
2. Navigate to the project directory:
    ```sh
    cd Chinese-News-Summarization
    ```
3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## ⚙️ Usage

Once you've installed the dependencies, you can start using the summarization model for your own dataset.
### 1. **Prepare the Dataset:**
   - Ensure your Chinese news articles are properly formatted in a `.jsonl` file.
   - The format should be as follows:
        ```json
        {
        "date_publish": "2015-03-02 00:00:00",
        "title": "榜首進台大醫科創休學 27歲拿到法國天文博士 李悅寧破跟蹤人眼鏡返台任教",
        "source_domain": "udn.com",
        "maintext": "從小就很會念書的李悅寧，在眾人殷殷期盼下，以榜首之姿進入臺大醫學院，但始終忘不了對天文的熱情。..."
        }
        ```

### 2. **Training, Validation, Testing, and Plotting:**

After preparing your dataset, you can train, validate, test the model, and generate plots to evaluate its performance. You can combine all these steps into a single command as follows:

```bash
python main.py \
    --model_name_or_path google/mt5-small \
    --train_file ./data/train.jsonl \
    --validation_file ./data/public.jsonl \
    --test_file ./data/sample_test.jsonl \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --per_device_eval_batch_size 8 \
    --max_source_length 1024 \
    --max_target_length 128 \
    --num_train_epochs 3 \
    --learning_rate 5e-5 \
    --seed 11207330 \
    --model_type mt5 \
    --output_dir ./mt5 \
    --prediction_path ./predictions.jsonl \
    --plot
```

#### ⚠️ Special Note:
If you enable the `--plot`option, the script will calculate the ROUGE score and visualize the score curves. Please ensure the following additional libraries are installed:
```bash
pip install matplotlib tensorflow
```

### 3. Training the Model:
If you prefer to train the model without validation or testing in a simplified manner, you can use the following commands for **mT5** or **GPT-2** models:

- #### mT5:
    Use the following command to train the model with the mT5 model:
    ```bash
    python main.py \
        --model_name_or_path google/mt5-small \
        --train_file ./data/train.jsonl \
        --per_device_train_batch_size 8 \
        --gradient_accumulation_steps 1 \
        --max_source_length 512 \
        --max_target_length 128 \
        --num_train_epochs 3 \
        --learning_rate 5e-5 \
        --seed 11207330 \
        --model_type mt5 \
        --output_dir ./mt5
    ```

- #### GPT-2:
    Alternatively, you can use the GPT-2 model for training:
    ```bash
    python main.py \
        --model_name_or_path ckiplab/gpt2-base-chinese \
        --train_file ./data/train.jsonl \
        --per_device_train_batch_size 8 \
        --gradient_accumulation_steps 1 \
        --max_source_length 512 \
        --max_target_length 128 \
        --num_train_epochs 3 \
        --learning_rate 5e-5 \
        --seed 11207330 \
        --model_type gpt2 \
        --output_dir ./gpt2
    ```

## 🔮 Inference

To perform inference on new data after training the model, you can use the following command.
```bash
python main.py \
    --model_name_or_path ./mt5-small \
    --test_file ./data/sample_test.jsonl \
    --per_device_eval_batch_size 8 \
    --max_source_length 2048 \
    --max_target_length 128 \
    --seed 11207330 \
    --model_type mt5 \
    --prediction_path predictions.jsonl \
    --num_beams 5 \
    --top_k 50 \
    --top_p 0.9 \
    --temperature 0.7 \
    --do_sample
```

### Generation Strategies

- **Beam Search (`--num_beams`)**: Controls the number of beams for beam search. Higher values improve search diversity but increase computation. Setting it to 1 disables beam search (**Greedy decoding**).

- **Top-k Sampling (`--top_k`)**: Limits token selection to the top-k highest probability tokens. A higher value of k provides more diversity in output.

- **Top-p Sampling (`--top_p`)**: Selects tokens from the top cumulative probability p (between 0 and 1). A lower p results in more focused and deterministic outputs.

- **Temperature (`--temperature`)**: Adjusts the randomness of predictions. A value < 1 makes the model more conservative, while values > 1 make outputs more random.

- **Sampling (`--do_sample`)**: Enables stochastic sampling. If not set, the model will default to greedy decoding (always selecting the highest probability token).

These parameters can be adjusted together to fine-tune how creative or focused the model's generated output will be during inference.


## 🙏 Acknowledgements

This project is based on the example code provided by Hugging Face in their [Transformers repository](https://github.com/huggingface/transformers/tree/main/examples/pytorch). We have made modifications to adapt the code for our specific use case.

Special thanks to the [NTU Miulab](http://adl.miulab.tw) professors and teaching assistants for providing the dataset and offering invaluable support throughout the project.

## ⚖️ License

This project is licensed under the Apache License 2.0. See the [LICENSE](./LICENSE) file for more details.

## ✉️ Contact

For any questions or inquiries, please contact m11207330@mail.ntust.edu.tw