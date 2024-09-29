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
    
### 2. **Training the Model:**

You can choose between using **mT5** or **GPT-2** for your summarization model. Below are the commands for training both models:

- #### mT5:
    Use the following command to train the model with the mT5 model:
    ```bash
    python main.py \
        --model_name_or_path google/mt5-small \
        --train_file ./data/train.jsonl \
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
        --strategy greedy \
        --output_dir ./mt5 \
        --prediction_path ./mt5/prediction.jsonl
    ```

- #### GPT-2:
    Alternatively, you can use the GPT-2 model for training:
    ```bash
    python main.py \
        --model_name_or_path gpt2-base-chinese \
        --train_file ./data/train.jsonl \
        --test_file ./data/sample_test.jsonl \
        --per_device_train_batch_size 8 \
        --gradient_accumulation_steps 1 \
        --per_device_eval_batch_size 8 \
        --max_source_length 1024 \
        --max_target_length 128 \
        --num_train_epochs 3 \
        --learning_rate 5e-5 \
        --seed 11207330 \
        --model_type gpt2 \
        --strategy greedy \
        --output_dir ./gpt2 \
        --prediction_path ./gpt2/prediction.jsonl
    ```

#### ⚠️ Special Note:
If you want to visualize the training metrics like loss curves or calculate the ROUGE score for evaluation, you can add the `--plot` flag to the above commands.

- To enable plotting, make sure you have the necessary libraries installed:
  ```bash
  pip install matplotlib tensorflow
  ```
- Additionally, you must provide a `--validation_file` to compute the ROUGE score during evaluation.
## 🔮 Inference

## 🙏 Acknowledgements

This project is based on the example code provided by Hugging Face in their [Transformers repository](https://github.com/huggingface/transformers/tree/main/examples/pytorch). We have made modifications to adapt the code for our specific use case.

Special thanks to the [NTU Miulab](http://adl.miulab.tw) professors and teaching assistants for providing the dataset and offering invaluable support throughout the project.

## ⚖️ License

This project is licensed under the Apache License 2.0. See the [LICENSE](./LICENSE) file for more details.

## ✉️ Contact

For any questions or inquiries, please contact m11207330@mail.ntust.edu.tw