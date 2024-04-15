The original cross-KG datasets (FB15K-DB15K/YAGO15K) comes from MMKB, in which the image embeddings are extracted from the pre-trained VGG16.

	Original dataset: MMKB：https://github.com/mniepert/mmkb
	The converted dataset and Image embedding:
		https://pan.baidu.com/s/1MLGBNyFjb9LLa4urCk4hCA; key:stdt
	PLM model:
		https://huggingface.co/models
		Bert: https://huggingface.co/bert-base-uncased
		Roberta: https://huggingface.co/xlm-roberta-base
		Albert: https://huggingface.co/albert-base-v2
		T5: https://huggingface.co/albert-base-v2
		ChatPLM-6B: https://huggingface.co/THUDM/chatglm2-6b
		LLaMA-7B: https://huggingface.co/shalomma/llama-7b-embeddings

1. Data preparation
	Download original dataset or converted dataset;
	Generate the PLM embedding of relation and attribute triples using "data_process" folder.

2.Training PCMEA:
	Here is the example of training PCMEA on FB15K-DB15K and FB15K-YAGO15K.
	bash run_PCMEA.sh 0 42 FB15K_DB15K 0.2
	bash run_PCMEA.sh 0 42 FB15K_DB15K 0.5
	bash run_PCMEA.sh 0 42 FB15K_DB15K 0.8

	bash run_PCMEA.sh 0 42 FB15K-YAGO15K 0.2
	bash run_PCMEA.sh 0 42 FB15K-YAGO15K 0.5
	bash run_PCMEA.sh 0 42 FB15K-YAGO15K 0.8

