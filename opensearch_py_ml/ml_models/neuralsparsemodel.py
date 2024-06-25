# SPDX-License-Identifier: Apache-2.0
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.
# Any modifications Copyright OpenSearch Contributors. See
# GitHub history for details.

import json
import os
import re
from pathlib import Path
from zipfile import ZipFile

import torch
from mdutils.fileutils import MarkDownFile
from transformers import AutoformerConfig, AutoModelForMaskedLM, AutoTokenizer
from transformers.convert_graph_to_onnx import convert

# from opensearch_py_ml.ml_commons.ml_common_utils import (
#    _generate_model_content_hash_value,
# )
from opensearch_py_ml.ml_models.sparse_model import SparseModel


class NeuralSparseV2Model(SparseModel):
    """
    Class for  exporting and configuring the NeuralSparseV2Model model.
    """

    DEFAULT_MODEL_ID = "opensearch-project/opensearch-neural-sparse-encoding-v1"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        folder_path: str = None,
        overwrite: bool = False,
    ) -> None:
        """
        Initiate a sentence transformer model class object. The model id will be used to download
        pretrained model from the hugging-face and served as the default name for model files, and the folder_path
        will be the default location to store files generated in the following functions

        :param model_id: Optional, the huggingface mode id to download sentence transformer model,
            default model id: 'sentence-transformers/msmarco-distilbert-base-tas-b'
        :type model_id: string
        :param folder_path: Optional, the path of the folder to save output files, such as queries, pre-trained model,
            after-trained custom model and configuration files. if None, default as "/model_files/" under the current
            work directory
        :type folder_path: string
        :param overwrite: Optional,  choose to overwrite the folder at folder path. Default as false. When training
                    different sentence transformer models, it's recommended to give designated folder path every time.
                    Users can choose to overwrite = True to overwrite previous runs
        :type overwrite: bool
        :return: no return value expected
        :rtype: None
        """
        super().__init__(model_id, folder_path, overwrite)
        default_folder_path = os.path.join(
            os.getcwd(), "opensearch_neural_sparse_model_files"
        )
        if folder_path is None:
            self.folder_path = default_folder_path
        else:
            self.folder_path = folder_path
        # Check if self.folder_path exists
        if os.path.exists(self.folder_path) and not overwrite:
            print(
                "To prevent overwriting, please enter a different folder path or delete the folder or enable "
                "overwrite = True "
            )
            raise Exception(
                str("The default folder path already exists at : " + self.folder_path)
            )
        self.model_id = model_id
        self.torch_script_zip_file_path = None
        self.onnx_zip_file_path = None

    def save_as_pt(
        self,
        sentences: [str],
        model_id="opensearch-project/opensearch-neural-sparse-encoding-v1",
        model_name: str = None,
        save_json_folder_path: str = None,
        model_output_path: str = None,
        zip_file_name: str = None,
        add_apache_license: bool = False,
    ) -> str:
        """
        Download sentence transformer model directly from huggingface, convert model to torch script format,
        zip the model file and its tokenizer.json file to prepare to upload to the Open Search cluster

        :param sentences:
            Required, for example  sentences = ['today is sunny']
        :type sentences: List of string [str]
        :param model_id:
            sentence transformer model id to download model from sentence transformers.
            default model_id = "sentence-transformers/msmarco-distilbert-base-tas-b"
        :type model_id: string
        :param model_name:
            Optional, model name to name the model file, e.g, "sample_model.pt". If None, default takes the
            model_id and add the extension with ".pt"
        :type model_name: string
        :param save_json_folder_path:
             Optional, path to save model json file, e.g, "home/save_pre_trained_model_json/"). If None, default as
             default_folder_path from the constructor
        :type save_json_folder_path: string
        :param model_output_path:
             Optional, path to save traced model zip file. If None, default as
             default_folder_path from the constructor
        :type model_output_path: string
        :param zip_file_name:
            Optional, file name for zip file. e.g, "sample_model.zip". If None, default takes the model_id
            and add the extension with ".zip"
        :type zip_file_name: string
        :param add_apache_license:
            Optional, whether to add a Apache-2.0 license file to model zip file
        :type add_apache_license: string
        :return: model zip file path. The file path where the zip file is being saved
        :rtype: string
        """
        bert = AutoModelForMaskedLM.from_pretrained(model_id)
        model = NeuralSparseModel(bert)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # model = SentenceTransformer(model_id)

        if model_name is None:
            model_name = str(model_id.split("/")[-1] + ".pt")

        model_path = os.path.join(self.folder_path, model_name)

        if save_json_folder_path is None:
            save_json_folder_path = self.folder_path

        if model_output_path is None:
            model_output_path = self.folder_path

        if zip_file_name is None:
            zip_file_name = str(model_id.split("/")[-1] + ".zip")
        zip_file_path = os.path.join(model_output_path, zip_file_name)

        # handle when model_max_length is unproperly defined in model's tokenizer (e.g. "intfloat/e5-small-v2")
        # (See PR #219 and https://github.com/huggingface/transformers/issues/14561 for more context)
        # if model.tokenizer.model_max_length > model.get_max_seq_length():
        #    model.tokenizer.model_max_length = model.get_max_seq_length()
        #    print(
        #        f"The model_max_length is not properly defined in tokenizer_config.json. Setting it to be {model.tokenizer.model_max_length}"
        #    )

        # save tokenizer.json in save_json_folder_name
        bert.save_pretrained(save_json_folder_path)
        self.tokenizer.save_pretrained(save_json_folder_path)
        super()._fill_null_truncation_field(
            save_json_folder_path, self.tokenizer.model_max_length
        )

        # convert to pt format will need to be in cpu,
        # set the device to cpu, convert its input_ids and attention_mask in cpu and save as .pt format
        device = torch.device("cpu")
        cpu_model = model.to(device)
        features = self.tokenizer(
            sentences,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=512,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors="pt",
        ).to(device)

        compiled_model = torch.jit.trace(cpu_model, dict(features), strict=False)
        torch.jit.save(compiled_model, model_path)
        print("model file is saved to ", model_path)

        # zip model file along with self.tokenizer.json (and license file) as output
        with ZipFile(str(zip_file_path), "w") as zipObj:
            zipObj.write(
                model_path,
                arcname=str(model_name),
            )
            zipObj.write(
                os.path.join(save_json_folder_path, "tokenizer.json"),
                arcname="tokenizer.json",
            )
        if add_apache_license:
            super()._add_apache_license_to_model_zip_file(zip_file_path)

        self.torch_script_zip_file_path = zip_file_path
        print("zip file is saved to ", zip_file_path, "\n")
        return zip_file_path

    def save_as_onnx(
        self,
        model_id="opensearch-project/opensearch-neural-sparse-encoding-v1",
        model_name: str = None,
        save_json_folder_path: str = None,
        model_output_path: str = None,
        zip_file_name: str = None,
        add_apache_license: bool = False,
    ) -> str:
        """
        Download sentence transformer model directly from huggingface, convert model to onnx format,
        zip the model file and its tokenizer.json file to prepare to upload to the Open Search cluster

        :param model_id:
            sentence transformer model id to download model from sentence transformers.
            default model_id = "sentence-transformers/msmarco-distilbert-base-tas-b"
        :type model_id: string
        :param model_name:
            Optional, model name to name the model file, e.g, "sample_model.pt". If None, default takes the
            model_id and add the extension with ".pt"
        :type model_name: string
        :param save_json_folder_path:
             Optional, path to save model json file, e.g, "home/save_pre_trained_model_json/"). If None, default as
             default_folder_path from the constructor
        :type save_json_folder_path: string
        :param model_output_path:
             Optional, path to save traced model zip file. If None, default as
             default_folder_path from the constructor
        :type model_output_path: string
        :param zip_file_name:
            Optional, file name for zip file. e.g, "sample_model.zip". If None, default takes the model_id
            and add the extension with ".zip"
        :type zip_file_name: string
        :param add_apache_license:
            Optional, whether to add a Apache-2.0 license file to model zip file
        :type add_apache_license: string
        :return: model zip file path. The file path where the zip file is being saved
        :rtype: string
        """

        bert = AutoModelForMaskedLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        # model = NeuralSparseModel(bert)
        if model_name is None:
            model_name = str(model_id.split("/")[-1] + ".onnx")

        model_path = os.path.join(self.folder_path, "onnx", model_name)

        if save_json_folder_path is None:
            save_json_folder_path = self.folder_path

        if model_output_path is None:
            model_output_path = self.folder_path

        if zip_file_name is None:
            zip_file_name = str(model_id.split("/")[-1] + ".zip")

        zip_file_path = os.path.join(model_output_path, zip_file_name)

        # handle when model_max_length is unproperly defined in model's tokenizer (e.g. "intfloat/e5-small-v2")
        # (See PR #219 and https://github.com/huggingface/transformers/issues/14561 for more context)
        """
        if tokenizer.model_max_length > bert.get_max_seq_length():
            tokenizer.model_max_length = bert.get_max_seq_length()
            print(
                f"The model_max_length is not properly defined in tokenizer_config.json. Setting it to be {tokenizer.model_max_length}"
            )
        """

        # save tokenizer.json in output_path
        bert.save_pretrained(save_json_folder_path)
        tokenizer.save_pretrained(save_json_folder_path)
        self._fill_null_truncation_field(
            save_json_folder_path, tokenizer.model_max_length
        )

        convert(
            framework="pt",
            model=model_id,
            output=Path(model_path),
            opset=15,
        )

        print("model file is saved to ", model_path)
        # zip model file along with tokenizer.json (and license file) as output
        with ZipFile(str(zip_file_path), "w") as zipObj:
            zipObj.write(
                model_path,
                arcname=str(model_name),
            )
            zipObj.write(
                os.path.join(save_json_folder_path, "tokenizer.json"),
                arcname="tokenizer.json",
            )
        if add_apache_license:
            super()._add_apache_license_to_model_zip_file(zip_file_path)

        self.onnx_zip_file_path = zip_file_path
        print("zip file is saved to ", zip_file_path, "\n")
        return zip_file_path

    def _get_model_description_from_readme_file(self, readme_file_path) -> str:
        """
        Get description of the model from README.md file in the model folder
        after the model is saved in local directory

        See example here:
        https://huggingface.co/sentence-transformers/msmarco-distilbert-base-tas-b/blob/main/README.md)

        This function assumes that the README.md has the following format:

        # sentence-transformers/msmarco-distilbert-base-tas-b
        This is [ ... further description ... ]

        # [ ... Next section ...]
        ...

        :param readme_file_path: Path to README.md file
        :type readme_file_path: string
        :return: Description of the model
        :rtype: string
        """
        readme_data = MarkDownFile.read_file(readme_file_path)

        # Find the description section
        start_str = f"\n# {self.model_id}"
        start = readme_data.find(start_str)
        if start == -1:
            model_name = self.model_id.split("/")[1]
            start_str = f"\n# {model_name}"
            start = readme_data.find(start_str)
        end = readme_data.find("\n#", start + len(start_str))

        # If we cannot find the scope of description section, raise error.
        if start == -1 or end == -1:
            assert False, "Cannot find description in README.md file"

        # Parse out the description section
        description = readme_data[start + len(start_str) + 1 : end].strip()
        description = description.split("\n")[0]

        # Remove hyperlink and reformat text
        description = re.sub(r"\(.*?\)", "", description)
        description = re.sub(r"[\[\]]", "", description)
        description = re.sub(r"\*", "", description)

        # Remove unnecessary part if exists (i.e. " For an introduction to ...")
        # (Found in https://huggingface.co/sentence-transformers/multi-qa-mpnet-base-dot-v1/blob/main/README.md)
        unnecessary_part = description.find(" For an introduction to")
        if unnecessary_part != -1:
            description = description[:unnecessary_part]

        return description

    def _generate_default_model_description(self, embedding_dimension) -> str:
        """
        Generate default model description of the model based on embedding_dimension

        ::param embedding_dimension: Embedding dimension of the model.
        :type embedding_dimension: int
        :return: Description of the model
        :rtype: string
        """
        print(
            "Using default description from embedding_dimension instead (You can overwrite this by specifying description parameter in make_model_config_json function"
        )
        description = f"This is a neural sparse model: It maps sentences & paragraphs to a {embedding_dimension} dimensional dense vector space."
        return description

    def make_model_config_json(
        self,
        model_name: str = None,
        version_number: str = 1,
        model_format: str = "TORCH_SCRIPT",
    ) -> str:
        folder_path = self.folder_path
        if model_name is None:
            model_name = self.model_id
        config = AutoformerConfig.from_pretrained(self.model_id)
        model_config_content = {
            "name": model_name,
            "version": version_number,
            "model_format": model_format,
            "functino_name": "SPARSE_ENCODING",
            "model_config": {
                "model_type": config.model_type,
                "embedding_dimension": 768,
                "framework_type": "sentence_transformers",
            },
        }
        model_config_file_path = os.path.join(
            folder_path, "ml-commons_model_config.json"
        )
        os.makedirs(os.path.dirname(model_config_file_path), exist_ok=True)
        with open(model_config_file_path, "w") as file:
            json.dump(model_config_content, file, indent=4)
        print(
            "ml-commons_model_config.json file is saved at : ", model_config_file_path
        )
        return model_config_file_path

    def get_bert(self):
        return AutoModelForMaskedLM.from_pretrained(self.model_id)

    def get_model(self):
        return NeuralSparseModel(self.get_bert(), self.tokenizer)

    def save(self, path):
        bert = AutoModelForMaskedLM.from_pretrained(self.model_id)
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        bert.save_pretrained(path)
        tokenizer.save_pretrained(path)

    def post_process(self):
        pass

    def pre_process(self):
        pass

    def get_tokenizer(self):
        return self.tokenizer

    def process_queries(self, queries):
        return self.get_model().process_queries(queries)


class NeuralSparseModel(torch.nn.Module):
    def __init__(self, bert, tokenizer=None):
        super().__init__()
        self.bert = bert
        if tokenizer is not None:
            self.tokenizer = tokenizer
            self.special_token_ids = [
                tokenizer.vocab[token]
                for token in tokenizer.special_tokens_map.values()
            ]
            self.id_to_token = ["" for _ in range(len(tokenizer.vocab))]
            for token, idx in tokenizer.vocab.items():
                self.id_to_token[idx] = token

    def forward(self, input: dict[str, torch.Tensor]):
        return self.bert(**input)[0]

    def get_sparse_vector(self, feature):
        output = self.forward(feature)
        values, _ = torch.max(output * feature["attention_mask"].unsqueeze(-1), dim=1)
        values = torch.log(1 + torch.relu(values))
        values[:, self.special_token_ids] = 0
        return values

    def transform_sparse_vector_to_dict(self, sparse_vector):
        tokens = [
            self.id_to_token[i]
            for i in torch.nonzero(sparse_vector, as_tuple=True)[1].tolist()
        ]
        return {
            token: weight.item()
            for token, weight in zip(
                tokens, sparse_vector[sparse_vector.nonzero(as_tuple=True)]
            )
        }

    def process_queries(self, queries):
        features = self.tokenizer(
            queries,
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=False,
        )
        sparse_vector = self.get_sparse_vector(features)
        sparse_dict = self.transform_sparse_vector_to_dict(sparse_vector)
        return sparse_dict
